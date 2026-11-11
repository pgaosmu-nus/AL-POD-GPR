"""
Reusable active-learning model fitting and candidate selection.

Usage:
    Import this module from another Python script:
        This module is to configure ALConfig, fit_current_model on the training profiles, and compute_pool_selection_table on the remaining candidates.
        The calling script manages data loading, the active-learning loop, iteration history, result export, and plotting.
    See lower_hutt_application.py for a complete application.

Supported selection strategies:
    Profile-entropy gain uses cached determinant-lemma updates; 
    maximum profile standard deviation and maximin distance are also available.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import block_diag, cho_solve

from cpt_pod_gpr import (
    BasisConfig, CPTPoint, CoefficientGPConfig, build_z_grid,
    fit_cpt_basis_gp_from_points, normalize_point_name, point_sort_key,
)


# ===== 1. Active-learning configuration =====
# Settings needed for fitting and selection strategies.

@dataclass
class ALConfig:
    z_min: float = 0.0  # Starting model depth.
    z_max: float = 10.0  # Maximum model depth.
    dz: float = 0.1  # Shared depth increment.
    ci_level: float = 0.95  # Pointwise interval probability.
    max_modes: Optional[int] = None  # Optional POD mode cap.
    explained_variance_target: float = 1.0  # 1 retains all available POD modes.
    value_transform: str = "log1p"  # "log1p" or "none" before POD.
    n_restarts_optimizer: int = 1  # Additional GP optimizer starts.
    eval_prediction_kind: str = "truncated"  # Output and metric prediction representation.
    acquisition_prediction_kind: str = "original"  # Representation for maximum-standard-deviation scoring.
    acquisition_mode: str = "profile_entropy_gain"  # "profile_entropy_gain" or "max_std".
    selection_strategy: str = "active_learning"  # "active_learning" or "coverage_maximin".
    acquisition_n_jobs: int = 1  # Candidate-scoring threads.


# ===== 2. Model fitting, initial design, and evaluation =====
# Refit POD-GPR, choose a spatial initial design, and calculate prediction errors.

def fit_current_model(train_points: Sequence[CPTPoint], cfg: ALConfig):
    return fit_cpt_basis_gp_from_points(
        train_points,
        BasisConfig(z_grid=build_z_grid(cfg.z_min, cfg.z_max, cfg.dz), explained_variance_target=cfg.explained_variance_target, max_modes=cfg.max_modes, value_transform=cfg.value_transform),
        CoefficientGPConfig(n_restarts_optimizer=cfg.n_restarts_optimizer, ci_level=cfg.ci_level),
    )


def choose_uniform_initial_train_ids(points: Sequence[CPTPoint], n_select: int) -> List[str]:
    if n_select <= 0:
        raise ValueError("n_select must be positive.")
    if n_select > len(points):
        raise ValueError(f"Cannot select {n_select} initial points from only {len(points)} total points.")

    point_ids = [normalize_point_name(p.name) for p in points]
    xy = np.asarray([[float(p.x), float(p.y)] for p in points], dtype=float)
    centroid = np.mean(xy, axis=0)
    dist_to_centroid = np.sum((xy - centroid) ** 2, axis=1)
    first_idx = min(range(len(points)), key=lambda i: (dist_to_centroid[i], point_sort_key(point_ids[i])))

    selected_indices = [first_idx]
    selected_set = {first_idx}
    min_dist_sq = np.sum((xy - xy[first_idx]) ** 2, axis=1)

    while len(selected_indices) < n_select:
        best_idx = None
        best_score = -np.inf
        for idx in range(len(points)):
            if idx in selected_set:
                continue
            score = float(min_dist_sq[idx])
            if best_idx is None or score > best_score + 1e-12:
                best_idx = idx
                best_score = score
            elif abs(score - best_score) <= 1e-12 and point_sort_key(point_ids[idx]) < point_sort_key(point_ids[best_idx]):
                best_idx = idx
        if best_idx is None:
            raise RuntimeError("Failed to choose the next initial training point.")
        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        dist_sq = np.sum((xy - xy[best_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return sorted([point_ids[idx] for idx in selected_indices], key=point_sort_key)


def compute_profile_truth(point: CPTPoint, z_grid: np.ndarray) -> np.ndarray:
    if len(point.z) < 2 or len(point.qc) < 2:
        return np.full_like(np.asarray(z_grid, dtype=float), np.nan, dtype=float)
    return np.asarray(np.interp(z_grid, point.z, point.qc, left=np.nan, right=np.nan), dtype=float)


def compute_profile_rmse(pred_mean: np.ndarray, truth: np.ndarray) -> float:
    pred_arr = np.asarray(pred_mean, dtype=float).ravel()
    truth_arr = np.asarray(truth, dtype=float).ravel()
    valid = np.isfinite(pred_arr) & np.isfinite(truth_arr)
    if not np.any(valid):
        return float("nan")
    err = pred_arr[valid] - truth_arr[valid]
    return float(np.sqrt(np.mean(err ** 2)))

# ===== 3. Profile-entropy acquisition =====
# Cache each fitted model over all remaining candidate locations and evaluate covariance updates without GP refitting.

@dataclass
class FastEntropyCache:
    xy_eval_scaled: np.ndarray  # Standardized evaluation coordinates.
    profile_projected_precision: np.ndarray  # Profile precision projected onto coefficient space.
    n_eval: int  # Count of all remaining candidate locations used for entropy evaluation.
    n_modes: int  # Retained POD coefficient count.


def prepare_fast_entropy_cache(model, pool_points: Sequence[CPTPoint]) -> FastEntropyCache:
    coeff_model = getattr(model, "coefficient_model", None)
    if coeff_model is None or not hasattr(coeff_model, "gp_models") or coeff_model.xy_scaler is None:
        raise ValueError("Fast entropy acquisition requires fitted independent coefficient GPs.")
    eval_list = list(pool_points)  # Use every remaining candidate as an entropy evaluation location.
    xy_eval = np.asarray([[float(p.x), float(p.y)] for p in eval_list], dtype=float)
    xy_eval_scaled = coeff_model.xy_scaler.transform(xy_eval)
    covariances: List[np.ndarray] = []
    for gpr in coeff_model.gp_models:
        signal_kernel = gpr.kernel_.k1
        x_train = np.asarray(gpr.X_train_, dtype=float)
        k_eval_train = np.asarray(signal_kernel(xy_eval_scaled, x_train), dtype=float)
        solved_eval = cho_solve((gpr.L_, True), k_eval_train.T, check_finite=False)
        covariance = np.asarray(signal_kernel(xy_eval_scaled, xy_eval_scaled), dtype=float) - k_eval_train @ solved_eval
        covariance = 0.5 * (covariance + covariance.T)
        covariances.append(covariance)

    phi = np.asarray(model.basis.basis_functions, dtype=float)[:, :len(covariances)]
    noise = np.asarray(model._noise_variance_transformed(), dtype=float).ravel()
    if noise.size != phi.shape[0]:
        noise = np.zeros(phi.shape[0], dtype=float)
    noise = np.maximum(noise, 1e-10)
    gram = phi.T @ ((1.0 / noise)[:, None] * phi)
    gram = 0.5 * (gram + gram.T)
    h_matrix = np.kron(gram, np.eye(len(eval_list), dtype=float))
    coeff_cov = block_diag(*covariances)
    # M' (D + M C M')^-1 M = (I + H C)^-1 H.
    projected_precision = np.linalg.solve(
        np.eye(h_matrix.shape[0], dtype=float) + h_matrix @ coeff_cov,
        h_matrix,
    )
    projected_precision = 0.5 * (projected_precision + projected_precision.T)
    return FastEntropyCache(xy_eval_scaled, projected_precision, len(eval_list), len(covariances))


def _candidate_update_vectors(model, point: CPTPoint, cache: FastEntropyCache) -> List[np.ndarray]:
    coeff_model = model.coefficient_model
    x_cand = coeff_model.xy_scaler.transform(np.asarray([[float(point.x), float(point.y)]], dtype=float))
    vectors: List[np.ndarray] = []
    for gpr in coeff_model.gp_models[:cache.n_modes]:
        signal_kernel = gpr.kernel_.k1
        noise_level = float(max(gpr.kernel_.k2.noise_level, 1e-12))
        x_train = np.asarray(gpr.X_train_, dtype=float)
        k_eval_train = np.asarray(signal_kernel(cache.xy_eval_scaled, x_train), dtype=float)
        k_cand_train = np.asarray(signal_kernel(x_cand, x_train), dtype=float)
        solved_cand = cho_solve((gpr.L_, True), k_cand_train.T, check_finite=False)
        k_eval_cand = np.asarray(signal_kernel(cache.xy_eval_scaled, x_cand), dtype=float).reshape(-1)
        cross_cov = k_eval_cand - np.sum(k_eval_train * solved_cand.T, axis=1)
        cand_var = float(np.asarray(signal_kernel(x_cand, x_cand), dtype=float)[0, 0])
        cand_var -= float(np.asarray(k_cand_train @ solved_cand).ravel()[0])
        denom = max(cand_var, 0.0) + noise_level
        vectors.append(cross_cov / np.sqrt(max(denom, 1e-12)))
    return vectors


def compute_point_acquisition_fast(model, point: CPTPoint, cache: FastEntropyCache) -> float:
    vectors = _candidate_update_vectors(model, point, cache)
    if cache.profile_projected_precision is None:
        raise RuntimeError("Missing profile determinant-lemma cache.")
    update = np.zeros((cache.n_modes * cache.n_eval, cache.n_modes), dtype=float)
    for mode_index, vector in enumerate(vectors):
        start = mode_index * cache.n_eval
        update[start:start + cache.n_eval, mode_index] = vector
    small = np.eye(cache.n_modes) - update.T @ cache.profile_projected_precision @ update
    small = 0.5 * (small + small.T)
    eigvals = np.maximum(np.linalg.eigvalsh(small), 1e-12)
    return float(max(-0.5 * np.sum(np.log(eigvals)), 0.0))

# ===== 4. Candidate ranking =====
# Rank candidates by profile entropy, maximum standard deviation, or maximin distance.

def _xy_array(points: Sequence[CPTPoint]) -> np.ndarray:
    return np.asarray([[float(p.x), float(p.y)] for p in points], dtype=float)


def _coverage_maximin_table(
    train_points: Sequence[CPTPoint],
    pool_points: Sequence[CPTPoint],
) -> List[Dict[str, Any]]:
    if len(pool_points) == 0:
        return []
    xy_pool = _xy_array(pool_points)
    if len(train_points) == 0:
        centroid = np.mean(xy_pool, axis=0, keepdims=True)
        min_dist = np.sqrt(np.sum((xy_pool - centroid) ** 2, axis=1))
    else:
        xy_train = _xy_array(train_points)
        dist_sq = np.sum((xy_pool[:, None, :] - xy_train[None, :, :]) ** 2, axis=2)
        min_dist = np.sqrt(np.min(dist_sq, axis=1))
    rows = [
        {"candidate_point": str(p.name), "acquisition_value": float(score)}
        for p, score in zip(pool_points, min_dist)
    ]
    rows.sort(key=lambda r: (-r["acquisition_value"], point_sort_key(r["candidate_point"])))
    return rows


def compute_pool_acquisition_table(
    model,
    pool_points: Sequence[CPTPoint],  # Unsampled candidates to rank.
    cfg: ALConfig,
) -> List[Dict[str, Any]]:
    if not pool_points:
        return []
    mode = cfg.acquisition_mode.strip().lower()
    if mode not in {"profile_entropy_gain", "max_std"}:
        raise ValueError(f"Unsupported acquisition_mode: {cfg.acquisition_mode!r}")
    cache = None
    if mode == "profile_entropy_gain":
        cache = prepare_fast_entropy_cache(model, pool_points)

    def score(point):
        if mode == "profile_entropy_gain":
            value = compute_point_acquisition_fast(model, point, cache)
        else:
            pred = model.predict_profile(point.x, point.y, ci_level=cfg.ci_level, kind=cfg.acquisition_prediction_kind)
            std = pred.std_raw if cfg.acquisition_prediction_kind == "original" else pred.std
            valid = np.isfinite(std) & np.isfinite(pred.z)
            value = float(np.max(np.asarray(std)[valid])) if np.any(valid) else float("nan")
        return {"candidate_point": point.name, "acquisition_value": float(value)}

    n_jobs = max(1, int(cfg.acquisition_n_jobs))
    rows = [score(point) for point in pool_points] if n_jobs == 1 else Parallel(n_jobs=n_jobs, prefer="threads")(delayed(score)(point) for point in pool_points)
    rows.sort(key=lambda row: (-row["acquisition_value"], point_sort_key(row["candidate_point"])))
    return rows


def compute_pool_selection_table(
    model: Any,  # Fitted POD-GPR, no need for coverage_maxmin.
    train_points: Sequence[CPTPoint],  # Already sampled profiles.
    pool_points: Sequence[CPTPoint],  # Unsampled candidates.
    cfg: ALConfig,  # Acquisition and parallel-scoring settings.
) -> List[Dict[str, Any]]:
    strategy = cfg.selection_strategy.strip().lower()
    if strategy == "active_learning":
        return compute_pool_acquisition_table(model, pool_points, cfg)
    if strategy == "coverage_maximin":
        return _coverage_maximin_table(train_points, pool_points)
    raise ValueError(f"Unsupported selection_strategy: {cfg.selection_strategy!r}")

