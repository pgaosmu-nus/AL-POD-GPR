"""
CPT data loading, POD fitting, and independent Matérn 3/2 coefficient GPR.

Usage:
    Import this module from another Python script or notebook. See lower_hutt_application.py for a complete application.

Main functions and configurations:
    1. load_cpt_directory() and build_z_grid() read site data and define the shared depth grid.
    2. BasisConfig and CoefficientGPConfig specify the POD and coefficient GPR settings.
    3. fit_cpt_basis_gp_from_points() fits a CPTBasisGPModel; CPTBasisGPModel.predict_profile() reconstructs profiles and their pointwise intervals.
    See the corresponding definitions below for individual parameter meanings.

Methodology:
    1. Interpolate training profiles onto a common depth grid and apply the transform g.
    2. Fit a POD basis: g(q(x, y, z)) ~= mean_profile(z) + sum_i lambda_i(x, y) * phi_i(z).
    3. Fit an independent Matérn 3/2 GP to each coefficient field lambda_i(x, y).
    4. Reconstruct profiles, include POD residual variance, and calculate original-scale predictions and intervals.
    The transform g above is the identity or log1p; log1p clips negative inputs to zero.
"""
from __future__ import annotations

import glob
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler


# ===== 1. Data structures and configuration =====
# Store CPT profiles, POD results, GP settings, and prediction outputs.

@dataclass
class CPTPoint:
    name: str  # CPT identifier.
    x: float  # Horizontal x coordinate.
    y: float  # Horizontal y coordinate.
    z: np.ndarray  # Measured depths.
    qc: np.ndarray  # Corresponding cone resistances.


@dataclass
class BasisConfig:
    z_grid: np.ndarray  # Shared depth grid.
    explained_variance_target: float = 0.995  # Variance target, e.g., the cumulative energy fraction limit.
    max_modes: Optional[int] = None  # Mode cap; None retains the variance target.
    value_transform: str = "log1p"  # "none" or log1p.
    center_profiles: bool = True  # Subtract the mean profile before POD.


@dataclass
class CoefficientGPConfig:
    n_restarts_optimizer: int = 1  # Additional optimizer starts.
    ci_level: float = 0.95  # Default prediction interval probability.
    independent_white_noise_bounds: Tuple[float, float] = (1e-6, 1e0)  # Independent-GP noise bounds.
    independent_length_scale_bounds: Tuple[float, float] = (1e-2, 1e2)  # Independent-GP length-scale bounds.


@dataclass
class PODBasis:
    z_grid: np.ndarray
    mean_profile_transformed: np.ndarray
    basis_functions: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    residual_variance_transformed: np.ndarray
    summary: BasisFitSummary
    value_transform: str

    @property
    def n_modes(self) -> int:
        return int(self.basis_functions.shape[1])


@dataclass
class BasisFitSummary:
    n_points: int
    n_depths: int
    n_modes: int
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    reconstruction_rmse_transformed: float
    reconstruction_rmse_original: float
    max_modes_cap_reached: bool


@dataclass
class ProfilePrediction:
    z: np.ndarray  # Shared prediction depths.
    mean: np.ndarray  # Original-scale prediction centre after optional truncation.
    std: np.ndarray  # Original-scale approximate standard deviation after optional truncation.
    ci_lo: np.ndarray  # Lower pointwise prediction interval bound.
    ci_hi: np.ndarray  # Upper pointwise prediction interval bound.
    mean_raw: np.ndarray  # Back-transformed centre before optional truncation.
    std_raw: np.ndarray  # Approximate original-scale standard deviation before truncation.
    kind: str  # Requested prediction representation.
    ci_level: float  # Probability used to calculate the interval bounds.
    coeff_mean: np.ndarray  # Predicted POD coefficients, with shape (n_modes,).
    coeff_cov: np.ndarray  # Coefficient covariance, with shape (n_modes, n_modes).

# ===== 2. Data loading and point selection =====
# Read XLS/XLSX profiles (depth and cone resistance in the first two columns) and match their filenames to the location table. See the data directory for an example.

def _safe_read_excel(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception as e:
        msg = f"Failed to read Excel: {path}\nError: {e}\n"
        if path.lower().endswith(".xls"):
            msg += (
                "Hint: .xls may require 'xlrd'. Try:\n"
                "  pip install xlrd\n"
                "Or convert locations.xls to locations.xlsx.\n"
            )
        raise RuntimeError(msg)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip().lower()


def normalize_point_name(name: Any) -> str:
    if name is None:
        raise ValueError("Point name cannot be None.")
    s = str(name).strip().lower()
    if s == "" or s == "nan":
        raise ValueError(f"Invalid point name: {name!r}")
    try:
        f = float(s)
        if math.isfinite(f) and float(f).is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def point_sort_key(name: Any) -> Tuple[int, Any]:
    s = normalize_point_name(name)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def load_cpt_directory(
    data_dir: str,  # Data directory.
    locations_filename: str = "locations.xls",  # Workbook containing point IDs and x/y coordinates.
    cpt_glob: str = "*.xls",  # Filename pattern for individual CPT profiles.
    max_points_per_borehole: Optional[int] = None,  # Limit samples; None keeps all.
    depth_clip: Optional[Tuple[float, float]] = None,  # Inclusive depth range, optional.
) -> List[CPTPoint]:
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    loc_path = os.path.join(data_dir, locations_filename)
    if not os.path.exists(loc_path):
        alt = None
        if locations_filename.lower().endswith(".xls"):
            alt = os.path.join(data_dir, locations_filename[:-4] + ".xlsx")
        elif locations_filename.lower().endswith(".xlsx"):
            alt = os.path.join(data_dir, locations_filename[:-5] + ".xls")
        if alt and os.path.exists(alt):
            loc_path = alt
        else:
            raise FileNotFoundError(f"locations file not found: {loc_path}")

    loc_df = _normalize_columns(_safe_read_excel(loc_path))
    x_col = _find_col(loc_df, ["x", "x(m)", "easting", "east", "lon", "longitude"])
    y_col = _find_col(loc_df, ["y", "y(m)", "northing", "north", "lat", "latitude"])
    if x_col is None or y_col is None:
        raise ValueError(f"locations file must contain x and y columns. Found columns: {list(loc_df.columns)}")

    file_col = _find_col(loc_df, ["filename", "file", "filepath"])
    id_col = _find_col(loc_df, ["id", "name", "point", "point_id", "cpt", "cpt_id", "no", "no."])

    stem_to_xy: Dict[str, Tuple[float, float]] = {}
    if file_col is not None:
        for _, r in loc_df.iterrows():
            f = str(r[file_col])
            if f and f.lower() != "nan":
                stem_to_xy[normalize_point_name(_stem(f))] = (float(r[x_col]), float(r[y_col]))
    if id_col is not None:
        for _, r in loc_df.iterrows():
            raw_id = r[id_col]
            if pd.isna(raw_id):
                continue
            key = normalize_point_name(raw_id)
            if key not in stem_to_xy:
                stem_to_xy[key] = (float(r[x_col]), float(r[y_col]))

    pattern = os.path.join(data_dir, cpt_glob)
    cpt_files = sorted(glob.glob(pattern))
    cpt_files = [f for f in cpt_files if not os.path.basename(f).startswith("~$")]
    cpt_files = [f for f in cpt_files if os.path.abspath(f) != os.path.abspath(loc_path)]
    if len(cpt_files) == 0:
        raise FileNotFoundError(f"No CPT files found with pattern: {pattern}")

    points: List[CPTPoint] = []
    order_match = len(stem_to_xy) == 0 and len(loc_df) == len(cpt_files)
    if order_match:
        warnings.warn(
            "locations file has no identifiable id/filename column. Falling back to matching by "
            "sorted order of CPT files vs locations rows. This is not recommended.",
            stacklevel=2,
        )

    for idx, fpath in enumerate(cpt_files):
        name = normalize_point_name(_stem(fpath))
        if order_match:
            r = loc_df.iloc[idx]
            x, y = float(r[x_col]), float(r[y_col])
        else:
            if name not in stem_to_xy:
                raise KeyError(
                    f"Cannot match CPT file '{os.path.basename(fpath)}' to locations. "
                    f"Expected a locations entry with key '{name}'."
                )
            x, y = stem_to_xy[name]

        df = _safe_read_excel(fpath)
        if df.shape[1] < 2:
            raise ValueError(f"CPT file must have at least 2 columns (depth, qc): {fpath}")

        z_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        qc_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        m = np.isfinite(z_raw) & np.isfinite(qc_raw)
        z = z_raw[m].astype(float)
        qc = qc_raw[m].astype(float)

        order = np.argsort(z)
        z = z[order]
        qc = qc[order]

        if depth_clip is not None:
            zmin, zmax = depth_clip
            mm = (z >= zmin) & (z <= zmax)
            z = z[mm]
            qc = qc[mm]

        if max_points_per_borehole is not None and len(z) > max_points_per_borehole:
            keep_idx = np.linspace(0, len(z) - 1, max_points_per_borehole).astype(int)
            z = z[keep_idx]
            qc = qc[keep_idx]

        points.append(CPTPoint(name=name, x=x, y=y, z=z, qc=qc))

    return sorted(points, key=lambda p: point_sort_key(p.name))


def build_point_dict(points: Sequence[CPTPoint]) -> Dict[str, CPTPoint]:
    out: Dict[str, CPTPoint] = {}
    for p in points:
        key = normalize_point_name(p.name)
        if key in out:
            raise ValueError(f"Duplicate point id encountered: {key}")
        out[key] = p
    return out


def select_points_by_ids(points_or_dict: Sequence[CPTPoint] | Dict[str, CPTPoint], ids: Sequence[Any]) -> List[CPTPoint]:
    pmap = points_or_dict if isinstance(points_or_dict, dict) else build_point_dict(points_or_dict)
    out: List[CPTPoint] = []
    missing: List[str] = []
    for pid in ids:
        key = normalize_point_name(pid)
        if key not in pmap:
            missing.append(key)
        else:
            out.append(pmap[key])
    if missing:
        raise KeyError(f"Requested point ids not found: {sorted(set(missing), key=point_sort_key)}")
    return out

# ===== 3. Shared grid and POD fitting =====
# Interpolate and transform training profiles, then fit POD without depthwise scaling.

def build_z_grid(
    z_min: float = 0.0,  # Starting depth.
    z_max: float = 30.0,  # Upper depth limit.
    dz: float = 0.05,  # Positive depth increment.
) -> np.ndarray:
    if dz <= 0:
        raise ValueError("dz must be positive.")
    n = int(math.floor((z_max - z_min) / dz)) + 1
    return z_min + dz * np.arange(n, dtype=float)


def _interp_profile_to_grid(point: CPTPoint, z_grid: np.ndarray) -> np.ndarray:
    z_grid = np.asarray(z_grid, dtype=float).ravel()
    z = np.asarray(point.z, dtype=float).ravel()
    qc = np.asarray(point.qc, dtype=float).ravel()
    valid = np.isfinite(z) & np.isfinite(qc)
    z = z[valid]
    qc = qc[valid]
    if z.size < 2:
        return np.full_like(z_grid, np.nan, dtype=float)
    order = np.argsort(z)
    z = z[order]
    qc = qc[order]
    z, uniq_idx = np.unique(z, return_index=True)
    qc = qc[uniq_idx]
    if z.size < 2:
        return np.full_like(z_grid, np.nan, dtype=float)
    out = np.full_like(z_grid, np.nan, dtype=float)
    mask = (z_grid >= z.min()) & (z_grid <= z.max())
    if np.any(mask):
        out[mask] = np.interp(z_grid[mask], z, qc)
    return out


def _transform_values(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mode_norm = str(mode).strip().lower()
    if mode_norm == "none":
        return arr
    if mode_norm == "log1p":
        return np.log1p(np.maximum(arr, 0.0))
    raise ValueError(f"Unsupported value_transform: {mode!r}")


def _inverse_transform_values(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mode_norm = str(mode).strip().lower()
    if mode_norm == "none":
        return arr
    if mode_norm == "log1p":
        return np.maximum(np.expm1(arr), 0.0)
    raise ValueError(f"Unsupported value_transform: {mode!r}")


def build_profile_matrix(points: Sequence[CPTPoint], z_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z_grid = np.asarray(z_grid, dtype=float).ravel()
    if z_grid.size == 0:
        raise ValueError("z_grid must not be empty.")
    profiles: List[np.ndarray] = []
    xy_rows: List[Tuple[float, float]] = []
    for p in points:
        profile = _interp_profile_to_grid(p, z_grid)
        if not np.any(np.isfinite(profile)):
            continue
        fill_value = float(np.nanmean(profile[np.isfinite(profile)]))
        profile = np.where(np.isfinite(profile), profile, fill_value)
        profiles.append(profile.astype(float))
        xy_rows.append((float(p.x), float(p.y)))
    if not profiles:
        raise ValueError("No valid profiles remained after interpolation to the common z-grid.")
    return np.vstack(profiles).astype(float), np.asarray(xy_rows, dtype=float)


def fit_pod_basis(points: Sequence[CPTPoint], cfg: BasisConfig) -> PODBasis:
    Y_raw, _ = build_profile_matrix(points, cfg.z_grid)
    Y_t = _transform_values(Y_raw, cfg.value_transform)

    if cfg.center_profiles:
        mean_profile = np.mean(Y_t, axis=0)
    else:
        mean_profile = np.zeros(Y_t.shape[1], dtype=float)
    Y_centered = Y_t - mean_profile[None, :]

    U, singular_values, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    explained = singular_values ** 2
    explained_ratio = explained / max(float(np.sum(explained)), 1e-12)
    cumulative = np.cumsum(explained_ratio)

    target = float(np.clip(cfg.explained_variance_target, 0.0, 1.0))
    n_modes_target = int(np.searchsorted(cumulative, target, side="left") + 1)
    if cfg.max_modes is None:
        n_modes = min(max(1, n_modes_target), Vt.shape[0])
        max_modes_cap_reached = False
    else:
        n_modes = min(max(1, n_modes_target), int(cfg.max_modes), Vt.shape[0])
        max_modes_cap_reached = n_modes_target > int(cfg.max_modes)

    basis_functions = Vt[:n_modes].T.astype(float)
    scores = (U[:, :n_modes] * singular_values[:n_modes][None, :]).astype(float)

    Y_recon_t = mean_profile[None, :] + scores @ basis_functions.T
    residual_t = Y_t - Y_recon_t
    residual_var_t = np.var(residual_t, axis=0, ddof=1) if Y_t.shape[0] > 1 else np.zeros(Y_t.shape[1], dtype=float)
    residual_var_t = np.maximum(residual_var_t, 0.0)

    rmse_t = float(np.sqrt(np.mean(residual_t ** 2)))
    Y_recon_raw = _inverse_transform_values(Y_recon_t, cfg.value_transform)
    rmse_raw = float(np.sqrt(np.mean((Y_recon_raw - Y_raw) ** 2)))

    summary = BasisFitSummary(
        n_points=int(Y_raw.shape[0]),
        n_depths=int(Y_raw.shape[1]),
        n_modes=int(n_modes),
        explained_variance_ratio=explained_ratio[:n_modes].copy(),
        cumulative_explained_variance_ratio=cumulative[:n_modes].copy(),
        reconstruction_rmse_transformed=rmse_t,
        reconstruction_rmse_original=rmse_raw,
        max_modes_cap_reached=bool(max_modes_cap_reached),
    )
    return PODBasis(
        z_grid=np.asarray(cfg.z_grid, dtype=float).copy(),
        mean_profile_transformed=mean_profile.astype(float),
        basis_functions=basis_functions,
        scores=scores,
        explained_variance_ratio=explained_ratio[:n_modes].copy(),
        cumulative_explained_variance_ratio=cumulative[:n_modes].copy(),
        residual_variance_transformed=residual_var_t.astype(float),
        summary=summary,
        value_transform=str(cfg.value_transform).strip().lower(),
    )

# ===== 4. Independent coefficient GPR =====
# Fit one Matérn 3/2 GP per retained POD coefficient.

class IndependentCoefficientGPModel:
    def __init__(self, config: CoefficientGPConfig):
        self.config = config
        self.xy_scaler: Optional[StandardScaler] = None
        self.gp_models: List[GaussianProcessRegressor] = []

    def _make_kernel(self) -> object:
        # GPR uses standardized horizontal coordinates and separate x/y length scales.
        base = Matern(length_scale=[1.0, 1.0], length_scale_bounds=self.config.independent_length_scale_bounds, nu=1.5)
        return C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-3, noise_level_bounds=self.config.independent_white_noise_bounds)

    def fit(self, xy: np.ndarray, scores: np.ndarray) -> "IndependentCoefficientGPModel":
        xy = np.asarray(xy, dtype=float)
        scores = np.asarray(scores, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be (n_points, 2).")
        if scores.ndim != 2 or scores.shape[0] != xy.shape[0]:
            raise ValueError("scores must be (n_points, n_modes).")
        self.xy_scaler = StandardScaler()
        xy_scaled = self.xy_scaler.fit_transform(xy)
        self.gp_models = []
        for mode_idx in range(scores.shape[1]):
            gpr = GaussianProcessRegressor(
                kernel=self._make_kernel(),
                normalize_y=True,
                n_restarts_optimizer=int(self.config.n_restarts_optimizer),
                random_state=0,
            )
            gpr.fit(xy_scaled, scores[:, mode_idx])
            self.gp_models.append(gpr)
        return self

    def predict_coefficients(self, xy_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.xy_scaler is None or not self.gp_models:
            raise RuntimeError("IndependentCoefficientGPModel is not fitted.")
        xq = np.asarray(xy_query, dtype=float).reshape(1, 2)
        xq_scaled = self.xy_scaler.transform(xq)
        means: List[float] = []
        variances: List[float] = []
        for gpr in self.gp_models:
            mean, std = gpr.predict(xq_scaled, return_std=True)
            means.append(float(mean[0]))
            variances.append(float(max(std[0] ** 2, 0.0)))
        coeff_mean = np.asarray(means, dtype=float)
        coeff_cov = np.diag(np.asarray(variances, dtype=float))
        return coeff_mean, coeff_cov

# ===== 5. Profile reconstruction and nonnegative intervals =====
# Propagate coefficient covariance, add POD residual variance, and optionally truncate predictions at zero.

def _truncnorm_stats(mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    a = (0.0 - mu) / sigma_safe
    sf = np.maximum(norm.sf(a), 1e-15)
    lam = norm.pdf(a) / sf
    mean_t = mu + sigma_safe * lam
    var_t = sigma_safe ** 2 * (1.0 + a * lam - lam ** 2)
    var_t = np.maximum(var_t, 0.0)
    near_det = sigma <= 1e-12
    if np.any(near_det):
        mean_t = np.where(near_det, np.maximum(mu, 0.0), mean_t)
        var_t = np.where(near_det, 0.0, var_t)
    return mean_t, np.sqrt(var_t)


def _truncnorm_quantile(mu: np.ndarray, sigma: np.ndarray, p: float) -> np.ndarray:
    p = float(np.clip(p, 0.0, 1.0))
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    a = (0.0 - mu) / sigma_safe
    cdf_a = norm.cdf(a)
    sf = np.maximum(1.0 - cdf_a, 1e-15)
    u = cdf_a + p * sf
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    q = mu + sigma_safe * norm.ppf(u)
    q = np.maximum(q, 0.0)
    near_det = sigma <= 1e-12
    if np.any(near_det):
        q = np.where(near_det, np.maximum(mu, 0.0), q)
    return q


class CPTBasisGPModel:
    def __init__(
        self,
        basis: PODBasis,  
        gp_config: CoefficientGPConfig,  
        xy_train: np.ndarray, 
    ):
        self.basis = basis
        self.gp_config = gp_config
        self.xy_train = np.asarray(xy_train, dtype=float)
        self.coefficient_model: Optional[object] = None

    def fit(self) -> "CPTBasisGPModel":
        self.coefficient_model = IndependentCoefficientGPModel(self.gp_config).fit(self.xy_train, self.basis.scores)
        return self

    def _curve_variance_from_coefficients(self, coeff_cov: np.ndarray) -> np.ndarray:
        Phi = np.asarray(self.basis.basis_functions, dtype=float)
        return np.einsum("zi,ij,zj->z", Phi, coeff_cov, Phi)

    def _noise_variance_transformed(self) -> np.ndarray:
        return np.asarray(self.basis.residual_variance_transformed, dtype=float)

    def predict_profile(
        self,
        x: float,  
        y: float,  
        ci_level: Optional[float] = None,  
        kind: str = "original",  # "truncated" uses a nonnegative truncated normal approximation.
    ) -> ProfilePrediction:
        if self.coefficient_model is None:
            raise RuntimeError("Call fit() before predict_profile().")
        coeff_mean, coeff_cov = self.coefficient_model.predict_coefficients(np.array([[x, y]], dtype=float))

        mean_t = self.basis.mean_profile_transformed + self.basis.basis_functions @ coeff_mean
        var_t = self._curve_variance_from_coefficients(coeff_cov) + self._noise_variance_transformed()
        var_t = np.maximum(var_t, 0.0)
        std_t = np.sqrt(var_t)
        
        mean_raw = _inverse_transform_values(mean_t, self.basis.value_transform)
        delta = np.maximum(std_t, 1e-12)
        lo_raw = _inverse_transform_values(mean_t - delta, self.basis.value_transform)
        hi_raw = _inverse_transform_values(mean_t + delta, self.basis.value_transform)
        std_raw = 0.5 * np.maximum(hi_raw - lo_raw, 0.0)

        use_ci_level = float(self.gp_config.ci_level if ci_level is None else ci_level)
        alpha = 1.0 - use_ci_level
        if str(kind).strip().lower() == "truncated":
            mean_use, std_use = _truncnorm_stats(mean_raw, std_raw)
            ci_lo = _truncnorm_quantile(mean_raw, std_raw, alpha / 2.0)
            ci_hi = _truncnorm_quantile(mean_raw, std_raw, 1.0 - alpha / 2.0)
        else:
            zcrit = float(norm.ppf(1.0 - alpha / 2.0))
            mean_use = mean_raw
            std_use = std_raw
            ci_lo = mean_raw - zcrit * std_raw
            ci_hi = mean_raw + zcrit * std_raw
        return ProfilePrediction(
            z=self.basis.z_grid.copy(),
            mean=np.asarray(mean_use, dtype=float),
            std=np.asarray(std_use, dtype=float),
            ci_lo=np.asarray(ci_lo, dtype=float),
            ci_hi=np.asarray(ci_hi, dtype=float),
            mean_raw=np.asarray(mean_raw, dtype=float),
            std_raw=np.asarray(std_raw, dtype=float),
            kind=str(kind).strip().lower(),
            ci_level=use_ci_level,
            coeff_mean=np.asarray(coeff_mean, dtype=float),
            coeff_cov=np.asarray(coeff_cov, dtype=float),
        )

    def save(self, path: str) -> None:
        # Serialize the fitted model to the specified pickle file.
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "CPTBasisGPModel":
        # Restore a model previously written by save.
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, CPTBasisGPModel):
            raise TypeError("Loaded object is not CPTBasisGPModel.")
        return obj

# ===== 6. Public fitting interface and evaluation =====
# Fit the complete model and evaluate profile accuracy.

def fit_cpt_basis_gp_from_points(
    points: Sequence[CPTPoint],  # Training profiles for POD and coefficient GPR.
    basis_config: BasisConfig,  # Shared grid, transformation, and mode settings.
    gp_config: CoefficientGPConfig,  # Independent GPR and optimizer settings.
) -> CPTBasisGPModel:
    basis = fit_pod_basis(points, basis_config)
    _, xy = build_profile_matrix(points, basis_config.z_grid)
    return CPTBasisGPModel(basis, gp_config, xy).fit()



def compute_profile_r2(
    pred_mean: np.ndarray,
    truth: np.ndarray,
) -> float:
    pred_arr = np.asarray(pred_mean, dtype=float).ravel()
    truth_arr = np.asarray(truth, dtype=float).ravel()
    valid = np.isfinite(pred_arr) & np.isfinite(truth_arr)
    if not np.any(valid):
        return float("nan")
    y_true = truth_arr[valid]
    y_pred = pred_arr[valid]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else float("nan")
    return float(1.0 - ss_res / ss_tot)
