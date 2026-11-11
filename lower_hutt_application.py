
"""
Lower Hutt profile-entropy active-learning application; see the following comments and README.md for usage.

Usage: 
    From this directory, install requirements.txt, then run python lower_hutt_application.py --jobs 1.

Quick check: 
    python lower_hutt_application.py --n-steps 2 --dz 0.5 --grid-spacing 50 --output-dir outputs/quick_check

Options: Run python lower_hutt_application.py --help to change data paths, depth/grid spacing, or acquisition count.

Script workflow：
    Step 1: Read the location table and CPT workbooks from data/@Lower Hutt (town hall).
    Step 2: Select six initial profiles by centroid-first maximin distance and fit independent Matérn 3/2 POD-GPR.
    Step 3: Score profile entropy, select one candidate, add its measured profile, refit, and record metrics (18 additions by default).
    Step 4: Export candidate scores, selection history, and all-point R2/RMSE; metrics include training points.
    Step 5: Fit the final model and predict the rectangular field (x=-20..80 m, y=-60..40 m, depth=0..10 m by default).
    Step 6: Save mean/median, standard deviation, and interval bounds; draw 3-D fields and cross sections near x=30 m and y=-10 m.

Outputs: 
    Saved under outputs/lower_hutt by default.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from cpt_pod_gpr import (build_point_dict, select_points_by_ids, normalize_point_name,
    point_sort_key, load_cpt_directory, build_z_grid, compute_profile_r2)
from al_pod_gpr import (ALConfig, fit_current_model, choose_uniform_initial_train_ids,
    compute_profile_truth, compute_profile_rmse, compute_pool_selection_table)
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import time

ROOT = Path(__file__).resolve().parent

# ===== 1. Data loading and configuration =====
# Read site workbooks and create the settings used for each model fit.

def load_site_points(site_dir: Path, depth_min: float, depth_max: float):
    """Load the site location table and CPT profiles from XLSX workbooks."""
    points = load_cpt_directory(
        str(site_dir),
        locations_filename="locations.xlsx",
        cpt_glob="*.xlsx",
        depth_clip=(float(depth_min), float(depth_max)),
    )
    return sorted(points, key=lambda p: point_sort_key(p.name))


def make_al_config(
    acquisition_mode: str,
    selection_strategy: str,
    n_restarts_optimizer: int,
    depth_min: float,
    depth_max: float,
    dz: float,
) -> ALConfig:
    return ALConfig(
        z_min=float(depth_min),
        z_max=float(depth_max),
        dz=float(dz),
        ci_level=0.95,
        acquisition_mode=str(acquisition_mode),
        selection_strategy=str(selection_strategy),
        eval_prediction_kind="truncated",
        acquisition_prediction_kind="original",
        value_transform="log1p",
        n_restarts_optimizer=int(n_restarts_optimizer),
    )


# ===== 2. Initial design and selection/metric plots =====
# Build a spatial initial design and visualize acquisition order and all-point errors.

def point_xy_table(points) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": [int(float(p.name)) for p in points],
            "x_m": [float(p.x) for p in points],
            "y_m": [float(p.y) for p in points],
        }
    )


def choose_boundary_center_initial_train_ids(points, n_select: int) -> list[str]:
    if int(n_select) != 9:
        return choose_uniform_initial_train_ids(points, int(n_select))
    xy = np.asarray([[float(p.x), float(p.y)] for p in points], dtype=float)
    point_ids = [normalize_point_name(p.name) for p in points]
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    targets = [
        (xmin, ymin), (xmid, ymin), (xmax, ymin),
        (xmin, ymid), (xmid, ymid), (xmax, ymid),
        (xmin, ymax), (xmid, ymax), (xmax, ymax),
    ]
    selected: list[int] = []
    for target in targets:
        target_xy = np.asarray(target, dtype=float)
        candidates = [i for i in range(len(points)) if i not in selected]
        selected.append(
            min(
                candidates,
                key=lambda i: (float(np.sum((xy[i] - target_xy) ** 2)), point_sort_key(point_ids[i])),
            )
        )
    return sorted([point_ids[i] for i in selected], key=point_sort_key)


def plot_sequence(ax, all_points, initial_points, selection_df: pd.DataFrame, title: str) -> None:
    all_xy = point_xy_table(all_points)
    ax.scatter(all_xy["x_m"], all_xy["y_m"], s=8, c="0.78", edgecolors="none", zorder=1)
    ax.scatter(
        [float(p.x) for p in initial_points],
        [float(p.y) for p in initial_points],
        s=38,
        c="#d73027",
        edgecolors="white",
        linewidths=0.7,
        marker="o",
        label="Initial",
        zorder=3,
    )
    if not selection_df.empty:
        ax.scatter(
            selection_df["x_m"],
            selection_df["y_m"],
            s=42,
            c="#d73027",
            edgecolors="white",
            linewidths=0.7,
            marker="s",
            label="Added",
            zorder=4,
        )
        for _, row in selection_df.iterrows():
            ax.text(
                float(row["x_m"]),
                float(row["y_m"]),
                str(int(row["step"])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=5,
            )
        xy = selection_df[["x_m", "y_m"]].to_numpy(dtype=float)
        for idx in range(len(xy) - 1):
            ax.annotate(
                "",
                xy=xy[idx + 1],
                xytext=xy[idx],
                arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.0, shrinkA=6, shrinkB=6),
                zorder=4,
            )

    xmin, xmax = float(all_xy["x_m"].min()), float(all_xy["x_m"].max())
    ymin, ymax = float(all_xy["y_m"].min()), float(all_xy["y_m"].max())
    ax.set_xlim(xmin - 0.75, xmax + 0.75)
    ax.set_ylim(ymin - 0.75, ymax + 0.75)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, color="0.88", linewidth=0.45)


def plot_metrics(ax, metrics_df: pd.DataFrame) -> None:
    x = metrics_df["iteration"].to_numpy(dtype=int)
    r2 = metrics_df["mean_profile_r2_all_points"].to_numpy(dtype=float)
    rmse = metrics_df["mean_profile_rmse_all_points"].to_numpy(dtype=float)
    ax2 = ax.twinx()
    line1 = ax.plot(x, r2, color="#1f78b4", marker="o", markersize=3.5, linewidth=1.3, label=r"$R^2$")
    line2 = ax2.plot(x, rmse, color="#d95f02", marker="s", markersize=3.5, linewidth=1.3, label="RMSE")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel(r"Mean profile $R^2$", color="#1f78b4")
    ax2.set_ylabel("Mean profile RMSE (MPa)", color="#d95f02")
    ax.tick_params(axis="y", labelcolor="#1f78b4")
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="best", frameon=False, fontsize=8)


# ===== 3. Active-learning loop =====
# Fit, score, select, and record one measured profile per iteration.

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def evaluate_field_metrics(model, points, z_grid: np.ndarray, cfg) -> tuple[float, float]:
    """Average profile R2 and RMSE over all points, including training points."""
    r2_values: list[float] = []
    rmse_values: list[float] = []
    for point in points:
        pred = model.predict_profile(float(point.x), float(point.y), ci_level=cfg.ci_level, kind=cfg.eval_prediction_kind)
        truth = compute_profile_truth(point, z_grid)
        r2 = compute_profile_r2(np.asarray(pred.mean, dtype=float), truth)
        rmse = compute_profile_rmse(np.asarray(pred.mean, dtype=float), truth)
        if np.isfinite(r2):
            r2_values.append(float(r2))
        if np.isfinite(rmse):
            rmse_values.append(float(rmse))
    return float(np.mean(r2_values)), float(np.mean(rmse_values))


def run_strategy_with_acquisition_history(
    points,
    point_map,
    initial_train_ids: list[str],
    acquisition_mode: str,
    depth_min: float,
    depth_max: float,
    dz: float,
    n_steps: int,
    n_restarts_optimizer: int,
    acquisition_n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = make_al_config(
        acquisition_mode=acquisition_mode,
        selection_strategy="active_learning",
        n_restarts_optimizer=n_restarts_optimizer,
        depth_min=depth_min,
        depth_max=depth_max,
        dz=dz,
    )
    if acquisition_mode == "maximum_coverage":
        cfg.selection_strategy = "coverage_maximin"
    cfg.acquisition_n_jobs = int(acquisition_n_jobs)
    z_grid = build_z_grid(depth_min, depth_max, dz)
    current_train_ids = list(initial_train_ids)
    pool_ids = sorted(
        [normalize_point_name(p.name) for p in points if normalize_point_name(p.name) not in set(current_train_ids)],
        key=point_sort_key,
    )
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    acquisition_rows: list[dict] = []

    for iteration in range(int(n_steps) + 1):
        iter_start = time.perf_counter()
        log(f"{acquisition_mode}: iteration {iteration}/{n_steps}, fitting model with {len(current_train_ids)} profiles")
        train_points = select_points_by_ids(point_map, current_train_ids)
        model = fit_current_model(train_points, cfg)
        log(f"{acquisition_mode}: iteration {iteration}, evaluating field metrics at {len(points)} site points")
        mean_r2, mean_rmse = evaluate_field_metrics(model, points, z_grid, cfg)
        metric_rows.append({
            "iteration": iteration,
            "n_train": len(current_train_ids),
            "n_pod_modes": int(model.basis.n_modes),
            "pod_cumulative_explained_variance": float(
                model.basis.cumulative_explained_variance_ratio[-1]
            ),
            "mean_profile_r2_all_points": mean_r2,
            "mean_profile_rmse_all_points": mean_rmse,
        })

        if iteration >= int(n_steps) or not pool_ids:
            log(
                f"{acquisition_mode}: iteration {iteration} complete in "
                f"{time.perf_counter() - iter_start:.1f}s; mean R2={mean_r2:.6f}, "
                f"RMSE={mean_rmse:.6f}"
            )
            break

        pool_points = select_points_by_ids(point_map, pool_ids)
        log(f"{acquisition_mode}: iteration {iteration}, scoring all {len(pool_points)} remaining candidates")
        table = compute_pool_selection_table(
            model=model,
            train_points=train_points,
            pool_points=pool_points,
            cfg=cfg,
        )
        next_id = normalize_point_name(table[0]["candidate_point"]) if table and iteration < int(n_steps) else None
        for rank, item in enumerate(table, start=1):
            candidate_id = normalize_point_name(item["candidate_point"])
            point = point_map[candidate_id]
            acquisition_rows.append({
                "acquisition_mode": acquisition_mode,
                "iteration": iteration,
                "n_train": len(current_train_ids),
                "candidate_point": int(float(candidate_id)),
                "x_m": float(point.x),
                "y_m": float(point.y),
                "acquisition_value": float(item["acquisition_value"]),
                "acquisition_rank": rank,
                "selected_next": candidate_id == next_id,
            })

        if next_id is not None:
            selected_point = point_map[next_id]
            selection_rows.append({
                "step": iteration + 1,
                "selection_strategy": cfg.selection_strategy,
                "selected_point": int(float(next_id)),
                "x_m": float(selected_point.x),
                "y_m": float(selected_point.y),
                "selection_value": float(table[0]["acquisition_value"]),
            })
            current_train_ids.append(next_id)
            pool_ids = [pid for pid in pool_ids if pid != next_id]
            log(f"{acquisition_mode}: iteration {iteration}, selected point {next_id}")
        log(
            f"{acquisition_mode}: iteration {iteration} complete in {time.perf_counter() - iter_start:.1f}s; "
            f"mean R2={mean_r2:.6f}, RMSE={mean_rmse:.6f}"
        )

    return pd.DataFrame(selection_rows), pd.DataFrame(metric_rows), pd.DataFrame(acquisition_rows)


# ===== 4. Whole-field visualization =====
# Draw final prediction and interval fields and representative cross sections.

def plot_box(x, y, z, cube, name, path, train_xy):
    xx, yy = np.meshgrid(x, y)
    norm = colors.Normalize(vmin=float(np.nanmin(cube)), vmax=float(np.nanmax(cube)))
    cmap = plt.get_cmap('turbo' if name != '95% CI width' else 'viridis')
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    style = dict(linewidth=0, shade=False, antialiased=False)
    ax.plot_surface(xx, yy, np.full_like(xx, z[0]), facecolors=cmap(norm(cube[:,:,0])), **style)
    xf = np.tile(x, (len(z),1)); zf = np.tile(z[:,None], (1,len(x)))
    ax.plot_surface(xf, np.full_like(xf,y[0]), zf, facecolors=cmap(norm(cube[0,:,:].T)), **style)
    ys = np.tile(y,(len(z),1)); zs = np.tile(z[:,None],(1,len(y)))
    ax.plot_surface(np.full_like(ys,x[-1]),ys,zs,facecolors=cmap(norm(cube[:,-1,:].T)),**style)
    ax.scatter(train_xy[:,0],train_xy[:,1],np.full(len(train_xy),z[0]),c='red',s=14,depthshade=False)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='Depth (m)',title=name)
    ax.set_zlim(z[-1], z[0]); ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1,1,.55)); fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.65,label='Cone resistance (MPa)')
    fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig)


def plot_cross_sections(x,y,z,fields,out):
    ix = int(np.argmin(abs(x-30))); iy = int(np.argmin(abs(y+10)))
    for name, cube in fields.items():
        fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax, horizontal, values, label, coord in ((axes[0],y,cube[:,ix,:].T,'y',x[ix]),(axes[1],x,cube[iy,:,:].T,'x',y[iy])):
            p=ax.pcolormesh(horizontal,z,values,shading='auto',cmap='turbo')
            ax.invert_yaxis(); ax.set(xlabel=f'{label} (m)',ylabel='Depth (m)',title=f'{name}: {"x" if label=="y" else "y"} = {coord:g} m')
            fig.colorbar(p,ax=ax,label='Cone resistance (MPa)')
        fig.savefig(out / f'{name}_cross_sections.png',dpi=200); plt.close(fig)


# ===== 5. The main entry point =====
# Parse options, run active learning, predict the final grid, and save outputs.

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "data" / "@Lower Hutt (town hall)", help="Folder containing locations.xlsx and the CPT workbooks.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "lower_hutt", help="Directory for result tables, prediction arrays, and figures.")
    parser.add_argument("--n-steps", type=int, default=18, help="Number of profiles added after the initial design.")
    parser.add_argument("--initial-train-size", type=int, default=6, help="Number of initially sampled profiles.")
    parser.add_argument("--depth-min", type=float, default=0.0, help="Minimum model depth in metres.")
    parser.add_argument("--depth-max", type=float, default=10.0, help="Maximum model depth in metres.")
    parser.add_argument("--dz", type=float, default=0.1, help="Model depth spacing in metres.")
    parser.add_argument("--grid-spacing", type=float, default=2.0, help="Horizontal prediction grid spacing in metres.")
    parser.add_argument("--n-restarts-optimizer", type=int, default=0, help="Additional GP optimizer starts.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel candidate-scoring threads.")
    parser.add_argument("--x-min", type=float, default=-20.0, help="Prediction grid minimum x coordinate.")
    parser.add_argument("--x-max", type=float, default=80.0, help="Prediction grid maximum x coordinate.")
    parser.add_argument("--y-min", type=float, default=-60.0, help="Prediction grid minimum y coordinate.")
    parser.add_argument("--y-max", type=float, default=40.0, help="Prediction grid maximum y coordinate.")
    args = parser.parse_args()
    if args.n_steps < 0 or args.dz <= 0 or args.grid_spacing <= 0 or args.depth_max <= args.depth_min or args.x_max <= args.x_min or args.y_max <= args.y_min:
        parser.error("Invalid iteration count, depth or grid bounds")
    return args


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Read measured profiles and construct the initial spatial design.
    points = load_site_points(args.site_dir, args.depth_min, args.depth_max)
    point_map = build_point_dict(points)
    initial = choose_boundary_center_initial_train_ids(points, args.initial_train_size)
    if args.n_steps > len(points) - len(initial):
        raise ValueError("More acquisition steps than available candidate points.")

    # Each acquisition uses the current model; only the selected profile is added to the training set.
    sequence, metrics, history = run_strategy_with_acquisition_history(
        points, point_map, initial, "profile_entropy_gain",
        args.depth_min, args.depth_max, args.dz, args.n_steps,
        args.n_restarts_optimizer, args.jobs,
    )
    pd.DataFrame({"point_id": initial}).to_csv(out / "initial_points.csv", index=False)
    sequence.to_csv(out / "selection_sequence.csv", index=False)
    metrics.to_csv(out / "metrics_all_points.csv", index=False)
    history.to_csv(out / "candidate_scores.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_sequence(axes[0], points, select_points_by_ids(point_map, initial), sequence, "Selection sequence")
    plot_metrics(axes[1], metrics)
    fig.tight_layout()
    fig.savefig(out / "active_learning.png", dpi=200)
    plt.close(fig)

    # Fit the final training set and predict the complete rectangular output grid.
    selected = [] if sequence.empty else [normalize_point_name(value) for value in sequence["selected_point"]]
    train = select_points_by_ids(point_map, initial + selected)
    cfg = make_al_config("profile_entropy_gain", "active_learning", args.n_restarts_optimizer, args.depth_min, args.depth_max, args.dz)
    model = fit_current_model(train, cfg)
    x = np.arange(args.x_min, args.x_max + 0.5 * args.grid_spacing, args.grid_spacing)
    y = np.arange(args.y_min, args.y_max + 0.5 * args.grid_spacing, args.grid_spacing)
    z = np.asarray(model.basis.z_grid)
    shape = (len(y), len(x), len(z))
    mean = np.empty(shape)
    std = np.empty(shape)
    lower = np.empty(shape)
    upper = np.empty(shape)
    for j, yq in enumerate(y):
        print(f"Predicting field row {j + 1}/{len(y)}", flush=True)
        for i, xq in enumerate(x):
            pred = model.predict_profile(float(xq), float(yq), ci_level=cfg.ci_level, kind=cfg.eval_prediction_kind)
            mean[j, i], std[j, i], lower[j, i], upper[j, i] = pred.mean, pred.std, pred.ci_lo, pred.ci_hi

    # Cubes use axis order (y, x, depth); interval bounds are truncated predictive quantiles.
    np.savez_compressed(
        out / "field_predictions.npz", x_m=x, y_m=y, depth_m=z,
        mean_mpa=mean, std_mpa=std, ci95_lower_mpa=lower, ci95_upper_mpa=upper,
    )
    train_xy = np.array([[point.x, point.y] for point in train])
    fields = {"mean": mean, "ci95_lower": lower, "ci95_upper": upper, "ci95_width": upper - lower}
    for key, cube in fields.items():
        title = "95% CI width" if key == "ci95_width" else key
        plot_box(x, y, z, cube, title, out / f"{key}_field.png", train_xy)
    plot_cross_sections(x, y, z, fields, out)
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (out / "run_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print("Results:", out)


if __name__ == "__main__":
    main()
