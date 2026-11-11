# AL-POD-GPR

Research code and CPT data for the AL-POD-GPR framework for CPT-based site characterization. The Lower Hutt application script has a single entry point, `lower_hutt_application.py`; reusable model and selection implementations are in the root directory, `al_pod_gpr.py` and `cpt_pod_gpr.py`.

## Contents

- `data/@VG2 (virtual ground)/`: 400 virtual-ground profiles and locations.
- `data/@VG4 (virtual ground)/`: 400 virtual-ground profiles and locations.
- `data/Jaksa1995_Adelaide/`: RG1/Adelaide spreadsheets, copied from the open-source database. 
- `data/@Lower Hutt (town hall)/`: 29 CPT profiles and their location table, copied from the open-source project.
- `cpt_pod_gpr.py`: reusable CPT reader, POD basis and GPR prediction model.
- `al_pod_gpr.py`: reusable active-learning configuration, model fitting and acquisition/selection functions.
- `lower_hutt_application.py`: one application script containing the Lower Hutt data loading, active-learning loop, outputs, field prediction and plots.

## Model API

Import `load_cpt_directory`, `BasisConfig`, `CoefficientGPConfig`, and `fit_cpt_basis_gp_from_points` from `cpt_pod_gpr`, or `ALConfig`, `fit_current_model`, and `compute_pool_selection_table` from `al_pod_gpr`. `lower_hutt_application.py` demonstrates how to configure and call these functions. The data reader accepts a site directory, location workbook and CPT workbook pattern; see its function signature for details. Fit the model with fit_cpt_basis_gp_from_points(points, basis_config, gp_config). Coefficient GPs are fixed to independent Matern 3/2.

The reusable active-learning module retains profile-entropy gain, maximum profile standard deviation, and maximin distance. Profile entropy is evaluated over all remaining candidate locations. Call compute_pool_selection_table(model, train_points, pool_points, cfg) to rank candidates; the application owns the iteration counter and file outputs.

## Installation

Tested with Python 3.12 and the versions in `requirements.txt` (Windows). From this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Lower Hutt: active learning and final field

```bash
python lower_hutt_application.py --jobs 1
```

Defaults: 6 initial points selected by centroid-first spatial maximin; 18 profile-entropy acquisitions; all available POD modes; 0–10 m depth at 0.1 m; field grid x = −20–80 m and y = −60–40 m, 2 m spacing; 95% predictive interval. The model uses log1p-transformed profiles, independent Matérn coefficient GPs, coefficient nugget, and nonnegative truncated predictions. `--jobs` controls parallel candidate scoring. See `python lower_hutt_application.py --help` for options.

Quick execution check:

```bash
python lower_hutt_application.py --n-steps 1 --dz 1 --grid-spacing 50 --jobs 1 --output-dir outputs/my_smoke_test
```

Outputs are written under `outputs/lower_hutt/` by default:

- `initial_points.csv`, `selection_sequence.csv`, `candidate_scores.csv`: initial design, acquisition sequence and every scored candidate.
- `metrics_all_points.csv`, `active_learning.png`: iteration metrics and the selection/metric plots. 
- `field_predictions.npz`: `x_m`, `y_m`, `depth_m`, and `mean_mpa`, `std_mpa`, `ci95_lower_mpa`, `ci95_upper_mpa` with array dimensions `(y, x, depth)`.
- `mean_field.png`, `ci95_lower_field.png`, `ci95_upper_field.png`, `ci95_width_field.png` and corresponding cross sections: views of the final predictions. Cross sections use the grid coordinates nearest x = 30 m and y = −10 m.
- `run_parameters.json`: recorded settings.

The 95% intervals are the model's predictive quantile intervals with nonnegative truncation. Cross sections and box surfaces are rendered from the saved 3-D predictions. Grid limits describe the chosen rectangular prediction domain and can include locations outside the surveyed footprint.

## Data provenance

Virtual-ground examples and Lower Hutt relate to the benchmark published by Phoon et al. (2022); The Jaksa Adelaide data and Lower Hutt data are also catalogued at https://tc304.org/cpt_databases.html.


