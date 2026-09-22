# PM2.5 Forecasting and Spatial Interpolation for Santiago, Chile

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and data for the paper:

> **Parra, F. & Astudillo, V.** (2025). Machine Learning for PM2.5 Forecasting in Inversion-Dominated Basins: Where It Adds Value over Persistence, Validated across Santiago and Salt Lake City. *Atmospheric Pollution Research*.

## Abstract

This study contrasts machine learning for two distinct PM2.5 prediction tasks in Santiago, Chile: temporal forecasting at monitored sites and spatial interpolation to unmeasured locations. Using 7 years of data (January 2019 – November 2025) from 8 SINCA monitoring stations integrated with ERA5 meteorology and Sentinel-5P satellite observations, we find:

- **Temporal forecasting (XGBoost)**: R² = 0.76, RMSE = 14.06 μg/m³ (1-day horizon), stable across 1–7 day horizons (R² degradation ≤3%)
- **Spatial interpolation**: not successful (R² = −1.09 under Leave-One-Station-Out CV), revealing the limits of coarse-resolution (7–10 km) satellite features for intra-urban variability
- **Computational efficiency**: 13.4× speedup over ARIMA in total runtime, enabling daily retraining for operational deployment

The central finding is the *contrasting* success of temporal versus spatial machine learning approaches, with direct implications for air quality forecasting system design in cities with complex topography.

## Repository Structure

```
PM25_Santiago/
├── src/
│   ├── temporal/           # Temporal forecasting models
│   │   ├── forecasting.py  # XGBoost walk-forward validation
│   │   ├── temporal_models.py
│   │   ├── temporal_models_comparison.py  # ARIMA, Prophet, XGBoost
│   │   ├── seasonal_analysis.py
│   │   └── critical_episodes_detection.py
│   ├── spatial/            # Spatial interpolation
│   │   ├── regression_kriging.py  # Regression Kriging with LOSO-CV
│   │   ├── generate_pm25_map.py
│   │   └── export_to_geotiff.py
│   ├── data_acquisition/   # Data download scripts
│   │   ├── gee_downloader.py      # Google Earth Engine
│   │   ├── sinca_downloader_auto.py
│   │   └── sinca_selenium_downloader.py
│   ├── data_processing/    # Feature engineering
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py
│   │   └── integrate_satellite_data.py
│   ├── feature_engineering/
│   │   └── add_osm_features.py
│   └── utils/
│       └── regenerate_figures_english.py
├── data/
│   ├── processed/          # Processed datasets (see data/README.md)
│   └── raw/                # Raw data (not tracked, see Data Availability)
├── results/
│   └── figures/            # Publication figures
├── requirements_paper.txt  # Exact versions used in paper
└── requirements.txt        # Flexible versions for installation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/franciscoparrao/PM25_Santiago.git
cd PM25_Santiago

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (exact paper versions)
pip install -r requirements_paper.txt
```

### Google Earth Engine Authentication

```bash
earthengine authenticate
```

## Usage

### Temporal Forecasting (XGBoost)

```python
from src.temporal.forecasting import run_walk_forward_validation

# Run walk-forward validation with XGBoost
results = run_walk_forward_validation(
    data_path="data/processed/sinca_features_selected.csv",
    horizons=[1, 3, 7],  # 1-day, 3-day, 7-day forecasts
    n_iterations=2161
)
```

### Spatial Interpolation (Regression Kriging)

```python
from src.spatial.regression_kriging import RegressionKriging

# Initialize and fit model
rk = RegressionKriging()
rk.fit(X_train, y_train, coords_train)

# Predict on grid
predictions = rk.predict(X_grid, coords_grid)
```

### Model Comparison

```python
from src.temporal.temporal_models_comparison import compare_models

# Compare XGBoost, ARIMA, and Prophet
comparison = compare_models(
    data_path="data/processed/sinca_features_selected.csv",
    test_days=47
)
```

## Key Results

### Temporal Forecasting Performance (1-Day Ahead, walk-forward validation)

| Model           | n      | R²     | RMSE (μg/m³) | MAE (μg/m³) |
|-----------------|--------|--------|--------------|-------------|
| **XGBoost**     | 15,128 | **0.76** | 14.06      | 8.54        |
| Prophet         | 47     | 0.68   | 10.21        | 7.54        |
| ARIMA           | 47     | 0.58   | 11.60        | 7.20        |
| Persistence     | 15,128 | 0.74   | 14.72        | 9.11        |
| Historical Mean | 15,128 | 0.64   | 17.33        | 11.06       |

XGBoost attains the highest R² under honest walk-forward validation (2,161 iterations); ARIMA/Prophet were limited to 47 iterations by computational cost, and reach lower absolute errors only on their smaller, non-comparable test subsets. Total-runtime speedup: **13.4× vs ARIMA**, **~2.9× vs Prophet**.

### Spatial Interpolation (LOSO-CV) — negative result

| Approach                              | R²     | RMSE (μg/m³) |
|---------------------------------------|--------|--------------|
| Satellite-feature spatial models (avg)| −1.09  | 25.08        |

Interpolation to **unmeasured** locations was **not successful** (mean R² = −1.09 under Leave-One-Station-Out CV). Coarse satellite resolution (7–10 km) cannot capture intra-urban PM2.5 variability. This negative result is a central, deliberate contrast to the temporal success — not an omission.

### Feature Importance (Top 5, XGBoost 1-day)

| Feature              | Importance |
|----------------------|------------|
| pm25_lag_1d          | 38.7%      |
| pm25_rolling_mean_3d | 28.5%      |
| pm25_diff_1d         | 14.4%      |
| pm25_rolling_mean_14d| 9.9%       |
| elevation            | 1.0%       |

## Data Availability

### Ground-Truth Data
- **SINCA** (Sistema de Información Nacional de Calidad del Aire): https://sinca.mma.gob.cl/
- 8 stations, hourly PM2.5, January 2019 – November 2025

### Satellite Data (via Google Earth Engine)
- **Sentinel-5P NO₂**: `COPERNICUS/S5P/NRTI/L3_NO2`
- **MODIS AOD**: `MODIS/061/MCD19A2_GRANULES`
- **ERA5 Meteorology**: `ECMWF/ERA5/DAILY`

### Processed Datasets
All processed datasets used in this study are available in `data/processed/`. See `data/README.md` for detailed documentation.

## Archiving & Reproducibility

**Archived release (DOI):** _pending_ — an archived snapshot of this repository will be deposited on Zenodo and the DOI added here (and to `.zenodo.json` / `CITATION.cff`) upon release. Cite that DOI for the exact code and processed data behind the paper.

**Deterministic environment:**
- Python 3.12; exact package versions pinned in [`requirements_paper.txt`](requirements_paper.txt) (e.g. `xgboost==2.0.3`, `scikit-learn==1.3.2`).
- All estimators are seeded (`random_state=42`), so the reported metrics reproduce on the provided processed datasets.
- Processed datasets are included under `data/processed/`; raw satellite/ground data are not tracked and are re-downloadable from the sources listed under [Data Availability](#data-availability) (Google Earth Engine, SINCA).

**To reproduce the headline results** (after installation above):

```bash
# Temporal forecasting (R² = 0.76 at 1 day)
python -c "from src.temporal.forecasting import run_walk_forward_validation as r; r('data/processed/sinca_features_selected.csv', horizons=[1,3,7], n_iterations=2161)"
```

## Citation

This repository is citable via GitHub's "Cite this repository" button (see [`CITATION.cff`](CITATION.cff)). If you use this code or the derived datasets, please cite the manuscript:

```bibtex
@article{parra2025pm25santiago,
  title={Machine Learning for PM2.5 Forecasting in Inversion-Dominated Basins: Where It Adds Value over Persistence, Validated across Santiago and Salt Lake City},
  author={Parra, Francisco and Astudillo, Valentina},
  journal={Atmospheric Pollution Research},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Francisco Parra** - Departamento de Ingeniería Informática, Universidad de Santiago de Chile
- **Valentina Astudillo** - Independent Researcher, Santiago, Chile

## Acknowledgments

- SINCA (Sistema de Información Nacional de Calidad del Aire) for ground-truth PM2.5 data
- Google Earth Engine for satellite data access
- European Space Agency (Sentinel-5P) and NASA (MODIS, ERA5) for satellite products
