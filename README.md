# PM2.5 Forecasting and Spatial Interpolation for Santiago, Chile

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and data for the paper:

> **Parra, F. & Astudillo, V.** (2025). Spatiotemporal PM2.5 Prediction in Santiago, Chile: Integrating Satellite Data, Machine Learning Forecasting, and Geostatistical Interpolation. *Environmental Modelling & Software*.

## Abstract

This study develops a hybrid framework combining XGBoost-based temporal forecasting with regression kriging for spatial interpolation of PM2.5 concentrations in Santiago, Chile. Using 6 years of data (2019-2024) from 11 SINCA monitoring stations and satellite-derived features, we achieve:

- **Temporal forecasting**: R² = 0.76, RMSE = 8.52 μg/m³ (1-day horizon)
- **Spatial interpolation**: R² = 0.89, RMSE = 5.23 μg/m³ via LOSO-CV
- **107× faster** than ARIMA, **185× faster** than Prophet

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
git clone https://github.com/franciscoparraUSACH/PM25_Santiago.git
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

### Temporal Forecasting Performance

| Model   | R²    | RMSE (μg/m³) | MAE (μg/m³) | Time (s) |
|---------|-------|--------------|-------------|----------|
| XGBoost | 0.76  | 8.52         | 5.78        | 127.3    |
| ARIMA   | 0.71  | 9.38         | 6.42        | 13,601   |
| Prophet | 0.69  | 9.67         | 6.89        | 23,551   |

### Spatial Interpolation (LOSO-CV)

| Model              | R²   | RMSE (μg/m³) |
|--------------------|------|--------------|
| Regression Kriging | 0.89 | 5.23         |

### Feature Importance (Top 5)

| Feature              | Importance |
|----------------------|------------|
| pm25_lag_1d          | 38.7%      |
| pm25_rolling_mean_3d | 28.5%      |
| temperature_2m       | 8.9%       |
| relative_humidity    | 6.2%       |
| wind_speed           | 4.8%       |

## Data Availability

### Ground-Truth Data
- **SINCA** (Sistema de Información Nacional de Calidad del Aire): https://sinca.mma.gob.cl/
- 11 stations, hourly PM2.5, 2019-2024

### Satellite Data (via Google Earth Engine)
- **Sentinel-5P NO₂**: `COPERNICUS/S5P/NRTI/L3_NO2`
- **MODIS AOD**: `MODIS/061/MCD19A2_GRANULES`
- **ERA5 Meteorology**: `ECMWF/ERA5/DAILY`

### Processed Datasets
All processed datasets used in this study are available in `data/processed/`. See `data/README.md` for detailed documentation.

## Citation

If you use this code or data, please cite:

```bibtex
@article{parra2025pm25santiago,
  title={Spatiotemporal PM2.5 Prediction in Santiago, Chile: Integrating Satellite Data, Machine Learning Forecasting, and Geostatistical Interpolation},
  author={Parra, Francisco and Astudillo, Valentina},
  journal={Environmental Modelling \& Software},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Francisco Parra** - Departamento de Ingeniería Informática, Universidad de Santiago de Chile
- **Valentina Astudillo** - Departamento de Geología, Universidad de Chile

## Acknowledgments

- SINCA (Sistema de Información Nacional de Calidad del Aire) for ground-truth PM2.5 data
- Google Earth Engine for satellite data access
- European Space Agency (Sentinel-5P) and NASA (MODIS, ERA5) for satellite products
