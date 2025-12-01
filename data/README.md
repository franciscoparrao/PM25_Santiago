# Data Documentation

This directory contains the datasets used in the study:

> Parra, F. & Astudillo, V. (2025). Spatiotemporal PM2.5 Prediction in Santiago, Chile. *Environmental Modelling & Software*.

## Directory Structure

```
data/
├── processed/          # Analysis-ready datasets (included in repo)
├── raw/                # Original downloads (not tracked, see instructions below)
└── external/           # Auxiliary spatial data
```

## Processed Datasets

### Main Feature Matrices

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `sinca_features_selected.csv` | Final feature matrix for temporal forecasting | ~15,000 | 25 |
| `sinca_features_spatial.csv` | Feature matrix for spatial interpolation | ~15,000 | 30 |
| `sinca_pm25_master.csv` | Consolidated PM2.5 observations from all stations | ~40,000 | 8 |

### Temporal Forecasting Results

| File | Description |
|------|-------------|
| `forecast_1d_predictions.csv` | 1-day ahead XGBoost predictions (2,161 iterations) |
| `forecast_3d_predictions.csv` | 3-day ahead predictions |
| `forecast_7d_predictions.csv` | 7-day ahead predictions |
| `forecast_metrics_comparison.csv` | Summary metrics by horizon |
| `temporal_feature_importance.csv` | XGBoost feature importance scores |
| `arima_predictions_1d.csv` | ARIMA baseline predictions (47 days) |
| `prophet_predictions_1d.csv` | Prophet baseline predictions (47 days) |

### Spatial Interpolation Results

| File | Description |
|------|-------------|
| `spatial_loso_results.csv` | Leave-One-Station-Out CV results |
| `spatial_feature_importance.csv` | Regression Kriging feature importance |
| `pm25_grid_predictions.csv` | 1km grid predictions for Santiago |
| `pm25_map_santiago.tif` | GeoTIFF map output |

### Seasonal Analysis

| File | Description |
|------|-------------|
| `seasonal_pm25_statistics.csv` | PM2.5 statistics by season |
| `seasonal_model_metrics.csv` | Model performance by season |
| `critical_episodes_metrics_by_horizon.csv` | Exceedance detection metrics |

## Raw Data Sources (Not Tracked)

### Ground-Truth: SINCA

- **Source**: Sistema de Información Nacional de Calidad del Aire
- **URL**: https://sinca.mma.gob.cl/
- **Variables**: Hourly PM2.5 concentrations
- **Stations**: 11 in Santiago Metropolitan Region
- **Period**: 2019-01-01 to 2024-12-31
- **Download**: Use `src/data_acquisition/sinca_selenium_downloader.py`

### Satellite Data: Google Earth Engine

| Product | GEE ID | Variables |
|---------|--------|-----------|
| Sentinel-5P NO₂ | `COPERNICUS/S5P/NRTI/L3_NO2` | tropospheric_NO2_column_number_density |
| MODIS AOD | `MODIS/061/MCD19A2_GRANULES` | Optical_Depth_047 |
| ERA5 | `ECMWF/ERA5/DAILY` | temperature_2m, relative_humidity, wind_speed |

**Download**: Use `src/data_acquisition/gee_downloader.py`

## Data Dictionary

### Key Variables in `sinca_features_selected.csv`

| Variable | Description | Unit | Source |
|----------|-------------|------|--------|
| `pm25` | Target: Daily mean PM2.5 | μg/m³ | SINCA |
| `pm25_lag_1d` | PM2.5 from previous day | μg/m³ | Derived |
| `pm25_lag_3d` | PM2.5 from 3 days ago | μg/m³ | Derived |
| `pm25_rolling_mean_3d` | 3-day rolling mean PM2.5 | μg/m³ | Derived |
| `pm25_rolling_mean_7d` | 7-day rolling mean PM2.5 | μg/m³ | Derived |
| `temperature_2m` | Air temperature at 2m | °C | ERA5 |
| `relative_humidity` | Relative humidity | % | ERA5 |
| `wind_speed` | Wind speed | m/s | ERA5 |
| `wind_direction` | Wind direction | degrees | ERA5 |
| `pressure` | Surface pressure | hPa | ERA5 |
| `no2_tropospheric` | Tropospheric NO₂ column | mol/m² | Sentinel-5P |
| `aod_modis` | Aerosol optical depth | dimensionless | MODIS |
| `month` | Month of year | 1-12 | Derived |
| `day_of_week` | Day of week | 0-6 | Derived |
| `is_weekend` | Weekend indicator | 0/1 | Derived |
| `station_id` | SINCA station identifier | - | SINCA |
| `latitude` | Station latitude | degrees | SINCA |
| `longitude` | Station longitude | degrees | SINCA |

## Reproduction

To reproduce the raw data acquisition:

```bash
# 1. Authenticate with Google Earth Engine
earthengine authenticate

# 2. Download SINCA data (requires Selenium + Chrome)
python src/data_acquisition/sinca_selenium_downloader.py

# 3. Download satellite data
python src/data_acquisition/gee_downloader.py

# 4. Run feature engineering
python src/data_processing/feature_engineering.py
```

## File Sizes

| Directory | Size | Note |
|-----------|------|------|
| `processed/` | ~48 MB | Included in repository |
| `raw/` | ~80 MB | Not tracked (regenerate or download) |
| `external/` | ~2 MB | Auxiliary GIS data |

## License

- **SINCA data**: Public domain (Chilean government)
- **Satellite data**: Open access (ESA Copernicus, NASA MODIS, ECMWF ERA5)
- **Processed datasets**: MIT License (same as code)
