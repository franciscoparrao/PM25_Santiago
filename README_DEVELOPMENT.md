P# Spatiotemporal PM2.5 Prediction for Santiago, Chile using Satellite Data and Machine Learning

**Project Status:** 🚀 Active Development
**Start Date:** November 10, 2025
**Target Journal:** Atmospheric Environment (Q1, IF 5.0) or Environmental Pollution (Q1, IF 8.9)

---

## 📋 Project Overview

This research develops a high-resolution (1km × 1km) spatiotemporal prediction model for PM2.5 concentrations in the Santiago Metropolitan Region, Chile, using Google Earth Engine satellite data and Machine Learning.

### Research Questions

1. Can we accurately predict PM2.5 at 1km resolution using only satellite data?
2. Which satellite-derived features are most important for PM2.5 prediction?
3. How does model performance compare across different ML algorithms?
4. What is the spatial distribution of population exposure to PM2.5 in Santiago?

### Key Features

- **Multi-source satellite data:** Sentinel-5P, MODIS, ERA5
- **ML models:** Random Forest, XGBoost, LightGBM, Ensemble
- **Validation:** 32 SINCA stations × 6 years (2019-2025)
- **High resolution:** 1km spatial, daily temporal
- **Open science:** Reproducible code and data

---

## 🛰️ Data Sources

### Satellite Data (Google Earth Engine)

| Dataset | Variables | Resolution | Temporal | GEE ID |
|---------|-----------|------------|----------|---------|
| **Sentinel-5P TROPOMI** | NO₂, SO₂, CO, O₃, AOD | 7 km | Daily | `COPERNICUS/S5P/OFFL/L3_*` |
| **MODIS MCD19A2** | AOD (550nm) | 1 km | Daily | `MODIS/006/MCD19A2_GRANULES` |
| **MODIS MOD11A1** | Land Surface Temperature | 1 km | Daily | `MODIS/006/MOD11A1` |
| **MODIS MOD13A2** | NDVI (vegetation) | 1 km | 16 days | `MODIS/006/MOD13A2` |
| **ERA5** | Wind, Temp, RH, Pressure | 25 km | Hourly | `ECMWF/ERA5/DAILY` |
| **WorldPop** | Population density | 100 m | Annual | `WorldPop/GP/100m/pop` |
| **SRTM** | Elevation | 30 m | Static | `USGS/SRTMGL1_003` |

### Ground-Truth Data

- **SINCA** (Sistema de Información Nacional de Calidad del Aire)
- 32 monitoring stations in Santiago Metropolitan Region
- Hourly PM2.5 measurements: 2019-2025
- URL: https://sinca.mma.gob.cl/

---

## 🧪 Methodology

### 1. Data Acquisition
- Extract satellite data via Google Earth Engine Python API
- Download SINCA ground-truth data
- Spatial matching: stations ↔ satellite pixels
- Temporal synchronization

### 2. Feature Engineering
**Satellite features (10-15):**
- NO₂, SO₂, CO, O₃, AOD (Sentinel-5P)
- AOD, LST, NDVI (MODIS)

**Meteorological features (6-8):**
- Temperature, Relative Humidity, Wind Speed/Direction, Pressure

**Temporal features (8-10):**
- Hour, day of week, month, season
- Holidays, weekday/weekend
- Lag features: PM2.5(t-1), PM2.5(t-24)

**Spatial features (5-7):**
- Elevation, distance to roads, population density
- Land use type, distance to industrial areas

**Total features:** ~30-40 variables

### 3. Machine Learning Models

**Baseline:**
- Linear Regression
- Persistence Model (yesterday's value)

**Advanced ML:**
- Random Forest (RF)
- Gradient Boosting Machine (GBM)
- XGBoost
- LightGBM
- Ensemble (weighted combination)

**Hyperparameter Tuning:**
- Bayesian Optimization (Optuna)
- 5-fold time-series cross-validation

### 4. Evaluation Metrics

- **R² (Coefficient of Determination):** Target > 0.75
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

**Validation strategy:**
- Temporal split: 70% train / 15% validation / 15% test
- Spatial validation: Leave-one-station-out cross-validation

### 5. Analysis

- Feature importance analysis (SHAP values)
- Spatial mapping of predictions (1km grid)
- Population exposure assessment by comuna
- Temporal trend analysis (2019-2025)
- Hotspot identification

---

## 📁 Project Structure

```
PM25_Santiago/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── config/
│   ├── config.yaml                 # Project configuration
│   └── study_area.geojson          # Santiago boundary
├── data/
│   ├── raw/                        # Raw data (not tracked in git)
│   │   ├── sinca/                  # SINCA ground-truth
│   │   ├── sentinel5p/             # Sentinel-5P exports
│   │   ├── modis/                  # MODIS exports
│   │   └── era5/                   # ERA5 meteorology
│   ├── processed/                  # Processed datasets
│   │   ├── features_train.csv
│   │   ├── features_test.csv
│   │   └── metadata.json
│   └── external/                   # Auxiliary data (population, roads)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sinca_analysis.ipynb
│   ├── 03_satellite_data_extraction.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_ml_modeling.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_spatial_analysis.ipynb
│   └── 09_population_exposure.ipynb
├── src/
│   ├── __init__.py
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── gee_downloader.py       # GEE data extraction
│   │   ├── sinca_scraper.py        # SINCA data download
│   │   └── data_matcher.py         # Spatial/temporal matching
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── data_cleaning.py
│   │   └── quality_control.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── ml_models.py
│   │   ├── ensemble.py
│   │   └── hyperparameter_tuning.py
│   └── visualization/
│       ├── __init__.py
│       ├── maps.py
│       ├── plots.py
│       └── reports.py
├── models/                         # Saved trained models
│   ├── random_forest_v1.pkl
│   ├── xgboost_v1.pkl
│   └── ensemble_v1.pkl
├── results/
│   ├── figures/                    # Publication-ready figures
│   │   ├── fig1_study_area.png
│   │   ├── fig2_temporal_trends.png
│   │   ├── fig3_model_comparison.png
│   │   ├── fig4_spatial_predictions.png
│   │   └── fig5_feature_importance.png
│   ├── tables/                     # Result tables
│   │   ├── table1_model_performance.csv
│   │   └── table2_feature_importance.csv
│   └── reports/
│       └── final_report.pdf
└── docs/
    ├── project_proposal.md
    ├── data_dictionary.md
    ├── methodology.md
    └── paper_outline.md
```

---

## 🚀 Getting Started

### 1. Set up environment

```bash
# Clone repository (if using git)
cd /home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago

# Create conda environment
conda env create -f environment.yml
conda activate pm25-santiago

# Or use pip
pip install -r requirements.txt
```

### 2. Configure Google Earth Engine

```bash
# Authenticate with GEE
earthengine authenticate

# Initialize in Python
import ee
ee.Initialize()
```

### 3. Download SINCA data

```bash
python src/data_acquisition/sinca_scraper.py --start-date 2019-01-01 --end-date 2025-11-10
```

### 4. Extract satellite data

```bash
# Run GEE extraction script
python src/data_acquisition/gee_downloader.py --config config/config.yaml
```

### 5. Run analysis

Open Jupyter notebooks in sequence:
```bash
jupyter lab notebooks/
```

---

## 📊 Expected Results

### Model Performance (Target)

| Model | R² | RMSE (µg/m³) | MAE (µg/m³) |
|-------|-----|--------------|-------------|
| Linear Regression | 0.50-0.60 | 15-20 | 10-15 |
| Random Forest | **0.75-0.80** | 10-12 | 7-9 |
| XGBoost | **0.78-0.82** | 9-11 | 6-8 |
| LightGBM | **0.76-0.80** | 10-12 | 7-9 |
| Ensemble | **0.80-0.85** | 8-10 | 6-7 |

### Key Findings (Anticipated)

1. **High predictive accuracy** (R² > 0.75) achieved with satellite-only data
2. **AOD and meteorology** are top predictors of PM2.5
3. **Spatial heterogeneity:** Hotspots in western and southern Santiago
4. **Temporal patterns:** Peak pollution in winter months (June-August)
5. **Population exposure:** ~4-5 million people exposed to PM2.5 > 25 µg/m³

---

## 📝 Publications

### Target Journals (Q1)

**Primary:**
1. **Atmospheric Environment** (Q1, IF 5.0)
   - Scope: Air quality modeling and monitoring
   - Audience: Atmospheric scientists

2. **Environmental Pollution** (Q1, IF 8.9)
   - Scope: Environmental contamination and health
   - Audience: Environmental scientists, public health

**Alternative:**
3. **Remote Sensing of Environment** (Q1, IF 13.5)
   - Scope: Remote sensing methodology
   - Emphasize GEE + ML innovation

4. **Science of the Total Environment** (Q1, IF 9.8)
   - Scope: Multidisciplinary environmental science

### Paper Outline

**Title:** "High-Resolution Spatiotemporal Prediction of PM2.5 in Santiago, Chile using Sentinel-5P, MODIS and Machine Learning"

**Sections:**
1. Abstract (250 words)
2. Introduction (1,200 words)
3. Materials and Methods (2,500 words)
4. Results (2,000 words)
5. Discussion (1,800 words)
6. Conclusions (500 words)
7. References (60-80)

**Target:** 8,000 words, 6-8 figures, 3-4 tables

---

## 📅 Timeline (6 months)

| Phase | Duration | Weeks | Deliverable |
|-------|----------|-------|-------------|
| **Setup & Data Acquisition** | 4 weeks | 1-4 | Raw datasets |
| **Data Preprocessing** | 2 weeks | 5-6 | Clean datasets |
| **Feature Engineering** | 2 weeks | 7-8 | Feature matrix |
| **Baseline Models** | 1 week | 9 | Baseline results |
| **ML Modeling** | 3 weeks | 10-12 | Trained models |
| **Model Evaluation** | 2 weeks | 13-14 | Performance metrics |
| **Spatial Analysis** | 2 weeks | 15-16 | Maps, exposure analysis |
| **Visualization & Figures** | 2 weeks | 17-18 | Publication figures |
| **Manuscript Writing** | 4 weeks | 19-22 | Draft manuscript |
| **Revision & Submission** | 2 weeks | 23-24 | Submitted paper |

**Total:** 24 weeks (~6 months)

---

## 👥 Team

- **Lead Researcher:** Francisco Parrao
- **Collaborators:** TBD (atmospheric science, epidemiology, GIS)

---

## 📚 References

### Key Papers (to cite)

1. **Methodology:**
   - Hu et al. (2017) - Estimating PM2.5 with satellite data and ML
   - Wei et al. (2021) - Reconstructing 1-km-resolution high-quality PM2.5

2. **Sentinel-5P applications:**
   - Zhao et al. (2023) - NO2 prediction using TROPOMI
   - Liu et al. (2024) - Multi-pollutant modeling with S5P

3. **Santiago air quality:**
   - Gramsch et al. (2006) - Air pollution in Santiago
   - Toro et al. (2014) - PM2.5 sources in Santiago

4. **GEE + ML:**
   - Gorelick et al. (2017) - Google Earth Engine
   - Chen et al. (2018) - XGBoost for air quality

---

## 📄 License

This project is for academic research purposes.

**Data:**
- Satellite data: Open access (ESA, NASA)
- SINCA data: Public domain (Chilean government)

**Code:**
- MIT License (to be confirmed)

---

## 📧 Contact

- **Francisco Parrao**
- **Institution:** TBD
- **Email:** TBD

---

## 🔄 Updates

- **2025-11-10:** Project initialized
- **2025-11-10:** Directory structure created
- **2025-11-10:** README drafted

---

## 🎯 Success Criteria

- ✅ R² > 0.75 on test set
- ✅ Validation with 32 SINCA stations
- ✅ High-resolution spatial maps (1km)
- ✅ Comprehensive temporal analysis (2019-2025)
- ✅ Reproducible code and data
- ✅ Manuscript accepted in Q1 journal
- ✅ Code published on GitHub
- ✅ Data published on Zenodo/Figshare

**Let's build something impactful! 🌍📊🔬**
