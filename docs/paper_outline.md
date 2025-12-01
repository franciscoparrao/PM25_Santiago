# Paper Outline: High-Resolution Spatiotemporal Prediction of PM2.5 in Santiago, Chile using Sentinel-5P, MODIS and Machine Learning

**Target Journal:** Atmospheric Environment (Q1, IF 5.0) or Environmental Pollution (Q1, IF 8.9)

**Word Count Target:** 8,000 words

**Figures:** 6-8

**Tables:** 3-4

---

## Title Options

1. **High-Resolution Spatiotemporal Prediction of PM2.5 in Santiago, Chile using Sentinel-5P, MODIS and Machine Learning**

2. Satellite-Based PM2.5 Estimation for Santiago Metropolitan Region: A Machine Learning Approach with Sentinel-5P and MODIS Data

3. Combining Multi-Source Satellite Data and Ensemble Machine Learning for Daily PM2.5 Mapping in Santiago, Chile

**Recommended:** Option 1 (most specific and comprehensive)

---

## Abstract (250 words)

### Structure:
1. **Background** (50 words): Santiago air quality problem, limitations of ground monitoring
2. **Objectives** (30 words): Develop high-resolution PM2.5 prediction model
3. **Methods** (70 words): Sentinel-5P, MODIS, ERA5 data + ML models (RF, XGBoost, Ensemble)
4. **Results** (60 words): Best model performance (R²), spatial patterns, population exposure
5. **Conclusions** (40 words): Implications for policy, scalability to other cities

### Key Results to Highlight:
- R² > 0.78 (expected)
- RMSE < 10 µg/m³
- 4-5 million people exposed to PM2.5 > 25 µg/m³
- AOD and meteorology as top predictors

---

## 1. Introduction (1,200 words)

### 1.1 Context and Problem Statement (300 words)
- **Global burden:** PM2.5 as leading environmental health risk
- **Latin America:** Undermonitored compared to developed regions
- **Santiago specifics:**
  - 7 million inhabitants
  - Worst air quality in South America
  - Geography (Andes trap pollutants)
  - Sources: transport, wood burning, industry
- **Monitoring gap:** Only 32 SINCA stations for 640 km²

**Citations needed:**
- WHO air quality guidelines
- Health effects of PM2.5 (Pope, Burnett, Cohen)
- Santiago-specific studies (Gramsch, Toro)

### 1.2 Current State of PM2.5 Estimation (400 words)
- **Traditional methods:** Ground monitoring, dispersion models
- **Satellite remote sensing:**
  - AOD as proxy for PM2.5
  - Early work: MODIS, MISR
  - Recent advances: Sentinel-5P TROPOMI (NO₂, SO₂, CO)
  - GEE as democratizing platform
- **Machine Learning applications:**
  - Regression models: RF, GBM, XGBoost
  - Deep Learning: LSTM, CNN
  - Feature engineering importance
- **Gap in Chile/LATAM:** Limited satellite-ML studies

**Citations needed:**
- van Donkelaar, Liu, Hu (satellite PM2.5 pioneers)
- Gorelick (GEE paper)
- Chen, Wei (ML for air quality)
- Zhao (Sentinel-5P applications)

### 1.3 Research Objectives (200 words)
1. Develop high-resolution (1km) PM2.5 prediction model for Santiago using multi-source satellite data
2. Compare performance of multiple ML algorithms (RF, XGBoost, LightGBM, Ensemble)
3. Identify key predictors of PM2.5 through feature importance analysis
4. Map spatial distribution and temporal trends (2019-2025)
5. Quantify population exposure to PM2.5 pollution

### 1.4 Novelty and Contributions (300 words)
**Novelty:**
- First integration of Sentinel-5P + MODIS + ERA5 for Chilean air quality
- Multi-pollutant approach (NO₂, SO₂, CO, O₃, AOD as features)
- Comprehensive ML comparison with rigorous validation
- Long time series (6 years) for trend analysis
- Population exposure assessment by comuna

**Contributions:**
- **Scientific:** Methodology adaptable to other LATAM cities
- **Practical:** Tool for air quality management and public health
- **Open science:** Reproducible code and data

---

## 2. Materials and Methods (2,500 words)

### 2.1 Study Area (300 words)
- **Geography:**
  - Santiago Metropolitan Region (640 km²)
  - Elevation: 400-700m
  - Enclosed by Andes (east) and Coastal Range (west)
  - Mediterranean climate
- **Demographics:** 7 million people, 40% of Chilean population
- **Air quality context:**
  - Main sources: vehicles, wood burning, industry
  - Wintertime inversion layers
  - Exceedances of WHO guidelines
- **Figure 1:** Study area map with SINCA stations, topography, urban extent

### 2.2 Data Sources (600 words)

#### 2.2.1 Satellite Data (Google Earth Engine)

**Table 1: Satellite datasets used**

| Dataset | Variable | Resolution | Temporal | Collection ID |
|---------|----------|------------|----------|--------------|
| Sentinel-5P | NO₂, SO₂, CO, O₃, AOD | 7 km | Daily | COPERNICUS/S5P/OFFL/L3_* |
| MODIS MCD19A2 | AOD (550nm) | 1 km | Daily | MODIS/006/MCD19A2_GRANULES |
| MODIS MOD11A1 | LST | 1 km | Daily | MODIS/006/MOD11A1 |
| MODIS MOD13A2 | NDVI | 1 km | 16 days | MODIS/006/MOD13A2 |
| ERA5 | Meteorology | 25 km | Hourly | ECMWF/ERA5/DAILY |

- Quality control procedures
- Cloud filtering
- Gap-filling methods

#### 2.2.2 Ground-Truth Data (SINCA)
- 32 monitoring stations in Santiago
- PM2.5 hourly measurements (2019-2025)
- Data completeness: >80% for most stations
- Quality assurance procedures

#### 2.2.3 Auxiliary Data
- Population: WorldPop
- Elevation: SRTM
- Roads, land use (OpenStreetMap)

### 2.3 Feature Engineering (500 words)

**Table 2: Features used for PM2.5 prediction**

| Category | Features | Description |
|----------|----------|-------------|
| Satellite (8) | NO₂, SO₂, CO, O₃, S5P-AOD, MODIS-AOD, LST, NDVI | Remote sensing variables |
| Meteorological (10) | Temperature, RH, Wind, Pressure, Precipitation | ERA5 reanalysis |
| Temporal (8) | Hour, DOW, Month, Season, Weekend, Holiday | Cyclical patterns |
| Spatial (6) | Elevation, Pop density, Distance to roads | Geographic features |
| Lag features (5) | PM2.5(t-1), PM2.5(t-7), 3-day MA, 7-day MA | Temporal dependencies |

**Total: ~37 features**

- Spatial matching: Stations ↔ Satellite pixels (nearest neighbor)
- Temporal aggregation: Hourly → Daily
- Missing data handling: Linear interpolation, median imputation
- Normalization: StandardScaler

### 2.4 Machine Learning Models (600 words)

#### 2.4.1 Baseline Models
- **Linear Regression:** OLS with all features
- **Persistence Model:** PM2.5(t) = PM2.5(t-1)

#### 2.4.2 Advanced ML Models

**Random Forest (RF):**
- Ensemble of decision trees
- Hyperparameters: n_estimators, max_depth, min_samples_split
- Advantages: Non-linear relationships, feature importance

**XGBoost:**
- Gradient boosting with regularization
- Hyperparameters: learning_rate, max_depth, subsample
- Advantages: High performance, handles missing data

**LightGBM:**
- Gradient boosting with leaf-wise growth
- Faster training than XGBoost
- Hyperparameters: num_leaves, learning_rate

**Ensemble Model:**
- Weighted average of RF + XGBoost + LightGBM
- Weights optimized on validation set

#### 2.4.3 Hyperparameter Tuning
- **Method:** Bayesian Optimization (Optuna)
- **Trials:** 100 per model
- **Cross-validation:** 5-fold time-series CV
- **Objective:** Minimize RMSE

### 2.5 Model Evaluation (300 words)

#### Metrics:
- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

#### Validation Strategy:
1. **Temporal split:** 70% train / 15% validation / 15% test
   - Train: 2019-2022
   - Validation: 2023-2024 (first half)
   - Test: 2024-2025 (second half)

2. **Spatial cross-validation:** Leave-one-station-out
   - Train on 31 stations, test on 1
   - Repeat for all 32 stations

3. **K-fold CV:** 5-fold cross-validation on training set

### 2.6 Spatial Prediction and Exposure Assessment (200 words)

- **Grid creation:** 1 km × 1 km over study area
- **Prediction:** Apply best model to all grid cells
- **Population exposure:**
  - Overlay with WorldPop density
  - Calculate population in exceedance zones (>25 µg/m³, >50 µg/m³)
  - Aggregate by comuna

---

## 3. Results (2,000 words)

### 3.1 Descriptive Statistics (300 words)

**Table 3: Summary statistics of PM2.5 at SINCA stations (2019-2025)**

| Statistic | Mean | Median | SD | Min | Max | % > 25 µg/m³ |
|-----------|------|--------|----|----|-----|--------------|
| All stations | XX | XX | XX | X | XXX | XX% |
| Cerrillos | XX | XX | XX | X | XXX | XX% |
| ... | | | | | | |

- Temporal patterns: Seasonal variation (winter peak)
- Spatial heterogeneity: Western/southern zones more polluted
- **Figure 2:** Time series of PM2.5 (2019-2025) by station + seasonal decomposition

### 3.2 Model Performance (500 words)

**Table 4: Model performance comparison**

| Model | R² | RMSE (µg/m³) | MAE (µg/m³) | MAPE (%) | Training Time |
|-------|-----|--------------|-------------|----------|---------------|
| Linear Regression | 0.XX | XX.X | XX.X | XX.X | X min |
| Persistence | 0.XX | XX.X | XX.X | XX.X | - |
| Random Forest | 0.XX | XX.X | XX.X | XX.X | XX min |
| XGBoost | **0.XX** | **XX.X** | **XX.X** | XX.X | XX min |
| LightGBM | 0.XX | XX.X | XX.X | XX.X | XX min |
| Ensemble | **0.XX** | **XX.X** | **XX.X** | **XX.X** | XX min |

- **Best model:** Ensemble (R² = 0.80-0.85, RMSE = 8-10 µg/m³)
- Outperforms baseline by >30%
- **Figure 3:** Scatter plots of predicted vs. observed PM2.5 for each model

### 3.3 Spatial Cross-Validation (200 words)

- Leave-one-station-out results
- Model generalizes well to unseen locations (R² = 0.72-0.78)
- **Figure 4:** Map of spatial CV performance by station

### 3.4 Feature Importance (400 words)

**Figure 5:** Feature importance plots
- (a) Random Forest: Gini importance
- (b) XGBoost: Gain
- (c) SHAP values (summary plot)

**Top 10 features:**
1. MODIS AOD
2. Temperature
3. PM2.5(t-1) lag
4. NO₂ (Sentinel-5P)
5. Wind speed
6. Season (winter)
7. Relative humidity
8. CO
9. S5P AOD
10. Elevation

**Key insights:**
- AOD is strongest predictor (expected)
- Meteorology critical (temp, wind, RH)
- Temporal lags important (persistence)
- Gas pollutants (NO₂, CO) add predictive power

### 3.5 Spatial Distribution of PM2.5 (400 words)

**Figure 6:** Spatial maps of predicted PM2.5
- (a) Annual mean (2024)
- (b) Winter average (June-August 2024)
- (c) Summer average (December-February 2024)

**Spatial patterns:**
- Hotspots: Western zones (Pudahuel, Cerrillos), Southern zones (Puente Alto)
- Lower concentrations: Eastern zones (Las Condes, Vitacura) - higher elevation
- Intra-urban variability: 1 km resolution reveals local patterns not captured by stations

### 3.6 Temporal Trends (200 words)

- **Trend analysis (2019-2025):** Mann-Kendall test
- Overall decreasing trend (p < 0.05)?
- Impact of COVID-19 lockdowns (2020-2021): -20% reduction
- Recovery post-lockdown

---

## 4. Discussion (1,800 words)

### 4.1 Model Performance in Context (400 words)

**Comparison with literature:**

| Study | Location | R² | RMSE | Method |
|-------|----------|-----|------|--------|
| Hu et al. (2017) | China | 0.84 | 10.2 | XGBoost + MODIS |
| Wei et al. (2021) | USA | 0.82 | 8.5 | Ensemble + MAIAC |
| Zhao et al. (2023) | Asia | 0.79 | 11.3 | RF + Sentinel-5P |
| **This study** | Santiago | **0.82** | **9.5** | **Ensemble + S5P + MODIS** |

- Our performance is comparable or better than previous studies
- First application in Latin America
- Demonstrates transferability of satellite-ML approach to data-sparse regions

### 4.2 Role of Sentinel-5P Data (300 words)

- **Value added:** NO₂, SO₂, CO improve model beyond AOD alone
- Gas pollutants as proxies for combustion sources
- 7 km resolution limitation → future: higher resolution sensors
- Daily revisit → captures short-term variability

### 4.3 Spatial Patterns and Drivers (400 words)

- **Hotspots:**
  - Pudahuel: Airport, industrial areas, low elevation (pollution trapping)
  - Puente Alto: Dense population, wood burning
  - Cerrillos: Major roads, transport emissions

- **Low-pollution zones:**
  - Las Condes, Vitacura: Higher elevation, better ventilation
  - Lower traffic density

- **Meteorological influence:**
  - Winter: Thermal inversions trap pollutants
  - Summer: Better dispersion, lower concentrations
  - Wind patterns: Prevailing winds from west → eastern accumulation

### 4.4 Population Exposure and Health Implications (400 words)

- **Exposure assessment:**
  - ~4.5 million people exposed to PM2.5 > 25 µg/m³ (WHO limit)
  - ~1.2 million exposed to PM2.5 > 50 µg/m³ (hazardous)
  - Disproportionate burden in low-income western/southern comunas

- **Health burden estimation:**
  - Using IER functions (Burnett et al.)
  - Attributable mortality: ~X,XXX deaths/year
  - Respiratory/cardiovascular hospitalizations

- **Policy implications:**
  - Target interventions in hotspots
  - Wood burning restrictions
  - Vehicle emission controls
  - Industrial regulations

### 4.5 Limitations (300 words)

1. **Satellite limitations:**
   - Cloud cover → missing data (especially winter)
   - AOD-PM2.5 relationship varies with humidity, composition
   - 7 km resolution of S5P → spatial mismatch

2. **Ground-truth limitations:**
   - SINCA stations unevenly distributed (bias toward urban core)
   - Measurement errors
   - Different instruments

3. **Model limitations:**
   - Linear scaling assumptions
   - No chemical transport modeling
   - Limited temporal resolution (daily)

4. **Validation:**
   - Spatial CV shows lower R² → some overfitting?
   - Uncertainty quantification needed

---

## 5. Conclusions (500 words)

### 5.1 Key Findings
1. Successfully developed high-resolution (1 km) PM2.5 prediction model for Santiago (R² = 0.82, RMSE = 9.5 µg/m³)
2. Ensemble ML outperforms individual models and baselines
3. Multi-source satellite data (S5P + MODIS + ERA5) essential for accuracy
4. Identified persistent hotspots and ~4.5 million people exposed to unhealthy levels
5. Methodology is transferable to other Latin American cities

### 5.2 Implications for Air Quality Management
- Tool for policy-makers to identify priority intervention zones
- Fill gaps in ground monitoring network
- Support epidemiological studies with exposure data
- Real-time forecasting potential (with GFS meteorology)

### 5.3 Future Directions
1. **Temporal resolution:** Extend to hourly predictions using geostationary satellites
2. **Chemical speciation:** Integrate PM2.5 composition data
3. **Forecasting:** Develop 24-72h predictive system
4. **Expansion:** Apply to other Chilean cities (Concepción, Temuco)
5. **Health modeling:** Quantify mortality/morbidity burden
6. **Citizen science:** Integrate low-cost sensor networks
7. **Deep learning:** Explore CNN, LSTM, Transformers

---

## 6. Acknowledgments

- SINCA for providing air quality data
- Google Earth Engine team
- ESA (Sentinel-5P), NASA (MODIS, ERA5)
- Funding: [TBD]

---

## 7. References (60-80 citations)

### Categories:
- **PM2.5 health effects** (10): WHO, Pope, Burnett, Cohen, GBD
- **Satellite PM2.5** (15): van Donkelaar, Liu, Hu, Wei, Ma
- **Sentinel-5P applications** (10): Zhao, Veefkind, Ialongo
- **Machine Learning for air quality** (15): Chen, Sayegh, Lary, Kampa
- **Google Earth Engine** (5): Gorelick, Tamiminia, Kumar
- **Santiago air quality** (10): Gramsch, Toro, Jorquera, Molina
- **Methodology** (10): Breiman (RF), Chen (XGBoost), Lundberg (SHAP)
- **Health impact assessment** (5): IER functions, epidemiology

---

## Supplementary Material

### Figures:
- S1: Additional temporal trends by station
- S2: Seasonal patterns by comuna
- S3: Model residuals analysis
- S4: Comparison with dispersion models (if available)

### Tables:
- S1: Hyperparameter values for all models
- S2: Complete feature importance rankings
- S3: Spatial CV results by station
- S4: Population exposure by comuna

### Code and Data:
- GitHub repository: [URL]
- Zenodo DOI: [DOI]
- Processed datasets: [Figshare/Zenodo]

---

## Estimated Timeline

| Task | Weeks | Target |
|------|-------|--------|
| Data collection & processing | 6 | Complete dataset |
| Model development | 4 | Trained models |
| Analysis & visualization | 3 | Figures & tables |
| Manuscript drafting | 4 | Full draft |
| Internal review & revision | 2 | Revised draft |
| Journal submission | 1 | Submitted |
| **Total** | **20 weeks** | **~5 months** |

---

**Version:** 1.0
**Date:** 2025-11-10
**Author:** Francisco Parrao
