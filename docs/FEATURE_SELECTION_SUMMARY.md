# Feature Selection - Resumen

**Fecha**: 13 de noviembre de 2025
**Dataset Input**: `sinca_features_engineered.csv` (47 columnas, 16,344 registros)

---

## 📊 Resultados Generales

### Features Evaluadas
- **Total inicial**: 33 features (sin contar metadatos ni target)
- **Métodos aplicados**:
  - ✅ Análisis de correlación (threshold: 0.90)
  - ✅ Feature importance con Random Forest (100 árboles)
  - ✅ Mutual Information Score
  - ⚠️ VIF (Variance Inflation Factor) - omitido (requiere statsmodels)

---

## 🔍 Hallazgos Principales

### 1. Dominancia de Lag Features

Las **lag features** (PM2.5 rezagado) dominan completamente el modelo:

| Feature | RF Importance | MI Score | Descripción |
|---------|---------------|----------|-------------|
| `pm25_lag1` | **65.97%** | 0.759 | PM2.5 del día anterior |
| `pm25_diff1` | **20.33%** | 0.269 | Cambio respecto al día anterior |
| `pm25_ma7` | **13.31%** | 0.782 | Promedio móvil 7 días |
| `pm25_std7` | 0.18% | 0.458 | Volatilidad 7 días |
| `pm25_ma30` | 0.02% | 0.625 | Promedio móvil 30 días |
| `pm25_lag7` | 0.02% | 0.385 | PM2.5 hace 7 días |

**Total lag features**: **99.81% de la importancia RF**

**Interpretación**:
- PM2.5 es extremadamente autocorrelacionado
- El valor de ayer (`pm25_lag1`) es el predictor más fuerte
- Las 3 primeras lag features capturan casi todo el poder predictivo

---

### 2. Features Satelitales (MODIS, Sentinel-5P)

A pesar de su baja importancia RF, tienen **MI scores razonables**:

| Feature | RF Importance | MI Score | Descripción |
|---------|---------------|----------|-------------|
| `modis_aod` | 0.008% | **0.319** | Aerosol Optical Depth |
| `s5p_no2` | 0.008% | **0.346** | Dióxido de nitrógeno |

**Importancia**:
- Aportan información **independiente** de las lags (datos remotos, no ground-based)
- Útiles para **predicción espacial** (estaciones sin datos históricos)
- Relevantes para **interpretabilidad** (causas físicas de contaminación)

---

### 3. Features Meteorológicas (ERA5)

Importancia individual muy baja, pero **MI scores moderados**:

| Feature | RF Importance | MI Score | Descripción |
|---------|---------------|----------|-------------|
| `era5_total_precipitation_hourly` | 0.018% | 0.258 | Precipitación horaria |
| `relative_humidity` | 0.011% | 0.275 | Humedad relativa |
| `era5_surface_pressure` | 0.011% | 0.314 | Presión superficial |
| `wind_speed` | 0.008% | 0.290 | Velocidad del viento |

**Importancia**:
- Procesos físicos que afectan dispersión de contaminantes
- Relevantes para **generalización** a condiciones no vistas
- Útiles para **predicción a futuro** (escenarios meteorológicos)

---

### 4. Features Temporales

Importancia muy baja, excepto `day_of_year`:

| Feature | RF Importance | MI Score | Descripción |
|---------|---------------|----------|-------------|
| `day_of_year` | 0.009% | **0.305** | Día del año (1-365) |
| `season` | 0.001% | 0.157 | Estación del año |
| `day_of_week` | 0.007% | **0.000** | Día de la semana (0-6) |
| `is_weekend` | 0.001% | **0.000** | Fin de semana (0/1) |

**Nota**: `day_of_week` y `is_weekend` tienen MI=0 (no aportan información independiente).

---

## 📁 Datasets Generados

### 1. Selección Agresiva (Solo Lags)
**Archivo**: `data/processed/sinca_features_selected.csv`

- **Features seleccionadas**: 4 (solo lag features)
- **Tamaño**: 2.3 MB
- **Criterio**: Importancia RF > 0.001 AND MI > 0.01

**Features**:
1. `pm25_lag1`
2. `pm25_diff1`
3. `pm25_ma7`
4. `pm25_std7`

**Uso recomendado**:
- ✅ Predicción pura (máxima precisión)
- ✅ Benchmark de performance
- ❌ Interpretabilidad física
- ❌ Generalización espacial

---

### 2. Selección Balanceada (Recomendada)
**Archivo**: `data/processed/sinca_features_balanced.csv`

- **Features seleccionadas**: 13
- **Tamaño**: 4.35 MB
- **Criterio**: Balance entre precisión e interpretabilidad

**Features por categoría**:

#### Lag Features (4)
1. `pm25_lag1` - PM2.5 ayer
2. `pm25_diff1` - Cambio diario
3. `pm25_ma7` - Promedio 7 días
4. `pm25_std7` - Volatilidad 7 días

#### Satellite Features (2)
5. `modis_aod` - Aerosol Optical Depth
6. `s5p_no2` - Dióxido de nitrógeno

#### Meteorological Features (5)
7. `era5_total_precipitation_hourly` - Precipitación
8. `precipitation_sum7` - Precipitación acumulada 7 días
9. `era5_surface_pressure` - Presión superficial
10. `relative_humidity` - Humedad relativa
11. `wind_speed` - Velocidad del viento

#### Temporal Features (2)
12. `day_of_year` - Día del año (estacionalidad)
13. `season` - Estación del año

**Uso recomendado**:
- ✅ Predicción + interpretabilidad
- ✅ Generalización espacial (nuevas estaciones)
- ✅ Análisis de causas físicas
- ✅ **RECOMENDADO PARA MODELADO ML**

---

## 📊 Visualizaciones Generadas

### 1. Feature Importance (Top 20)
**Archivo**: `reports/figures/feature_importance_top20.png`

Muestra las 20 features más importantes según Random Forest. Dominancia clara de `pm25_lag1` (66%).

### 2. Correlation Heatmap (Top 20)
**Archivo**: `reports/figures/correlation_heatmap_top20.png`

Matriz de correlación entre las top 20 features. Identifica redundancias.

### 3. Correlation with Target (Top 20)
**Archivo**: `reports/figures/correlation_with_target.png`

Correlación absoluta de cada feature con PM2.5.

---

## 🎯 Rankings Completos

**Archivo**: `data/processed/feature_rankings.csv`

Contiene para cada feature:
- `importance`: Feature importance de Random Forest
- `mi_score`: Mutual Information Score

---

## 💡 Insights Clave

### 1. Autocorrelación Extrema
PM2.5 es **altamente autocorrelacionado**:
- `pm25_lag1` tiene 66% de importancia
- Las 3 top lag features suman 99.6% de importancia

**Implicación**:
- Modelos simples (Linear Regression) lograrán buen R² solo con lags
- Para ML avanzado, agregar features físicas mejorará generalización

---

### 2. Features Satelitales: Baja Importancia, Alto MI

Aunque tienen baja importancia RF (0.008%), sus MI scores son altos (0.32-0.35).

**Interpretación**:
- RF las ignora porque las lags son más fáciles de usar
- Pero contienen información **complementaria** (no correlacionada con lags)
- Modelos lineales o GLMs podrían beneficiarse más

**Recomendación**: **Mantenerlas en el dataset**

---

### 3. Meteorología: Contexto Físico

Features meteorológicas tienen importancia muy baja individualmente, pero:
- Explican **mecanismos causales** (dispersión, inversión térmica)
- Mejoran **generalización** a condiciones meteorológicas extremas
- Relevantes para **predicción a futuro** (sin lags disponibles)

**Recomendación**: Mantener top 5 meteorológicas

---

### 4. Temporal: Estacionalidad Capturada por Lags

Features temporales (`day_of_week`, `is_weekend`) tienen MI=0:
- La autocorrelación de PM2.5 ya captura patrones semanales
- `day_of_year` y `season` tienen algo de información independiente

**Recomendación**: Solo mantener `day_of_year` y `season`

---

## 🚨 Consideraciones Importantes

### 1. Data Leakage en Producción

Si el objetivo es **predicción a futuro**:
- ❌ **NO usar** `pm25_lag1` en producción (no disponible en tiempo real)
- ✅ Usar solo features exógenas (satelitales, meteorológicas, temporales)

### 2. Dos Escenarios de Modelado

#### Escenario A: Predicción con Lags (Nowcasting)
**Objetivo**: Predecir PM2.5 de **hoy** usando datos de **ayer**.

**Dataset**: `sinca_features_balanced.csv`

**Features clave**:
- `pm25_lag1`, `pm25_ma7`, `pm25_diff1`, `pm25_std7`
- Meteorología actual
- Satelitales actuales

**Uso**: Sistema de alerta temprana (predicción día siguiente).

---

#### Escenario B: Predicción sin Lags (Forecasting)
**Objetivo**: Predecir PM2.5 usando **solo features exógenas**.

**Dataset**: Filtrar lag features de `sinca_features_balanced.csv`

**Features clave**:
- Satelitales: `modis_aod`, `s5p_no2`
- Meteorología: precipitación, presión, humedad, viento
- Temporal: `day_of_year`, `season`

**Uso**: Predicción a largo plazo, generalización espacial.

---

## 📋 Datasets Comparativos

| Dataset | Features | Tamaño | Uso Recomendado |
|---------|----------|--------|-----------------|
| `sinca_features_engineered.csv` | 33 | 9.25 MB | Exploración, experimentación |
| `sinca_features_selected.csv` | 4 | 2.3 MB | Benchmark (solo lags) |
| `sinca_features_balanced.csv` | **13** | **4.35 MB** | **Modelado ML (recomendado)** |

---

## 🎯 Próximos Pasos

### 1. Modelado con Lags (Nowcasting)
**Dataset**: `sinca_features_balanced.csv`

**Modelos sugeridos**:
- Baseline: Linear Regression
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- LSTM (captura secuencias temporales)

**Métricas esperadas**:
- R² > 0.80 (por autocorrelación fuerte)
- RMSE < 10 μg/m³

---

### 2. Modelado sin Lags (Forecasting)
**Dataset**: `sinca_features_balanced.csv` (filtrar lag features)

**Modelos sugeridos**:
- Ridge/Lasso Regression (regularización)
- Random Forest
- Gradient Boosting
- Neural Networks

**Métricas esperadas**:
- R² ~ 0.40-0.60 (más desafiante)
- RMSE ~ 15-20 μg/m³

---

### 3. Análisis de Importancia Real

Entrenar modelos con:
1. Solo lags
2. Solo exógenas (satelitales + meteorología)
3. Ambas (balanceado)

Comparar performance y feature importance.

---

### 4. Feature Engineering Adicional

Considerar:
- **Interacciones entre satelitales y meteorología**
  - `modis_aod × relative_humidity`
  - `s5p_no2 × wind_speed`

- **Lags de features exógenas**
  - `modis_aod_lag1`
  - `wind_speed_lag1`

- **Features cíclicas** (sin/cos)
  - `sin(2π × day_of_year / 365)`
  - `cos(2π × day_of_year / 365)`

---

## ✅ Resumen Ejecutivo

### Hallazgo Principal
PM2.5 es **extremadamente autocorrelacionado**. `pm25_lag1` solo explica 66% de la varianza.

### Recomendación
Usar **`sinca_features_balanced.csv`** con **13 features**:
- 4 lag features (autocorrelación)
- 2 satelitales (información remota)
- 5 meteorológicas (contexto físico)
- 2 temporales (estacionalidad)

### Siguiente Paso
**Modelado ML** con dos estrategias:
1. **Con lags** (nowcasting, R² > 0.80)
2. **Sin lags** (forecasting, R² ~ 0.50)

---

**Script**: `src/data_processing/feature_selection.py`
**Visualizaciones**: `reports/figures/*.png`
**Rankings**: `data/processed/feature_rankings.csv`

---

**Estado**: ✅ COMPLETADO
**Siguiente paso**: Modelado ML (baseline models)
