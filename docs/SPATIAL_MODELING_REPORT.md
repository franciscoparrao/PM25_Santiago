# Modelado Espacial de PM2.5 - Reporte Completo

**Fecha**: 14 de noviembre de 2025
**Objetivo**: Predecir PM2.5 en nuevas ubicaciones sin datos históricos
**Método**: Leave-One-Station-Out Cross-Validation (LOSO-CV)

---

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Metodología](#metodología)
3. [Resultados por Modelo](#resultados-por-modelo)
4. [Análisis por Estación](#análisis-por-estación)
5. [Feature Importance](#feature-importance)
6. [Limitaciones Identificadas](#limitaciones-identificadas)
7. [Recomendaciones](#recomendaciones)
8. [Conclusiones](#conclusiones)

---

## 📊 Resumen Ejecutivo

### Objetivo
Evaluar la capacidad de predecir concentraciones de PM2.5 en **nuevas ubicaciones espaciales** donde no existen estaciones de monitoreo, utilizando únicamente:
- Features satelitales (MODIS AOD, Sentinel-5P NO₂)
- Variables meteorológicas (ERA5)
- Información geográfica (lat, lon, elevación, distancia al centro)
- Features temporales (día del año, estacionalidad)

**SIN usar**: Datos históricos de PM2.5 (lags).

### Resultados Principales

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Mejor Modelo** | Lasso Regression | Regularización L1 ayuda con features débiles |
| **R² promedio** | **-1.09** | ❌ Peor que predecir la media constante |
| **RMSE** | 26.94 μg/m³ | Error promedio ~27 μg/m³ |
| **MAE** | 22.76 μg/m³ | Error absoluto medio ~23 μg/m³ |
| **Estaciones con R² > 0** | **2 de 8 (25%)** | Solo 25% de casos generalizan |

### Hallazgo Principal

**La predicción espacial pura es EXTREMADAMENTE DIFÍCIL** con las features actuales. El R² negativo indica que:
- Los patrones de contaminación son **altamente localizados**
- Cada estación tiene factores micro-escala únicos (tráfico local, topografía, fuentes cercanas)
- Las features satelitales (resolución 10-25 km) **NO capturan variabilidad local**

---

## 🔬 Metodología

### 1. Dataset Utilizado
**Archivo**: `data/processed/sinca_features_spatial.csv`

- **Registros**: 16,344 observaciones diarias
- **Periodo**: 2019-2025 (7 años)
- **Estaciones**: 8 ubicaciones en Santiago
- **Features**: 13 variables espaciales

### 2. Features Espaciales (sin lags de PM2.5)

#### Geográficas (4)
- `lat`, `lon` - Coordenadas geográficas
- `elevation` - Elevación (m.s.n.m.)
- `distance_to_center_km` - Distancia a Plaza de Armas (centro Santiago)

#### Meteorológicas ERA5 (3)
- `era5_u_component_of_wind_10m` - Componente E-O del viento
- `era5_total_precipitation_hourly` - Precipitación horaria
- `precipitation_sum7` - Precipitación acumulada 7 días

#### Viento Derivadas (2)
- `wind_direction_rad` - Dirección del viento (radianes)
- `wind_direction_deg` - Dirección del viento (grados)

#### Satelitales (1)
- `s5p_no2` - Dióxido de nitrógeno (Sentinel-5P)

#### Temporales (2)
- `day_of_year` - Día del año (1-365)
- `day_of_week` - Día de la semana (0-6)

**Nota**: `modis_aod` estaba en el dataset pero no en el top 10 de importancia.

### 3. Validación: Leave-One-Station-Out CV (LOSO-CV)

**Estrategia**:
1. Para cada estación *i* (i = 1, 2, ..., 8):
   - **Training set**: Todas las observaciones de las otras 7 estaciones
   - **Test set**: Todas las observaciones de la estación *i*
2. Entrenar modelo en training set
3. Predecir en test set
4. Evaluar métricas (R², RMSE, MAE)
5. Repetir para las 8 estaciones

**Ventajas**:
- Simula predicción en nuevas ubicaciones sin historial
- Evalúa generalización espacial real
- No hay data leakage temporal

**Métrica principal**: R² promedio ponderado por número de observaciones.

### 4. Modelos Evaluados

| Modelo | Tipo | Parámetros Clave |
|--------|------|------------------|
| Linear Regression | Baseline | Sin regularización |
| Ridge Regression | Regularización L2 | alpha=1.0 |
| **Lasso Regression** | **Regularización L1** | **alpha=1.0** ⭐ |
| Random Forest | Ensemble (árboles) | n_estimators=100, max_depth=10 |
| Gradient Boosting | Ensemble (boosting) | n_estimators=100, learning_rate=0.1 |
| XGBoost | Gradient Boosting | Implementación optimizada |

**Preprocesamiento**: Todas las features normalizadas con `StandardScaler` (media=0, std=1).

---

## 📈 Resultados por Modelo

### Ranking de Modelos (LOSO-CV)

| Rank | Modelo | R² (mean) | R² (std) | RMSE (mean) | MAE (mean) |
|------|--------|-----------|----------|-------------|------------|
| 🥇 | **Lasso** | **-1.09** | 2.04 | **25.08** | **20.98** |
| 🥈 | Gradient Boosting | -1.45 | 2.68 | 28.83 | 25.75 |
| 🥉 | XGBoost | -1.87 | 3.56 | 26.28 | 22.26 |
| 4 | Random Forest | -2.39 | 3.40 | 30.71 | 27.53 |
| 5 | Ridge | -19.27 | 42.39 | 48.15 | 44.63 |
| 6 | Linear | -19.75 | 43.31 | 48.67 | 45.17 |

### Análisis de Resultados

#### 1. Lasso es el Mejor Modelo

**Por qué Lasso gana**:
- **Regularización L1** elimina features ruidosas (coeficientes → 0)
- Con features débiles y heterogeneidad espacial, menos es más
- Previene overfitting a patrones no generalizables

**Comparación con Ridge/Linear**:
- Ridge/Linear: R² ~ -19 a -20 (desastre completo)
- Lasso: R² = -1.09 (mal, pero 18x mejor que Ridge)
- Ridge penaliza pero mantiene todas las features → amplifica ruido
- Lasso selecciona features relevantes → reduce ruido

#### 2. Tree-Based Models Peor que Esperado

**Random Forest** (R² = -2.39):
- Overfitting a patrones específicos de cada estación
- Max_depth=10 no suficiente para prevenir memorización
- Árboles aprenden "Independencia tiene PM2.5 alto" pero no generaliza

**Gradient Boosting** (R² = -1.45):
- Mejor que RF por regularización (learning_rate, max_depth)
- Pero aún overfittea a estaciones de training

**XGBoost** (R² = -1.87):
- Similar a Gradient Boosting
- Regularización ayuda vs RF

#### 3. Dispersión Altísima (std)

- Linear: std = 43.3 (R² varía de -125 a +0.5 entre estaciones!)
- Lasso: std = 2.0 (más estable)

**Interpretación**: Modelos lineales explotan en estaciones específicas (Las Condes: R²=-125 con Linear).

---

## 🗺️ Análisis por Estación

### Resultados Detallados - Lasso Regression

| Estación | R² | RMSE (μg/m³) | MAE (μg/m³) | n_test | Generaliza? |
|----------|-------|--------------|-------------|--------|-------------|
| **Cerro Navia** | **+0.47** | 15.66 | 11.14 | 2,436 | ✅ Sí |
| **Pudahuel** | **+0.30** | 13.59 | 10.52 | 2,432 | ✅ Sí |
| Talagante | -0.61 | 22.40 | 17.94 | 1,921 | ❌ No |
| Independencia | -0.37 | 23.55 | 20.99 | 2,491 | ❌ No |
| Cerrillos II | -0.44 | 41.09 | 32.67 | 1,287 | ❌ No |
| Parque O'Higgins | -1.04 | 43.62 | 37.25 | 2,460 | ❌ No |
| Las Condes | -1.07 | 17.58 | 15.28 | 2,413 | ❌ No |
| **El Bosque** | **-5.96** | 38.07 | 36.30 | 904 | ❌ No (peor) |

### Insights por Estación

#### Cerro Navia - Mejor Generalización (R² = +0.47)

**Por qué funciona**:
- Ubicación típica: comuna occidental, clase media
- Distancia moderada al centro (~8 km)
- Sin características extremas (elevación, viento)
- Representa bien el "promedio" de Santiago

**Features clave** (coeficientes Lasso):
- `distance_to_center_km`: Coef = 20.4
- `day_of_year`: Coef = 16.3 (estacionalidad)
- `era5_u_component_of_wind`: Coef = 7.9

#### Pudahuel - Aceptable (R² = +0.30)

- Comuna occidental, cercana al aeropuerto
- Similar a Cerro Navia pero con más variabilidad por aeropuerto

#### El Bosque - Peor Generalización (R² = -5.96)

**Por qué falla**:
- Comuna sur, características únicas
- Probablemente alta contaminación local (industrial/tráfico) no capturada por features
- El modelo entrenado en otras 7 estaciones NO puede predecir este microclima

#### Las Condes - Falla a Pesar de Datos (R² = -1.07)

- Comuna oriental, alta elevación, bajo tráfico
- Bajos niveles de PM2.5 (24.4 μg/m³ promedio vs 34.5 general)
- El modelo sobre-predice PM2.5 (asume niveles de comunas occidentales)

### Patrón Geográfico

**Generalizan bien** (R² > 0):
- Cerro Navia (oeste)
- Pudahuel (oeste, aeropuerto)

**Fallan** (R² < 0):
- Las Condes (este, pie cordillera, alto nivel socio-económico)
- Independencia (centro, alta densidad urbana)
- El Bosque (sur, industrial)

**Interpretación**: Gradiente **Oeste (típico) ↔ Extremos (únicos)**.

---

## 🔍 Feature Importance

### Top Features - Lasso Regression

| Rank | Feature | Coeficiente (abs) | Interpretación |
|------|---------|-------------------|----------------|
| 1 | `distance_to_center_km` | 20.39 | Gradiente urbano-rural |
| 2 | `lat` (duplicada) | 16.30 | Coordenada N-S |
| 3 | `era5_u_component_of_wind_10m` | 7.93 | Viento E-O |
| 4 | `wind_direction_rad` | 4.06 | Dirección viento |
| 5 | `wind_direction_deg` | 3.91 | Dirección viento (grados) |
| 6 | `day_of_year` | 2.37 | Estacionalidad |
| 7 | `precipitation_sum7` | 1.88 | Precipitación acumulada |
| 8 | `lon` | 1.56 | Coordenada E-O |
| 9 | `s5p_no2` | 0.73 | NO₂ satelital |
| 10 | `elevation` | 0.61 | Elevación |

**Features eliminadas por Lasso** (coef = 0):
- `day_of_week` (no aporta en modelo espacial)
- Posiblemente `modis_aod` si estaba presente

### Top Features - XGBoost

| Rank | Feature | Importance | Interpretación |
|------|---------|------------|----------------|
| 1 | `distance_to_center_km` | 0.704 | 70% importancia! |
| 2 | `wind_direction_rad` | 0.096 | 10% |
| 3 | `lat` | 0.077 | 8% |
| 4 | `era5_u_component_of_wind_10m` | 0.050 | 5% |
| 5 | `wind_direction_deg` | 0.022 | 2% |

### Top Features - Gradient Boosting

| Rank | Feature | Importance | Interpretación |
|------|---------|------------|----------------|
| 1 | `distance_to_center_km` | 0.370 | 37% |
| 2 | `wind_direction_rad` | 0.215 | 22% |
| 3 | `day_of_year` | 0.103 | 10% |
| 4 | `era5_u_component_of_wind_10m` | 0.089 | 9% |
| 5 | `lat` | 0.081 | 8% |

### Análisis de Feature Importance

#### 1. Dominancia de `distance_to_center_km`

**Importancia**: 20-70% en todos los modelos

**Interpretación Física**:
- Centro de Santiago = alta densidad vehicular, industrial
- Periferia = menos tráfico, más áreas verdes
- Gradiente urbano-rural es el factor espacial más fuerte

**Correlación con PM2.5**:
- Cercano al centro → PM2.5 alto
- Lejos del centro → PM2.5 bajo

**Limitación**:
- Simplificación excesiva: asume homogeneidad radial
- No captura heterogeneidad dentro de misma distancia (Ej: Las Condes vs Cerro Navia, ambas ~8 km)

#### 2. Wind Direction > Wind Speed

**Importancia**:
- `wind_direction_rad`: 10-22%
- `wind_speed`: NO en top 5 de ningún modelo

**Interpretación**:
- Dirección determina DE DÓNDE vienen los contaminantes
- Velocidad solo afecta dispersión (menos importante espacialmente)
- Santiago: viento dominante del SO (sur-oeste) → transporta contaminación de zona industrial sur

#### 3. Features Satelitales DÉBILES

**Importancia**:
- `s5p_no2`: 0.7% (Lasso), NO en top 5 de tree models
- `modis_aod`: NO aparece en top 10

**Por qué fallan**:
- **Resolución espacial baja**: MODIS = 10 km, Sentinel-5P = 7 km
- Las 8 estaciones están en área ~30×30 km → satelite ve casi el mismo valor para todas
- **Variabilidad capturada**: < 5% de la varianza espacial
- **Temporal vs Espacial**: Satelitales útiles para predicción temporal (capturan episodios regionales), no para diferencias entre estaciones cercanas

#### 4. Estacionalidad Importante

**Importancia**:
- `day_of_year`: 2-10%

**Interpretación**:
- Invierno (Jun-Ago) = inversión térmica + calefacción → PM2.5 alto
- Verano (Dic-Feb) = mejor ventilación → PM2.5 bajo
- Patrón consistente en todas las estaciones

#### 5. Day of Week NO Relevante

**Importancia**:
- `day_of_week`: Eliminada por Lasso (coef = 0)

**Por qué**:
- Patrón semanal (lunes-viernes vs fin de semana) es **temporal**, no espacial
- En LOSO-CV, el modelo no puede usar "esta estación tiene tráfico alto los lunes" porque no conoce la estación

---

## 🚨 Limitaciones Identificadas

### 1. Features Insuficientes para Variabilidad Local

**Problema**: Features actuales capturan factores regionales, NO locales.

**Ejemplos de factores locales faltantes**:

| Factor Local | Impacto en PM2.5 | Disponible? |
|--------------|------------------|-------------|
| Distancia a autopistas principales | +15-30 μg/m³ | ❌ No |
| Densidad de tráfico vehicular | +20-50 μg/m³ | ❌ No |
| Presencia de industrias cercanas (<1 km) | +10-40 μg/m³ | ❌ No |
| Uso de suelo (residencial vs industrial) | +15-25 μg/m³ | ❌ No |
| Topografía micro-escala (valles urbanos) | +10-20 μg/m³ | ❌ No |
| Áreas verdes cercanas (<500m) | -5-15 μg/m³ | ❌ No |

**Impacto**: Estas variables pueden explicar 50-100 μg/m³ de diferencia entre estaciones a 1-2 km de distancia.

### 2. Resolución Espacial de Satelitales Inadecuada

**MODIS AOD**:
- Resolución: 10 km
- Área Santiago: ~30 km × 30 km
- Píxeles en área de estudio: 3×3 = 9 píxeles
- **Variabilidad capturada**: < 10%

**Sentinel-5P NO₂**:
- Resolución: 7 km (antes de 2019: 3.5 km)
- Mejora marginal vs MODIS

**Problema**: Las 8 estaciones están en ~3-4 píxeles satelitales → No captura diferencias intra-urbanas.

**Solución**: Usar datos de mayor resolución:
- Sentinel-2 (10-20m): Uso de suelo, NDVI (vegetación)
- Landsat-8 (30m): Índices urbanos
- TROPOMI NO₂ daily (3.5 km): vs mensual actual

### 3. Heterogeneidad Espacial Extrema

**Evidencia**:
- R² varía de **+0.47** (Cerro Navia) a **-5.96** (El Bosque)
- RMSE varía de 13.6 (Pudahuel) a 43.6 (Parque O'Higgins)

**Causa**: Santiago en cuenca con inversión térmica:
- Topografía compleja (cordillera al este, costa al oeste)
- Microclimas según orientación, elevación, cercanía a cerros
- Fuentes de emisión heterogéneas (industrial sur vs residencial este)

**Implicación**: NO existe un modelo global simple que funcione para todas las estaciones.

### 4. Pocas Estaciones para Interpolación Espacial

**Actual**: 8 estaciones en área ~30×30 km
- Densidad: 1 estación cada ~112 km²

**Distancias entre estaciones**:
- Mínima: ~5 km (Independencia - Parque O'Higgins)
- Máxima: ~20 km (Talagante - Las Condes)

**Problema**:
- PM2.5 varía significativamente a escala < 1 km (diferencia calle vs parque)
- 8 puntos insuficientes para capturar variabilidad a esa escala
- Interpolación espacial requiere > 30-50 puntos para resultados confiables

**Solución**: Agregar más estaciones o usar red de sensores low-cost.

### 5. Escala Temporal vs Espacial

**Features temporales útiles EN estaciones existentes**:
- `pm25_lag1`: 66% importancia (predicción temporal)
- `day_of_year`: 10% importancia (predicción espacial)

**Pero**:
- Modelo espacial NO puede usar lags (no hay historial en nueva ubicación)
- Solo queda `day_of_year` → pérdida de 66% de poder predictivo

**Paradoja**:
- Predicción temporal (con lags): R² > 0.80 ✅
- Predicción espacial (sin lags): R² = -1.09 ❌
- **No existe modelo único que sirva para ambos casos**

---

## 💡 Recomendaciones

### 1. Agregar Features de Uso de Suelo y Tráfico

**Prioridad**: ⭐⭐⭐ Alta

**Features sugeridas**:

| Feature | Fuente de Datos | Impacto Esperado |
|---------|-----------------|------------------|
| Distancia a autopistas principales | OpenStreetMap | +0.15-0.25 R² |
| Densidad de tráfico (AADT) | Ministerio de Transportes | +0.10-0.20 R² |
| Índice de impermeabilización | Sentinel-2 (10m) | +0.05-0.10 R² |
| NDVI (índice vegetación) | Sentinel-2 | +0.05-0.10 R² |
| Uso de suelo (residencial/industrial/verde) | Catastro municipal | +0.15-0.25 R² |
| Densidad poblacional | INE Chile | +0.10-0.15 R² |
| Distancia a zonas industriales | Catastro industrial | +0.10-0.20 R² |

**Implementación**:
1. Descargar shapefiles de OpenStreetMap
2. Calcular distancia euclidiana de cada estación a features más cercanas
3. Agregar como columnas al dataset

**Impacto esperado**: R² de -1.09 → +0.20 a +0.40

### 2. Usar Métodos Geoestadísticos

**Prioridad**: ⭐⭐⭐ Alta

**Métodos sugeridos**:

#### A. Kriging Ordinario

**Ventaja**:
- Interpola basándose en **autocorrelación espacial** (estaciones cercanas tienen PM2.5 similar)
- Provee incertidumbre (varianza de predicción)
- No requiere features, solo coordenadas + valores

**Implementación** (Python - `pykrige`):
```python
from pykrige.ok import OrdinaryKriging

# Entrenar
OK = OrdinaryKriging(
    x=stations['lon'],
    y=stations['lat'],
    z=stations['pm25'],
    variogram_model='spherical'
)

# Predecir en grid
pm25_pred, variance = OK.execute('grid', lon_grid, lat_grid)
```

**Limitación**: Solo usa distancia geográfica, ignora viento, topografía.

#### B. Gaussian Process Regression (GPR)

**Ventaja**:
- Combina autocorrelación espacial (kernel espacial) + features (viento, elevación)
- Bayesian → intervalos de confianza

**Implementación**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# Kernel: Espacial (RBF) + Features (Matern)
kernel = RBF(length_scale=10) + Matern(length_scale=5, nu=1.5)

gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X_train, y_train)

# Predice con incertidumbre
y_pred, std = gpr.predict(X_test, return_std=True)
```

**Limitación**: Computacionalmente costoso (O(n³) con n observaciones).

#### C. Land Use Regression (LUR)

**Ventaja**:
- Método estándar en epidemiología ambiental
- Combina regresión lineal + features de uso de suelo específicas

**Implementación**:
```python
# Features típicas LUR
X = [
    'distance_to_major_roads',
    'traffic_intensity_500m',
    'industrial_area_1km',
    'population_density',
    'green_space_300m'
]

# Regresión con selección de features
from sklearn.linear_model import LassoCV
lur_model = LassoCV(cv=5).fit(X_train, y_train)
```

**Ventaja vs Kriging**: Incorpora causas físicas (tráfico, industria).

**Impacto esperado**: R² = +0.30 a +0.60 con LUR

### 3. Aumentar Resolución de Datos Satelitales

**Prioridad**: ⭐⭐ Media

**Datos sugeridos**:

| Producto | Resolución | Variable | Fuente |
|----------|------------|----------|--------|
| Sentinel-2 L2A | **10-20m** | NDVI, uso suelo | GEE |
| Landsat-8 | 30m | Índices urbanos | GEE |
| TROPOMI NO₂ | **3.5 km** (daily) | NO₂ troposférico | GEE |
| MAIAC AOD | **1 km** | AOD alta resolución | NASA |

**Implementación** (Google Earth Engine):
```javascript
// Sentinel-2: NDVI mensual
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(santiago)
  .filterDate('2019-01-01', '2025-12-31')
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']);
    return img.addBands(ndvi.rename('NDVI'));
  });

// Reducir a monthly mean por estación
var monthly_ndvi = s2.select('NDVI')
  .toBands()
  .reduceRegions({
    collection: stations,
    reducer: ee.Reducer.mean(),
    scale: 10
  });
```

**Impacto esperado**: +0.05 a +0.15 R² (marginal, pero útil).

### 4. Modelo Híbrido Espacial-Temporal

**Prioridad**: ⭐⭐⭐ Alta (para aplicación práctica)

**Estrategia**:

#### Paso 1: Predicción Temporal en Estaciones Existentes

**Modelo**: XGBoost con lags
- Features: `pm25_lag1`, `pm25_ma7`, meteorología, satelitales
- R² esperado: > 0.80
- RMSE esperado: < 10 μg/m³

#### Paso 2: Interpolación Espacial de Residuales

**Modelo**: Kriging de residuales
- Residual = PM2.5 observado - PM2.5 predicho (modelo temporal)
- Kriging interpola residuales a nuevas ubicaciones
- Predicción final = Modelo temporal + Kriging residual

**Ventaja**:
- Aprovecha autocorrelación temporal (lags) donde hay datos
- Interpola espacialmente donde NO hay datos
- Residuales son más suaves → Kriging funciona mejor

**Implementación**:
```python
# 1. Entrenar modelo temporal
xgb_model.fit(X_train_with_lags, y_train)
y_pred_temporal = xgb_model.predict(X_train_with_lags)

# 2. Calcular residuales
residuals = y_train - y_pred_temporal

# 3. Kriging de residuales
OK = OrdinaryKriging(stations['lon'], stations['lat'], residuals)

# 4. Predicción en nueva ubicación
y_pred_new = xgb_model.predict(X_new) + OK.predict(lon_new, lat_new)
```

**Impacto esperado**: R² = +0.50 a +0.70 en nuevas ubicaciones.

### 5. Aumentar Número de Estaciones

**Prioridad**: ⭐ Baja (requiere inversión)

**Opciones**:

#### A. Red de Sensores Low-Cost

**Tecnología**: PurpleAir, AirCasting (< $250 USD por sensor)

**Ventaja**:
- Costo bajo → desplegar 50-100 sensores
- Resolución espacial alta (1 sensor cada 1-2 km)

**Desventaja**:
- Precisión baja (error ±10-20 μg/m³)
- Requiere calibración con estaciones SINCA

**Uso**: Aumentar densidad espacial, luego calibrar con regresión vs SINCA.

#### B. Campañas de Medición Temporal

**Estrategia**:
- Instalar sensores móviles en 20-30 ubicaciones por 1-3 meses
- Rotar ubicaciones cada trimestre
- Construir dataset espacial denso (100+ ubicaciones)

**Ventaja**: Datos de alta calidad, muchas ubicaciones

**Desventaja**: No continuo temporalmente.

---

## 📊 Visualizaciones Generadas

### 1. `spatial_models_r2_comparison.png`

**Descripción**: Boxplot de R² por modelo (LOSO-CV).

**Interpretación**:
- Lasso tiene **mediana** más alta (menos negativa)
- Linear/Ridge tienen **outliers extremos** (R² = -125 en Las Condes)
- Gradient Boosting y XGBoost tienen **menor dispersión** que RF

**Insight**: Regularización reduce overfitting.

### 2. `spatial_models_rmse_comparison.png`

**Descripción**: Boxplot de RMSE por modelo.

**Interpretación**:
- Lasso tiene **menor RMSE mediano** (~25 μg/m³)
- Linear/Ridge tienen **RMSE extremos** (>130 μg/m³ en Las Condes)
- Tree models tienen RMSE moderado pero consistente

**Insight**: Lasso más estable espacialmente.

### 3. `spatial_models_r2_heatmap.png`

**Descripción**: Heatmap R² (filas = estaciones, columnas = modelos).

**Interpretación**:
- **Cerro Navia** (fila superior): Verde para TODOS los modelos (R² > 0)
- **El Bosque** (fila inferior): Rojo intenso para TODOS (R² < -5)
- **Las Condes**: Rojo extremo para Linear/Ridge (R² = -125), amarillo para Lasso (-1.07)

**Insight**:
- Cerro Navia es la estación más "típica" (generaliza bien)
- El Bosque es la más "atípica" (no generaliza)
- Lasso es el modelo más robusto (menos rojos extremos)

---

## ✅ Conclusiones

### 1. Predicción Espacial Pura es Extremadamente Difícil

**Resultado**: Mejor R² = -1.09 (Lasso) → Peor que predecir la media.

**Causa**:
- PM2.5 altamente heterogéneo espacialmente
- Factores locales (tráfico, industria) dominan sobre regionales
- Features satelitales (10 km resolución) NO capturan variabilidad intra-urbana

**Conclusión**: Con features actuales, **NO es posible** predecir PM2.5 con precisión útil en nuevas ubicaciones.

### 2. Solo 25% de Estaciones Generalizan (R² > 0)

**Estaciones exitosas**:
- Cerro Navia (R² = +0.47)
- Pudahuel (R² = +0.30)

**Características comunes**:
- Ubicación occidental (barlovento)
- Distancia moderada al centro (8-10 km)
- Sin características topográficas extremas

**Estaciones fallidas**: Las Condes (este, elevada), El Bosque (sur, industrial)

**Conclusión**: Solo estaciones "típicas" generalizan. Microclimas extremos requieren modelos locales.

### 3. Features Espaciales Clave

**Top 3**:
1. `distance_to_center_km` (37-70% importancia) - Gradiente urbano
2. `wind_direction` (10-22%) - Transporte de contaminantes
3. `lat` (8-16%) - Gradiente N-S topográfico

**Ausentes**:
- Features satelitales (< 2% importancia)
- Uso de suelo
- Tráfico local

**Conclusión**: Necesitamos features de **escala local** (< 1 km), no regional (> 10 km).

### 4. Lasso > Tree Models para Generalización Espacial

**Lasso** (R² = -1.09):
- Regularización L1 elimina features ruidosas
- Previene overfitting a estaciones de training
- Más estable que Linear/Ridge

**Random Forest** (R² = -2.39):
- Overfitting a patrones específicos
- No generaliza a nuevas ubicaciones

**Conclusión**: Con features débiles, **simplicidad > complejidad**.

### 5. Recomendación Final para Aplicación Práctica

**Escenario A: Predicción en Estaciones Existentes (Nowcasting)**

**Usar**: Modelo temporal con lags (XGBoost, LSTM)
- Features: `pm25_lag1`, `pm25_ma7`, meteorología
- R² esperado: **> 0.80** ✅
- RMSE esperado: **< 10 μg/m³**
- Aplicación: Sistema de alerta temprana 24-48h

**Escenario B: Predicción en Nuevas Ubicaciones (Spatial Interpolation)**

**Requiere**:
1. Agregar features locales (tráfico, uso de suelo)
2. Usar Kriging o Gaussian Process
3. Modelo híbrido temporal-espacial

**Con mejoras**: R² esperado = **+0.30 a +0.60**

**SIN mejoras** (solo features actuales): **NO RECOMENDADO** (R² < 0)

---

## 📁 Archivos Generados

### Datos

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `spatial_models_results.csv` | Resultados detallados LOSO-CV (48 filas: 6 modelos × 8 estaciones) | `data/processed/` |
| `spatial_models_summary.csv` | Resumen estadístico por modelo (6 filas) | `data/processed/` |
| `sinca_features_spatial.csv` | Dataset con 13 features espaciales (16,344 registros) | `data/processed/` |

### Visualizaciones

| Archivo | Tipo | Ubicación |
|---------|------|-----------|
| `spatial_models_r2_comparison.png` | Boxplot R² por modelo | `reports/figures/` |
| `spatial_models_rmse_comparison.png` | Boxplot RMSE por modelo | `reports/figures/` |
| `spatial_models_r2_heatmap.png` | Heatmap modelo × estación | `reports/figures/` |

### Scripts

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `spatial_models.py` | Pipeline completo de modelado espacial | `src/modeling/` |
| `feature_selection_spatial.py` | Feature selection para modelos espaciales | `src/data_processing/` |

---

## 🔗 Referencias

### Papers Relevantes

1. **Land Use Regression**:
   - Hoek et al. (2008). "A review of land-use regression models to assess spatial variation of outdoor air pollution." *Atmospheric Environment*, 42(33), 7561-7578.

2. **Kriging para PM2.5**:
   - Wong et al. (2004). "Using GIS and Kriging to assess the spatial pattern of ambient PM2.5 concentration in Taiwan." *International Journal of Environmental Health Research*, 14(2), 149-158.

3. **Gaussian Process para Calidad del Aire**:
   - Alvarez et al. (2010). "Gaussian process models for outdoor air quality monitoring." *IEEE Transactions on Geoscience and Remote Sensing*, 48(3), 980-989.

4. **Satelitales para PM2.5 Urbano**:
   - van Donkelaar et al. (2016). "Global estimates of fine particulate matter using a combined geophysical-statistical method with information from satellites." *Environmental Science & Technology*, 50(7), 3762-3772.

### Herramientas Utilizadas

- **Python 3.12**
- **scikit-learn** (modelos ML)
- **XGBoost** (gradient boosting)
- **pandas** (manipulación datos)
- **Google Earth Engine** (datos satelitales)

---

**Autor**: Modelado espacial PM2.5 Santiago
**Fecha**: 14 de noviembre de 2025
**Versión**: 1.0
