# Feature Engineering - Resumen

**Fecha**: 13 de noviembre de 2025
**Dataset**: SINCA + Satélite (ERA5, MODIS, Sentinel-5P)

---

## 📊 Dataset Final

**Archivo**: `data/processed/sinca_features_engineered.csv`

### Métricas Generales
- **Registros**: 16,344 (de 16,400 originales)
- **Columnas**: 47 (22 originales + 25 engineered)
- **Tamaño**: 9.25 MB
- **Periodo**: 2019-01-08 → 2025-11-11 (2,499 días)
- **Estaciones**: 8

### Distribución por Estación
| Estación | Registros |
|----------|-----------|
| Independencia | 2,491 |
| Parque O'Higgins | 2,460 |
| Cerro Navia | 2,436 |
| Pudahuel | 2,432 |
| Las Condes | 2,413 |
| Talagante | 1,921 |
| Cerrillos II | 1,287 |
| El Bosque | 904 |

---

## 🎯 Variable Target

**PM2.5** (μg/m³):
- Media: **34.52**
- Mediana: **23.00**
- Desviación estándar: **28.84**
- Rango: **2.00 - 212.00**

---

## 🔧 Features Creadas (25 nuevas)

### 1. Wind-Derived Features (3)
Derivadas de componentes u/v del viento ERA5.

| Feature | Descripción | Unidad |
|---------|-------------|--------|
| `wind_speed` | Magnitud del vector viento: √(u² + v²) | m/s |
| `wind_direction_rad` | Dirección del viento: atan2(v, u) | radianes |
| `wind_direction_deg` | Dirección del viento convertida | grados (0-360°) |

**Importancia**: La velocidad y dirección del viento afectan la dispersión de contaminantes.

---

### 2. Temporal Features (5)
Características cíclicas y estacionales.

| Feature | Descripción | Valores |
|---------|-------------|---------|
| `day_of_week` | Día de la semana | 0-6 (0=Lunes) |
| `is_weekend` | Indicador de fin de semana | 0/1 |
| `season` | Estación del año (hemisferio sur) | 1=verano, 2=otoño, 3=invierno, 4=primavera |
| `day_of_year` | Día del año | 1-365 |
| `quarter` | Trimestre | 1-4 |

**Importancia**: Patrones semanales (tráfico) y estacionales (calefacción, inversión térmica).

---

### 3. Lag Features (6)
Features de rezago temporal de PM2.5, calculadas **por estación**.

| Feature | Descripción | Ventana |
|---------|-------------|---------|
| `pm25_lag1` | PM2.5 del día anterior | 1 día |
| `pm25_lag7` | PM2.5 de hace una semana | 7 días |
| `pm25_ma7` | Promedio móvil 7 días | 7 días |
| `pm25_ma30` | Promedio móvil 30 días | 30 días |
| `pm25_std7` | Volatilidad 7 días (desv. std.) | 7 días |
| `pm25_diff1` | Cambio respecto al día anterior | 1 día |

**Importancia**: Captura inercia y tendencias de contaminación. PM2.5 es altamente autocorrelacionado.

**Nota**: Se eliminaron 56 registros (0.3%) por NaNs en lag features (primeros 7 días de cada estación).

---

### 4. Meteorological Features (5)
Variables meteorológicas derivadas de ERA5.

| Feature | Descripción | Unidad | Fórmula |
|---------|-------------|--------|---------|
| `temperature_celsius` | Temperatura en Celsius | °C | T(K) - 273.15 |
| `dewpoint_celsius` | Punto de rocío en Celsius | °C | Td(K) - 273.15 |
| `relative_humidity` | Humedad relativa | % | Magnus formula |
| `surface_pressure_hpa` | Presión superficial | hPa | P(Pa) / 100 |
| `precipitation_sum7` | Precipitación acumulada 7 días | mm | Suma móvil |

**Fórmula Magnus (Humedad Relativa)**:
```
RH = 100 × exp((17.625×Td) / (243.04+Td)) / exp((17.625×T) / (243.04+T))
```

**Importancia**:
- Temperatura e inversión térmica afectan dispersión
- Humedad afecta formación de aerosoles secundarios
- Presión relacionada con estabilidad atmosférica
- Precipitación limpia la atmósfera

---

### 5. Interaction Features (4)
Interacciones no lineales entre variables.

| Feature | Descripción | Componentes |
|---------|-------------|-------------|
| `temp_aod_interaction` | Aerosoles × Temperatura | temperature_celsius × modis_aod |
| `wind_no2_interaction` | Dispersión de contaminantes | wind_speed × s5p_no2 |
| `humidity_aod_interaction` | Aerosoles × Humedad | relative_humidity × modis_aod |
| `atmospheric_stability` | Estabilidad atmosférica | surface_pressure_hpa × temperature_celsius |

**Importancia**: Captura efectos combinados (ej: alta temperatura + alta AOD indica estancamiento de contaminantes).

---

### 6. Spatial Features (2)
Características geográficas de las estaciones.

| Feature | Descripción | Unidad |
|---------|-------------|--------|
| `distance_to_center_km` | Distancia a Plaza de Armas (-33.4372, -70.6506) | km |
| `elevation_normalized` | Elevación normalizada (z-score) | - |

**Importancia**:
- Distancia al centro captura gradiente urbano (tráfico, densidad)
- Elevación afecta inversión térmica

---

## 📈 Estadísticas de Completitud

### Features Originales (22)
- **Target**: `pm25` - 100% completo
- **ERA5**: 6 variables meteorológicas - 100% completo
- **MODIS**: `modis_aod` - 100% completo
- **Sentinel-5P**: `s5p_no2` - 100% completo
- **Metadatos**: `estacion`, `lat`, `lon`, `elevation`, `datetime` - 100% completo

### Features Engineered (25)
- **Valores faltantes eliminados**: 56 registros (0.3%)
  - `pm25_lag7`: 56 NaNs (primeros 7 días por estación)
  - `pm25_lag1`, `pm25_diff1`, `pm25_std7`: 8 NaNs (primer día por estación)

- **Resto de features**: 100% completo después de dropna()

---

## 🔍 Variables Disponibles para Modelado

### Target (1)
- `pm25` - Concentración de PM2.5 (μg/m³)

### Features Predictoras (46)

#### Meteorología ERA5 (6)
- `era5_temperature_2m`
- `era5_dewpoint_temperature_2m`
- `era5_surface_pressure`
- `era5_u_component_of_wind_10m`
- `era5_v_component_of_wind_10m`
- `era5_total_precipitation_hourly`

#### Satelital (2)
- `modis_aod` - Aerosol Optical Depth
- `s5p_no2` - Dióxido de nitrógeno

#### Wind-Derived (3)
- `wind_speed`
- `wind_direction_rad`
- `wind_direction_deg`

#### Temporal (5)
- `day_of_week`
- `is_weekend`
- `season`
- `day_of_year`
- `quarter`

#### Lag Features (6)
- `pm25_lag1`
- `pm25_lag7`
- `pm25_ma7`
- `pm25_ma30`
- `pm25_std7`
- `pm25_diff1`

#### Meteorological Derived (5)
- `temperature_celsius`
- `dewpoint_celsius`
- `relative_humidity`
- `surface_pressure_hpa`
- `precipitation_sum7`

#### Interaction (4)
- `temp_aod_interaction`
- `wind_no2_interaction`
- `humidity_aod_interaction`
- `atmospheric_stability`

#### Spatial (2)
- `distance_to_center_km`
- `elevation_normalized`

#### Metadatos (no usar como features) (13)
- `datetime`, `date`, `year`, `month`, `day`
- `estacion`, `lat`, `lon`, `elevation`
- `validado`, `pm25_validado`, `pm25_preliminar`
- `archivo`

---

## 🎯 Próximos Pasos

### 1. EDA de Features Engineered
- Distribuciones de nuevas features
- Correlaciones con PM2.5
- Análisis de importancia preliminar
- Detección de outliers

### 2. Feature Selection
- Eliminar features redundantes o de baja importancia
- Análisis de multicolinealidad
- Validación cruzada de features

### 3. Modelado ML
- Baseline models (Linear Regression, Random Forest)
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Deep Learning (LSTM para series temporales)
- Ensemble models

### 4. Validación
- Split temporal (train/val/test)
- Cross-validation por estación
- Métricas: RMSE, MAE, R²

---

## ✅ Checklist de Procesamiento

- [x] Descarga de datos SINCA (13 estaciones)
- [x] Limpieza y consolidación SINCA
- [x] Descarga de datos satelitales (ERA5, MODIS, S5P)
- [x] Integración espacial-temporal
- [x] **Feature Engineering** ← **COMPLETADO**
- [ ] EDA completo
- [ ] Feature Selection
- [ ] Modelado ML
- [ ] Validación y evaluación
- [ ] Deployment

---

## 📝 Notas Técnicas

### Consideraciones para Modelado
1. **Lag features altamente predictivas**: `pm25_lag1` y `pm25_ma7` probablemente dominen el modelo. Considerar:
   - Entrenar modelo con y sin lags para comparar
   - Evaluar si el objetivo es predicción pura o interpretabilidad

2. **Temporal split obligatorio**: NO usar K-Fold aleatorio (data leakage). Usar:
   - TimeSeriesSplit
   - Walk-forward validation
   - Train: 2019-2022, Val: 2023, Test: 2024-2025

3. **Estaciones con datos desbalanceados**:
   - El Bosque solo tiene 904 registros (vs 2,491 en Independencia)
   - Considerar pesos por estación o stratified sampling

4. **Features cíclicas**:
   - `day_of_week`, `day_of_year`, `month` son cíclicas
   - Considerar transformación sin/cos para capturar ciclicidad

5. **Multicolinealidad esperada**:
   - `temperature_celsius` vs `era5_temperature_2m`
   - `wind_speed` vs componentes u/v
   - Usar regularización (Ridge/Lasso) o tree-based models

---

**Script**: `src/data_processing/feature_engineering.py`
**Dataset Input**: `data/processed/sinca_satellite_complete.csv`
**Dataset Output**: `data/processed/sinca_features_engineered.csv`

---

**Estado**: ✅ COMPLETADO
**Siguiente paso**: EDA de features engineered o modelado ML directo
