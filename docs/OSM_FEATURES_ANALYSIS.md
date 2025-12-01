# Análisis: Por Qué las Features OSM Empeoraron el Modelo

**Fecha**: 14 de noviembre de 2025
**Resultado Inesperado**: Features OSM degradaron R² de -0.69 → -37.90 (Lasso)

---

## 📊 Resultados Observados

### Comparación Baseline vs Enhanced

| Modelo | Versión | Features | R² | RMSE | MAE |
|--------|---------|----------|-----|------|-----|
| **Lasso** | Baseline | 13 | **-0.69** | 25.08 | 20.98 |
| **Lasso** | Enhanced | 18 | **-37.90** ❌ | 63.00 | 59.43 |
| **XGBoost** | Baseline | 13 | **-1.87** | 26.28 | 22.26 |
| **XGBoost** | Enhanced | 18 | **-2.23** ❌ | 28.19 | 23.82 |

### Observaciones

1. **Lasso se degrada drásticamente** (-0.69 → -37.90): -5,361% peor
2. **XGBoost se degrada moderadamente** (-1.87 → -2.23): -19% peor
3. Ambos modelos empeoran, pero **Lasso colapsa**

---

## 🔍 Hipótesis: ¿Por Qué Empeoraron?

### Hipótesis 1: Multicolinealidad Extrema

**Problema**: Las features OSM están **altamente correlacionadas** entre sí y con features existentes.

**Features OSM agregadas**:
1. `dist_to_highway_km` - Distancia a autopista
2. `dist_to_primary_km` - Distancia a vía primaria
3. `road_density_500m` - Densidad vial 500m
4. `road_density_1km` - Densidad vial 1km
5. `highway_count_1km` - Número de autopistas 1km

**Correlaciones esperadas**:
- `dist_to_highway_km` ↔ `highway_count_1km`: Correlación negativa alta (-0.8 a -0.9)
- `road_density_500m` ↔ `road_density_1km`: Correlación positiva alta (0.9+)
- `road_density` ↔ `distance_to_center_km` (ya existente): Negativa alta (-0.7+)

**Efecto en Lasso**:
- Lasso regularización L1 → Penaliza coeficientes altos
- Con features correlacionadas, Lasso puede "saltar" entre features equivalentes
- Resultado: inestabilidad numérica, coeficientes extremos

**Efecto en XGBoost**:
- Tree-based models más robustos a correlación
- Pero aún así sufren si features redundantes confunden splits

---

### Hipótesis 2: Features Estáticas vs Variables Temporales

**Problema**: Features OSM son **estáticas** (no varían en el tiempo para cada estación).

**Implicación**:
- En LOSO-CV, modelo entrena en 7 estaciones, predice en 1
- Features estáticas solo capturan diferencia ENTRE estaciones, no DENTRO
- Si estación de test es muy diferente → modelo sobre-generaliza patrones de training

**Ejemplo**:
- Las Condes: `dist_to_highway_km` = 0.13 km, muy cercana
- Talagante: `dist_to_highway_km` = 3.01 km, muy lejana

Modelo aprende: "cerca de autopista → PM2.5 alto"
Pero en Las Condes: PM2.5 es **bajo** (zona residencial alta, poco tráfico local)

→ Modelo falla porque **contexto local** domina sobre proximidad a autopista

---

### Hipótesis 3: Escala de Features Incompatible

**Problema**: Features OSM tienen escalas muy diferentes.

**Rangos observados** (de estadísticas):
- `dist_to_highway_km`: 0.13 - 3.01 km (range: 2.88)
- `dist_to_primary_km`: 0.008 - 2.94 km (range: 2.93)
- `road_density_500m`: 0.5 - 53.7 km/km² (range: 53.2) ⚠️
- `road_density_1km`: 7.0 - 50.7 km/km² (range: 43.7)
- `highway_count_1km`: 0 - 43 (range: 43)

**Problema**: Aunque usamos `StandardScaler`, features con valores extremos (Cerro Navia: density=53.7) pueden dominar.

---

### Hipótesis 4: Overfitting a Estaciones de Training

**Problema**: Con solo **8 estaciones**, agregar 5 features más (40% aumento) causa overfitting espacial.

**Evidencia**:
- Baseline: 13 features para 8 estaciones (1.6 feat/station)
- Enhanced: 18 features para 8 estaciones (2.25 feat/station)

Lasso necesita **regularización más fuerte** con más features.

---

## 🧪 Verificación de Hipótesis

### 1. Verificar Multicolinealidad

Necesitamos calcular correlaciones entre features OSM y existentes.

**Script sugerido**:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/sinca_features_spatial_enhanced.csv')

# Features numéricas
osm_features = ['dist_to_highway_km', 'dist_to_primary_km',
                'road_density_500m', 'road_density_1km', 'highway_count_1km']

existing_features = ['distance_to_center_km', 'lat', 'lon',
                     'elevation', 'wind_direction_rad']

all_features = osm_features + existing_features

# Correlación
corr_matrix = df[all_features].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix: OSM + Existing Features')
plt.tight_layout()
plt.savefig('reports/figures/osm_correlation_matrix.png', dpi=150)
```

---

### 2. Verificar Varianza por Estación

¿Cuánta variabilidad tienen features OSM DENTRO de cada estación?

```python
# Por estación
for station in df['estacion'].unique():
    station_data = df[df['estacion'] == station]

    # Varianza de features OSM
    osm_variance = station_data[osm_features].var()

    print(f"{station}:")
    print(osm_variance)
```

**Hipótesis**: Varianza = 0 (features estáticas) → No aportan información temporal.

---

### 3. VIF (Variance Inflation Factor)

Calcular VIF para detectar multicolinealidad:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[all_features].dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = all_features
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(all_features))]

print(vif_data.sort_values('VIF', ascending=False))
```

**Interpretación**:
- VIF > 10: Multicolinealidad alta
- VIF > 100: Multicolinealidad extrema

---

## 💡 Soluciones Propuestas

### Solución 1: Feature Selection Agresiva

**Eliminar features redundantes**:

Mantener solo:
- `dist_to_highway_km` (eliminar `dist_to_primary_km`)
- `road_density_1km` (eliminar `road_density_500m`)
- Eliminar `highway_count_1km` (redundante con distancia)

**Resultado**: 3 features OSM en lugar de 5

---

### Solución 2: Regularización Más Fuerte

**Aumentar alpha en Lasso**:

```python
# Probar diferentes alphas
alphas = [0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=5000)
    # Evaluar...
```

**Hipótesis**: alpha=1.0 es insuficiente con 18 features.

---

### Solución 3: PCA o Feature Extraction

**Reducir dimensionalidad**:

```python
from sklearn.decomposition import PCA

# PCA en features OSM
pca = PCA(n_components=2)  # Reducir 5 → 2
osm_pca = pca.fit_transform(df[osm_features])

# Usar componentes principales
df['osm_pc1'] = osm_pca[:, 0]
df['osm_pc2'] = osm_pca[:, 1]
```

**Ventaja**: Elimina multicolinealidad, mantiene varianza.

---

### Solución 4: Interacciones en Lugar de Raw Features

**En lugar de agregar features crudas, crear interacciones**:

```python
# Interacción: densidad vial × distancia al centro
df['road_density_center_interaction'] = (
    df['road_density_1km'] * (1 / (1 + df['distance_to_center_km']))
)

# Solo agregar ESTA feature (1 en lugar de 5)
```

**Ventaja**: Captura relación no-lineal, menos features.

---

### Solución 5: Usar Features OSM Solo en Modelos No-Lineales

**Hipótesis**: Tree-based models manejan mejor redundancia que Lasso.

**Estrategia**:
- Lasso: Solo features originales (13)
- XGBoost/Random Forest: Features originales + OSM (18)

**Resultado esperado**: XGBoost mejora (aunque modestamente).

---

## 🎯 Plan de Acción

### Paso 1: Diagnóstico (Inmediato)

1. **Calcular matriz de correlación** OSM + existentes
2. **Calcular VIF** para detectar multicolinealidad
3. **Inspeccionar varianza** por estación (verificar si son estáticas)

### Paso 2: Corrección (Basado en diagnóstico)

**Si multicolinealidad alta (VIF > 10)**:
→ Aplicar Solución 1 o 3 (feature selection o PCA)

**Si features estáticas (var ≈ 0 por estación)**:
→ Las features OSM NO ayudan en LOSO-CV
→ Solo útiles si agregamos variabilidad temporal (ej: tráfico horario)

**Si escala es problema**:
→ Aplicar transformación log o rank-based

### Paso 3: Re-evaluación

Probar combinaciones:
1. Baseline (13 features)
2. Reduced OSM (13 + 2 seleccionadas)
3. OSM + PCA (13 + 2 componentes)
4. OSM + interacciones (13 + 1 interacción)

**Meta**: Mejorar R² de -0.69 → -0.40 a 0.00 (más realista que +0.30)

---

## 📊 Lecciones Aprendidas

### 1. Más Features ≠ Mejor Modelo

Con **solo 8 estaciones**, agregar features puede causar:
- Overfitting espacial
- Multicolinealidad
- Inestabilidad numérica

**Regla práctica**: n_features < n_samples / 10
- 8 estaciones → máximo ~1 feature
- 16,344 observaciones pero agrupadas en 8 estaciones → efectivamente n=8 para generalización espacial

### 2. Features Estáticas Inútiles para LOSO-CV

Features que NO varían temporalmente dentro de estación:
- No ayudan a predecir variabilidad temporal
- Solo útiles para diferencias ENTRE estaciones
- En LOSO-CV, modelo nunca ve la estación de test → features estáticas no transfieren

**Solución**: Necesitamos features que varíen TEMPORALMENTE:
- Tráfico por hora del día
- NDVI mensual (estacional)
- Meteorología (varía diariamente)

### 3. Validar SIEMPRE Antes de Asumir Mejora

Asumimos "tráfico = importante" pero:
- Proximidad a autopista NO captura tráfico real
- Contexto local (Las Condes: residencial vs Cerro Navia: industrial) domina
- Features proxy pueden no correlacionar con target

---

## ✅ Conclusión

**Las features OSM empeoraron el modelo por**:
1. **Multicolinealidad** (5 features correlacionadas)
2. **Features estáticas** (no varían temporalmente)
3. **Overfitting** (18 features para 8 estaciones)

**Próximos pasos**:
1. Calcular correlaciones/VIF (diagnóstico)
2. Reducir a 2-3 features OSM seleccionadas
3. Intentar features TEMPORALES (NDVI, población con variación horaria)
4. Considerar modelos geoestadísticos (Kriging) que NO dependen de features

---

**Archivo**: `docs/OSM_FEATURES_ANALYSIS.md`
**Status**: Análisis completado, soluciones propuestas
**Acción inmediata**: Ejecutar diagnóstico de correlación/VIF
