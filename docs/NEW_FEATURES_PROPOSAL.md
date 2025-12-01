# Propuesta de Nuevas Features para Mejorar Predicción Espacial

**Objetivo**: Mejorar R² de -1.09 → +0.30 a +0.60
**Estrategia**: Agregar features de escala local (< 1 km) que capturen heterogeneidad espacial

---

## 📊 Análisis de Gap Actual

### Features Actuales (13)

| Categoría | Features | Resolución Espacial | Limitación |
|-----------|----------|---------------------|------------|
| Geográficas | lat, lon, elevation, distance_to_center | Punto | Solo ubicación, no contexto local |
| Meteorológicas | wind_u, precipitation, etc. | 25 km (ERA5) | No captura micro-clima |
| Satelitales | s5p_no2 | 7 km | No captura variabilidad intra-urbana |
| Temporales | day_of_year, day_of_week | N/A | OK |

### Problema Principal

**PM2.5 varía ~50-100 μg/m³ en distancias < 1 km** debido a:
- Tráfico vehicular (autopistas vs calles residenciales)
- Uso de suelo (industrial vs parques)
- Topografía micro-escala (valles urbanos, cañones de edificios)

**Features actuales**: Resolución > 7 km → NO capturan esta variabilidad

---

## 🎯 Nuevas Features Propuestas

### Categoría 1: Uso de Suelo (Alta Prioridad ⭐⭐⭐)

#### 1.1. Índices de Vegetación

| Feature | Fuente | Resolución | Cálculo | Impacto Esperado |
|---------|--------|------------|---------|------------------|
| **NDVI** (Normalized Difference Vegetation Index) | Sentinel-2 | **10m** | (NIR - Red) / (NIR + Red) | +0.10 - 0.15 R² |
| **NDVI_500m** (promedio 500m radius) | Sentinel-2 | 10m agregado | Mean NDVI en buffer 500m | +0.08 - 0.12 R² |
| **Green_Space_Fraction** | Sentinel-2 | 10m | % píxeles con NDVI > 0.4 en 1km² | +0.05 - 0.10 R² |

**Interpretación Física**:
- NDVI alto (0.6-0.8) = parques, bosques → PM2.5 bajo (filtrado por vegetación)
- NDVI bajo (0.2-0.3) = urbano denso, pavimento → PM2.5 alto

**Implementación (Google Earth Engine)**:
```javascript
// Sentinel-2: NDVI mensual por estación
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(santiago)
  .filterDate('2019-01-01', '2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// Calcular NDVI
var addNDVI = function(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return img.addBands(ndvi);
};

var s2_ndvi = s2.map(addNDVI);

// Reducir mensualmente por estación (buffer 500m)
var monthly_ndvi = s2_ndvi.select('NDVI')
  .map(function(img) {
    return img.reduceRegions({
      collection: stations.map(function(f) {
        return f.buffer(500);  // 500m radius
      }),
      reducer: ee.Reducer.mean(),
      scale: 10
    }).map(function(f) {
      return f.set('date', img.date().format('YYYY-MM'));
    });
  })
  .flatten();

Export.table.toDrive({
  collection: monthly_ndvi,
  description: 'ndvi_500m_monthly',
  fileFormat: 'CSV'
});
```

**Datos necesarios**: Ninguno (Sentinel-2 gratis en GEE)

---

#### 1.2. Impermeabilización (Built-up Index)

| Feature | Fuente | Resolución | Cálculo | Impacto Esperado |
|---------|--------|------------|---------|------------------|
| **NDBI** (Normalized Difference Built-up Index) | Sentinel-2 | 10m | (SWIR - NIR) / (SWIR + NIR) | +0.08 - 0.12 R² |
| **Impervious_Surface_500m** | Sentinel-2 | 10m | % área impermeabilizada en 500m | +0.10 - 0.15 R² |

**Interpretación**:
- NDBI alto = edificios, pavimento → PM2.5 alto (menos dispersión, más tráfico)

**Implementación**:
```javascript
var addNDBI = function(img) {
  var ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI');
  return img.addBands(ndbi);
};

var s2_ndbi = s2.map(addNDBI);
```

---

#### 1.3. Clasificación de Uso de Suelo

| Feature | Fuente | Resolución | Descripción | Impacto Esperado |
|---------|--------|------------|-------------|------------------|
| **Land_Cover_Class** | ESA WorldCover | **10m** | Categórico: urbano, agrícola, vegetación | +0.05 - 0.10 R² |
| **Urban_Fraction_1km** | WorldCover | 10m | % urbano en 1km² | +0.08 - 0.12 R² |
| **Industrial_Fraction_1km** | WorldCover | 10m | % industrial en 1km² | +0.10 - 0.15 R² |

**Dataset**: ESA WorldCover 2021 (gratis, global, 10m)

**Implementación**:
```javascript
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first();

// Clases:
// 50 = Built-up
// 40 = Cropland
// 10 = Tree cover
// 95 = Bare / sparse vegetation

var urban_mask = worldcover.eq(50);

// Fracción urbana en 1km
var urban_fraction = urban_mask.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: ee.Kernel.circle(500, 'meters')  // 1km diameter
});
```

---

### Categoría 2: Tráfico Vehicular (Alta Prioridad ⭐⭐⭐)

#### 2.1. Distancia a Vías Principales

| Feature | Fuente | Resolución | Descripción | Impacto Esperado |
|---------|--------|------------|-------------|------------------|
| **Distance_to_Highway** | OpenStreetMap | Vector | Distancia a autopistas (km) | +0.15 - 0.25 R² |
| **Distance_to_Primary_Road** | OpenStreetMap | Vector | Distancia a vías primarias (km) | +0.10 - 0.15 R² |
| **Road_Density_500m** | OpenStreetMap | Vector | Longitud total vías en 500m (km/km²) | +0.12 - 0.18 R² |

**Interpretación**:
- Cerca de autopista (< 100m) → PM2.5 +20-40 μg/m³
- Calle residencial → PM2.5 bajo

**Implementación (Python - OSMnx)**:
```python
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# Descargar red vial de Santiago
G = ox.graph_from_place('Santiago, Chile', network_type='drive')

# Convertir a GeoDataFrame
edges = ox.graph_to_gdfs(G, nodes=False)

# Filtrar autopistas
highways = edges[edges['highway'].isin(['motorway', 'motorway_link', 'trunk'])]

# Calcular distancia de cada estación
stations_gdf = gpd.GeoDataFrame(
    stations,
    geometry=[Point(lon, lat) for lon, lat in zip(stations['lon'], stations['lat'])],
    crs='EPSG:4326'
)

# Proyectar a UTM (metros)
stations_utm = stations_gdf.to_crs('EPSG:32719')  # UTM Zone 19S (Chile)
highways_utm = highways.to_crs('EPSG:32719')

# Distancia mínima a autopista
stations_utm['dist_to_highway_m'] = stations_utm.geometry.apply(
    lambda pt: highways_utm.distance(pt).min()
)

stations_utm['dist_to_highway_km'] = stations_utm['dist_to_highway_m'] / 1000

print(stations_utm[['estacion', 'dist_to_highway_km']])
```

**Datos necesarios**: Ninguno (OpenStreetMap gratis)

---

#### 2.2. Intensidad de Tráfico (Proxy)

| Feature | Fuente | Resolución | Descripción | Impacto Esperado |
|---------|--------|------------|-------------|------------------|
| **Population_Density_1km** | WorldPop | **100m** | Habitantes/km² en 1km² | +0.08 - 0.12 R² |
| **Nighttime_Lights** | VIIRS | **500m** | Radiance nocturna (proxy de actividad urbana) | +0.05 - 0.10 R² |

**Interpretación**:
- Alta densidad poblacional → más tráfico → PM2.5 alto
- Nighttime lights captura actividad económica (comercial, industrial)

**Implementación (GEE - WorldPop)**:
```javascript
var worldpop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filterBounds(santiago)
  .filter(ee.Filter.eq('year', 2020))
  .first();

// Densidad en 1km buffer
var pop_density_1km = worldpop.reduceNeighborhood({
  reducer: ee.Reducer.sum(),
  kernel: ee.Kernel.circle(500, 'meters')
});
```

**Implementación (GEE - VIIRS Nighttime Lights)**:
```javascript
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2019-01-01', '2025-12-31')
  .select('avg_rad');

// Mensual por estación
var monthly_lights = viirs.map(function(img) {
  return img.reduceRegions({
    collection: stations.map(function(f) { return f.buffer(500); }),
    reducer: ee.Reducer.mean(),
    scale: 500
  }).map(function(f) {
    return f.set('date', img.date().format('YYYY-MM'));
  });
}).flatten();
```

---

### Categoría 3: Topografía Avanzada (Prioridad Media ⭐⭐)

#### 3.1. Índices Topográficos

| Feature | Fuente | Resolución | Descripción | Impacto Esperado |
|---------|--------|------------|-------------|------------------|
| **Slope** | SRTM DEM | **30m** | Pendiente (grados) | +0.03 - 0.07 R² |
| **Aspect** | SRTM DEM | 30m | Orientación (0-360°) | +0.05 - 0.10 R² |
| **Topographic_Position_Index** | SRTM DEM | 30m | Valle (-) vs cresta (+) | +0.08 - 0.12 R² |
| **Sky_View_Factor** | SRTM DEM | 30m | % cielo visible (0-1) | +0.05 - 0.08 R² |

**Interpretación**:
- Valle (TPI < 0) + baja pendiente → inversión térmica → PM2.5 alto
- Aspect sur (en Chile) → más radiación solar → mejor dispersión

**Implementación (GEE - SRTM)**:
```javascript
var srtm = ee.Image('USGS/SRTMGL1_003');

// Slope
var slope = ee.Terrain.slope(srtm);

// Aspect
var aspect = ee.Terrain.aspect(srtm);

// TPI: elevación - promedio 1km radius
var tpi = srtm.subtract(
  srtm.reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.circle(500, 'meters')
  })
);

// Extraer por estación
var topo_features = ee.Image.cat([slope, aspect, tpi])
  .reduceRegions({
    collection: stations,
    reducer: ee.Reducer.first(),
    scale: 30
  });
```

---

### Categoría 4: Fuentes de Emisión (Prioridad Alta ⭐⭐⭐)

#### 4.1. Proximidad a Fuentes Industriales

| Feature | Fuente | Descripción | Impacto Esperado |
|---------|--------|-------------|------------------|
| **Distance_to_Industry** | Catastro SMA Chile | Distancia a fuentes industriales (km) | +0.10 - 0.20 R² |
| **Industrial_Count_5km** | Catastro SMA | N° industrias en 5km | +0.08 - 0.15 R² |

**Fuente de datos**:
- [Superintendencia del Medio Ambiente (SMA)](https://sma.gob.cl/)
- Registro de fuentes fijas (industrias reguladas)

**Implementación**:
```python
# Cargar catastro industrial (shapefile o CSV con coordenadas)
industries = gpd.read_file('path/to/sma_industries.shp')

# O desde API SMA (si disponible)
# import requests
# industries = requests.get('https://sma.gob.cl/api/fuentes-fijas').json()

# Calcular distancia
stations_utm['dist_to_industry_km'] = stations_utm.geometry.apply(
    lambda pt: industries.distance(pt).min() / 1000
)

# Contar industrias en 5km
stations_utm['industry_count_5km'] = stations_utm.geometry.apply(
    lambda pt: industries[industries.distance(pt) < 5000].shape[0]
)
```

---

#### 4.2. Emisiones Vehiculares Estimadas

| Feature | Fuente | Descripción | Impacto Esperado |
|---------|--------|-------------|------------------|
| **Vehicle_Emissions_Index** | Modelo COPERT + tráfico | Emisiones PM2.5 vehicular estimadas | +0.15 - 0.25 R² |

**Modelo simplificado**:
```python
# Proxy: distancia inversa a autopistas ponderada por tráfico
vehicle_emissions_proxy = (
    1 / (1 + df['dist_to_highway_km']) *  # Cercanía
    df['population_density_1km'] / 1000  # Intensidad tráfico
)
```

---

### Categoría 5: Meteorología de Alta Resolución (Prioridad Baja ⭐)

#### 5.1. WRF Downscaling

| Feature | Fuente | Resolución | Descripción | Impacto Esperado |
|---------|--------|------------|-------------|------------------|
| **WRF_Wind_1km** | WRF-Chem | **1 km** | Viento downscaled | +0.03 - 0.07 R² |
| **Planetary_Boundary_Layer_Height** | WRF | 1 km | Altura capa límite (m) | +0.05 - 0.10 R² |

**Limitación**: Requiere correr WRF (computacionalmente costoso)

**Alternativa**: Usar ERA5-Land (9 km, mejor que ERA5 25 km)

---

## 📊 Priorización de Features

### Tabla Resumen

| Categoría | Feature | Impacto R² | Dificultad | Prioridad | Datos Requeridos |
|-----------|---------|------------|------------|-----------|------------------|
| **Tráfico** | Distance_to_Highway | +0.15-0.25 | Fácil | ⭐⭐⭐ | OpenStreetMap (gratis) |
| **Tráfico** | Road_Density_500m | +0.12-0.18 | Fácil | ⭐⭐⭐ | OpenStreetMap |
| **Uso Suelo** | NDVI_500m | +0.08-0.12 | Media | ⭐⭐⭐ | Sentinel-2 GEE |
| **Uso Suelo** | Impervious_Surface | +0.10-0.15 | Media | ⭐⭐⭐ | Sentinel-2 GEE |
| **Industria** | Distance_to_Industry | +0.10-0.20 | Difícil | ⭐⭐⭐ | Catastro SMA |
| **Población** | Population_Density | +0.08-0.12 | Fácil | ⭐⭐ | WorldPop GEE |
| **Topografía** | TPI | +0.08-0.12 | Media | ⭐⭐ | SRTM GEE |
| **Topografía** | Aspect | +0.05-0.10 | Media | ⭐⭐ | SRTM GEE |
| **Actividad** | Nighttime_Lights | +0.05-0.10 | Fácil | ⭐⭐ | VIIRS GEE |

---

## 🎯 Plan de Implementación

### Fase 1: Quick Wins (1-2 semanas)

**Objetivo**: +0.20 - 0.30 R² con features fáciles

1. **OpenStreetMap** (1 día):
   - `distance_to_highway`
   - `distance_to_primary_road`
   - `road_density_500m`

2. **Sentinel-2 GEE** (3 días):
   - `ndvi_500m` (mensual)
   - `ndbi_500m` (mensual)

3. **WorldPop GEE** (1 día):
   - `population_density_1km`

4. **VIIRS GEE** (1 día):
   - `nighttime_lights` (mensual)

**Impacto esperado**: R² de -1.09 → -0.20 a +0.10

---

### Fase 2: Features Avanzadas (2-4 semanas)

**Objetivo**: +0.30 - 0.50 R² con features complejas

1. **Topografía SRTM** (2 días):
   - `slope`, `aspect`, `tpi`

2. **ESA WorldCover** (2 días):
   - `urban_fraction_1km`
   - `land_cover_class`

3. **Catastro Industrial SMA** (1 semana):
   - Gestionar datos con SMA
   - Procesar shapefiles
   - `distance_to_industry`
   - `industry_count_5km`

**Impacto esperado**: R² de +0.10 → +0.30 a +0.50

---

### Fase 3: Optimización (1-2 semanas)

1. **Feature Engineering Avanzado**:
   - Interacciones: `ndvi × distance_to_highway`
   - Polinomios: `distance_to_highway²`

2. **Feature Selection**:
   - Eliminar features redundantes
   - Validar con LOSO-CV

3. **Hyperparameter Tuning**:
   - Grid Search extensivo

**Impacto esperado**: R² de +0.40 → +0.50 a +0.60

---

## 💻 Script de Ejemplo - Fase 1

### `add_osm_features.py`

```python
#!/usr/bin/env python3
"""
Agrega features de OpenStreetMap al dataset espacial.
"""

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from pathlib import Path

def calculate_road_features(stations_df):
    """
    Calcula features de proximidad a vías desde OSM.

    Args:
        stations_df: DataFrame con estaciones (lat, lon)

    Returns:
        DataFrame con nuevas features
    """
    # Convertir a GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=[Point(lon, lat) for lon, lat in
                  zip(stations_df['lon'], stations_df['lat'])],
        crs='EPSG:4326'
    )

    # Proyectar a UTM (metros)
    stations_utm = stations_gdf.to_crs('EPSG:32719')  # Chile

    # Descargar red vial de Santiago
    bbox = (-33.7, -70.9, -33.3, -70.4)  # Santiago aprox
    G = ox.graph_from_bbox(
        bbox[0], bbox[2], bbox[1], bbox[3],
        network_type='drive'
    )

    edges = ox.graph_to_gdfs(G, nodes=False)
    edges_utm = edges.to_crs('EPSG:32719')

    # Filtrar autopistas
    highways = edges_utm[edges_utm['highway'].isin([
        'motorway', 'motorway_link', 'trunk', 'trunk_link'
    ])]

    # Vías primarias
    primary_roads = edges_utm[edges_utm['highway'].isin([
        'primary', 'primary_link'
    ])]

    # Calcular distancias
    print("Calculando distancia a autopistas...")
    stations_utm['dist_to_highway_m'] = stations_utm.geometry.apply(
        lambda pt: highways.distance(pt).min()
    )

    print("Calculando distancia a vías primarias...")
    stations_utm['dist_to_primary_m'] = stations_utm.geometry.apply(
        lambda pt: primary_roads.distance(pt).min()
    )

    # Densidad de vías en 500m
    print("Calculando densidad vial...")
    def road_density_500m(point):
        buffer = point.buffer(500)  # 500m radius
        roads_in_buffer = edges_utm[edges_utm.intersects(buffer)]
        total_length_m = roads_in_buffer.geometry.length.sum()
        area_km2 = (3.14159 * 0.5**2)  # π * 0.5²
        return total_length_m / 1000 / area_km2  # km/km²

    stations_utm['road_density_500m'] = stations_utm.geometry.apply(
        road_density_500m
    )

    # Convertir a km
    stations_utm['dist_to_highway_km'] = stations_utm['dist_to_highway_m'] / 1000
    stations_utm['dist_to_primary_km'] = stations_utm['dist_to_primary_m'] / 1000

    # Volver a DataFrame
    result = pd.DataFrame(stations_utm.drop(columns='geometry'))

    return result[['estacion', 'dist_to_highway_km',
                   'dist_to_primary_km', 'road_density_500m']]


def main():
    # Cargar dataset
    df = pd.read_csv('data/processed/sinca_features_spatial.csv')

    # Estaciones únicas
    stations = df[['estacion', 'lat', 'lon']].drop_duplicates()

    # Calcular features OSM
    osm_features = calculate_road_features(stations)

    # Merge con dataset
    df_enhanced = df.merge(osm_features, on='estacion', how='left')

    # Guardar
    output_file = 'data/processed/sinca_features_spatial_osm.csv'
    df_enhanced.to_csv(output_file, index=False)

    print(f"\n✓ Features agregadas: {len(osm_features.columns) - 1}")
    print(f"✓ Dataset guardado: {output_file}")
    print(f"\nNuevas features:")
    print(osm_features.describe())


if __name__ == "__main__":
    main()
```

---

## 📈 Impacto Esperado Total

### Escenario Conservador

| Fase | Features Agregadas | R² Esperado | Mejora |
|------|-------------------|-------------|--------|
| Baseline | 13 actuales | -1.09 | - |
| Fase 1 | +5 (OSM, NDVI, Pop) | -0.20 a +0.10 | +0.89 a +1.19 |
| Fase 2 | +7 (Topo, Industria) | +0.20 a +0.40 | +1.29 a +1.49 |
| Fase 3 | Optimización | +0.30 a +0.50 | +1.39 a +1.59 |

**Meta Realista**: R² = **+0.30 a +0.50** (predicción útil)

### Escenario Optimista

Con todas las features + modelos geoestadísticos (Kriging):
- **R² = +0.50 a +0.70**
- **RMSE = 12-15 μg/m³** (vs 27 actual)

---

## ✅ Recomendación Final

### Prioridad Inmediata (hacer AHORA)

1. **OpenStreetMap features** (1 día de trabajo):
   - `distance_to_highway`
   - `road_density_500m`

   **Impacto**: +0.15-0.25 R²

2. **Sentinel-2 NDVI** (2 días):
   - `ndvi_500m` mensual

   **Impacto**: +0.08-0.12 R²

**Total Fase 1**: +0.23-0.37 R² → R² final = -0.86 a -0.72

### Prioridad Media (próximas 2 semanas)

3. **WorldPop + VIIRS** (2 días)
4. **Topografía SRTM** (2 días)
5. **Catastro Industrial** (1 semana)

**Total Fase 2**: +0.30-0.50 R² acumulado

### ¿Qué hacer con Hyperparameter Tuning?

- **Impacto esperado de tuning**: +0.05-0.15 R² (marginal)
- **Impacto de nuevas features**: +0.30-0.50 R² (mayor)

**Conclusión**: **Priorizar features sobre tuning**. Hacer tuning DESPUÉS de agregar features.

---

**Próximo Paso Sugerido**: Ejecutar `add_osm_features.py` para agregar las 3 features de tráfico (fácil, alto impacto).
