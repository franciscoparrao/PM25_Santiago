#!/usr/bin/env python3
"""
Agrega features de OpenStreetMap al dataset espacial.

Features agregadas:
1. distance_to_highway - Distancia a autopistas (km)
2. distance_to_primary_road - Distancia a vías primarias (km)
3. road_density_500m - Densidad de vías en 500m (km/km²)
4. road_density_1km - Densidad de vías en 1km (km/km²)
5. highway_count_1km - Número de autopistas en 1km

Impacto esperado: +0.20 - 0.35 R²
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import logging

# OSMnx para descargar datos de OpenStreetMap
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    logging.warning("OSMnx no disponible. Instalar: pip install osmnx")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_osm_features(stations_df, bbox=None):
    """
    Calcula features de proximidad y densidad vial desde OpenStreetMap.

    Args:
        stations_df: DataFrame con estaciones (lat, lon, estacion)
        bbox: Bounding box (south, west, north, east). Si None, usa Santiago completo

    Returns:
        DataFrame con nuevas features por estación
    """
    if not HAS_OSMNX:
        raise ImportError("OSMnx requerido. Instalar: pip install osmnx")

    logger.info("\n" + "="*70)
    logger.info("CALCULANDO FEATURES DE OPENSTREETMAP")
    logger.info("="*70)

    # Bounding box de Santiago (si no se proporciona)
    if bbox is None:
        bbox = (-33.7, -70.9, -33.3, -70.4)  # (south, west, north, east)

    logger.info(f"\nBounding box: {bbox}")

    # Convertir estaciones a GeoDataFrame
    logger.info(f"\nEstaciones a procesar: {len(stations_df)}")

    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=[Point(lon, lat) for lon, lat in
                  zip(stations_df['lon'], stations_df['lat'])],
        crs='EPSG:4326'
    )

    # Proyectar a UTM Zone 19S (Chile) para cálculos en metros
    logger.info("Proyectando a UTM 19S (metros)...")
    stations_utm = stations_gdf.to_crs('EPSG:32719')

    # Descargar red vial de OpenStreetMap
    logger.info("\nDescargando red vial de OpenStreetMap...")
    logger.info("(Esto puede tardar 1-2 minutos la primera vez)")

    try:
        G = ox.graph_from_bbox(
            bbox[0], bbox[2], bbox[1], bbox[3],
            network_type='drive',
            simplify=True
        )
        logger.info(f"✓ Red vial descargada: {len(G.nodes)} nodos, {len(G.edges)} aristas")

    except Exception as e:
        logger.error(f"Error descargando OSM: {e}")
        logger.info("Intentando con método alternativo...")

        # Alternativa: por nombre de lugar
        G = ox.graph_from_place(
            'Santiago, Región Metropolitana, Chile',
            network_type='drive',
            simplify=True
        )
        logger.info(f"✓ Red vial descargada: {len(G.nodes)} nodos, {len(G.edges)} aristas")

    # Convertir grafo a GeoDataFrame de aristas
    logger.info("\nProcesando geometrías de vías...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Proyectar vías a UTM
    edges_utm = edges_gdf.to_crs('EPSG:32719')

    logger.info(f"Total vías: {len(edges_utm)}")

    # Clasificar vías por tipo
    logger.info("\nClasificando vías por tipo...")

    # Autopistas (motorway, trunk)
    highways = edges_utm[edges_utm['highway'].isin([
        'motorway', 'motorway_link', 'trunk', 'trunk_link'
    ])].copy()

    logger.info(f"  • Autopistas: {len(highways)}")

    # Vías primarias
    primary_roads = edges_utm[edges_utm['highway'].isin([
        'primary', 'primary_link', 'secondary', 'secondary_link'
    ])].copy()

    logger.info(f"  • Vías primarias: {len(primary_roads)}")

    # Todas las vías
    all_roads = edges_utm.copy()

    # Calcular features por estación
    logger.info("\n" + "="*70)
    logger.info("CALCULANDO FEATURES POR ESTACIÓN")
    logger.info("="*70)

    results = []

    for idx, station in stations_utm.iterrows():
        station_name = station['estacion']
        point = station.geometry

        logger.info(f"\n[{idx+1}/{len(stations_utm)}] {station_name}")

        features = {'estacion': station_name}

        # 1. Distancia a autopista más cercana
        if len(highways) > 0:
            dist_highway = highways.distance(point).min()
            features['dist_to_highway_m'] = dist_highway
            features['dist_to_highway_km'] = dist_highway / 1000
            logger.info(f"  • Distancia a autopista: {dist_highway:.0f} m")
        else:
            features['dist_to_highway_m'] = np.nan
            features['dist_to_highway_km'] = np.nan

        # 2. Distancia a vía primaria más cercana
        if len(primary_roads) > 0:
            dist_primary = primary_roads.distance(point).min()
            features['dist_to_primary_m'] = dist_primary
            features['dist_to_primary_km'] = dist_primary / 1000
            logger.info(f"  • Distancia a vía primaria: {dist_primary:.0f} m")
        else:
            features['dist_to_primary_m'] = np.nan
            features['dist_to_primary_km'] = np.nan

        # 3. Densidad de vías en 500m
        buffer_500m = point.buffer(500)
        roads_in_500m = all_roads[all_roads.intersects(buffer_500m)]

        if len(roads_in_500m) > 0:
            total_length_m = roads_in_500m.geometry.length.sum()
            area_km2 = np.pi * (0.5 ** 2)  # π * r² con r=0.5km
            density_500m = (total_length_m / 1000) / area_km2  # km/km²
            features['road_density_500m'] = density_500m
            logger.info(f"  • Densidad vial 500m: {density_500m:.1f} km/km²")
        else:
            features['road_density_500m'] = 0

        # 4. Densidad de vías en 1km
        buffer_1km = point.buffer(1000)
        roads_in_1km = all_roads[all_roads.intersects(buffer_1km)]

        if len(roads_in_1km) > 0:
            total_length_m = roads_in_1km.geometry.length.sum()
            area_km2 = np.pi * (1.0 ** 2)
            density_1km = (total_length_m / 1000) / area_km2
            features['road_density_1km'] = density_1km
            logger.info(f"  • Densidad vial 1km: {density_1km:.1f} km/km²")
        else:
            features['road_density_1km'] = 0

        # 5. Número de autopistas en 1km
        highways_in_1km = highways[highways.intersects(buffer_1km)]
        features['highway_count_1km'] = len(highways_in_1km)
        logger.info(f"  • Autopistas en 1km: {len(highways_in_1km)}")

        results.append(features)

    # Convertir a DataFrame
    osm_features_df = pd.DataFrame(results)

    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE FEATURES CALCULADAS")
    logger.info("="*70)

    logger.info(f"\nEstaciones procesadas: {len(osm_features_df)}")
    logger.info(f"Features agregadas: {len(osm_features_df.columns) - 1}")

    logger.info("\nEstadísticas:")
    print("\n", osm_features_df.describe().round(2))

    return osm_features_df


def main():
    logger.info("\n" + "="*70)
    logger.info("AGREGANDO FEATURES DE OPENSTREETMAP")
    logger.info("="*70)

    # Cargar dataset espacial
    input_file = Path('data/processed/sinca_features_spatial.csv')
    logger.info(f"\nCargando dataset: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Columnas: {len(df.columns)}")

    # Extraer estaciones únicas
    stations = df[['estacion', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)

    logger.info(f"\nEstaciones únicas: {len(stations)}")
    for idx, row in stations.iterrows():
        logger.info(f"  {idx+1}. {row['estacion']:25s} ({row['lat']:.4f}, {row['lon']:.4f})")

    # Calcular features OSM
    osm_features = calculate_osm_features(stations)

    # Merge con dataset original
    logger.info("\n" + "="*70)
    logger.info("INTEGRANDO FEATURES AL DATASET")
    logger.info("="*70)

    # Seleccionar solo columnas útiles
    osm_cols = ['estacion', 'dist_to_highway_km', 'dist_to_primary_km',
                'road_density_500m', 'road_density_1km', 'highway_count_1km']

    df_enhanced = df.merge(
        osm_features[osm_cols],
        on='estacion',
        how='left'
    )

    logger.info(f"\nColumnas antes: {len(df.columns)}")
    logger.info(f"Columnas después: {len(df_enhanced.columns)}")
    logger.info(f"Features agregadas: {len(osm_cols) - 1}")

    # Verificar NaNs
    new_features = [col for col in osm_cols if col != 'estacion']
    nan_counts = df_enhanced[new_features].isnull().sum()

    logger.info("\nValores faltantes:")
    if nan_counts.sum() > 0:
        for col, count in nan_counts[nan_counts > 0].items():
            logger.info(f"  • {col}: {count}")
    else:
        logger.info("  ✓ Sin valores faltantes")

    # Guardar dataset mejorado
    output_file = Path('data/processed/sinca_features_spatial_enhanced.csv')
    df_enhanced.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"\n✓ Dataset guardado: {output_file}")
    logger.info(f"  Tamaño: {size_mb:.2f} MB")
    logger.info(f"  Registros: {len(df_enhanced):,}")
    logger.info(f"  Features totales: {len(df_enhanced.columns)}")

    # Guardar también solo las features OSM
    osm_output = Path('data/processed/osm_features.csv')
    osm_features.to_csv(osm_output, index=False)
    logger.info(f"\n✓ Features OSM guardadas: {osm_output}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ FEATURES OSM AGREGADAS EXITOSAMENTE ✓✓✓")
    logger.info("="*70)

    logger.info("\nNuevas features disponibles:")
    for feat in new_features:
        logger.info(f"  • {feat}")

    return df_enhanced, osm_features


if __name__ == "__main__":
    df_enhanced, osm_features = main()
