#!/usr/bin/env python3
"""
Script para integrar datos satelitales (ERA5, MODIS, Sentinel-5P) con datos SINCA.

Estrategia:
1. Consolidar archivos satelitales mensuales
2. Matching espacial: encontrar píxel satelital más cercano a cada estación SINCA
3. Agregación temporal: datos satelitales mensuales → promedio por estación
4. Join temporal: unir SINCA diario con datos satelitales mensuales
5. Dataset final integrado
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging
from datetime import datetime
from scipy.spatial import cKDTree

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula distancia en km entre dos puntos usando fórmula de Haversine.

    Args:
        lat1, lon1: Coordenadas del punto 1
        lat2, lon2: Coordenadas del punto 2

    Returns:
        Distancia en kilómetros
    """
    R = 6371  # Radio de la Tierra en km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def consolidate_satellite_files(pattern, source_name):
    """
    Consolida múltiples archivos CSV mensuales en un solo DataFrame.

    Args:
        pattern: Glob pattern para encontrar archivos
        source_name: Nombre de la fuente (para logging)

    Returns:
        DataFrame consolidado
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CONSOLIDANDO {source_name}")
    logger.info(f"{'='*60}")

    files = sorted(glob.glob(pattern))
    logger.info(f"Archivos encontrados: {len(files)}")

    if not files:
        logger.warning(f"No se encontraron archivos para {source_name}")
        return None

    logger.info(f"Rango: {Path(files[0]).name} → {Path(files[-1]).name}")

    dfs = []
    for i, file in enumerate(files, 1):
        if i % 20 == 0:
            logger.info(f"  Procesando {i}/{len(files)}...")

        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Error en {file}: {e}")

    df_consolidated = pd.concat(dfs, ignore_index=True)
    logger.info(f"Registros totales: {len(df_consolidated):,}")

    return df_consolidated


def find_nearest_pixels(stations_df, satellite_df, max_distance_km=10):
    """
    Encuentra el píxel satelital más cercano a cada estación SINCA.

    Args:
        stations_df: DataFrame con estaciones SINCA (debe tener lat, lon)
        satellite_df: DataFrame con datos satelitales (debe tener lat, lon)
        max_distance_km: Distancia máxima permitida en km

    Returns:
        DataFrame con mapping estación → píxel más cercano
    """
    logger.info("\nBuscando píxeles satelitales más cercanos...")

    # Obtener coordenadas únicas de satélite
    sat_coords = satellite_df[['lat', 'lon']].drop_duplicates()
    logger.info(f"  Píxeles satelitales únicos: {len(sat_coords):,}")

    # Crear KD-Tree para búsqueda eficiente
    # Convertir a radianes para cálculo en esfera
    sat_coords_rad = np.radians(sat_coords[['lat', 'lon']].values)
    tree = cKDTree(sat_coords_rad)

    results = []

    for _, station in stations_df.iterrows():
        station_name = station['estacion']
        station_lat = station['lat']
        station_lon = station['lon']

        # Buscar píxel más cercano
        station_rad = np.radians([station_lat, station_lon])
        distance_rad, idx = tree.query(station_rad)

        # Convertir distancia angular a km (aproximación)
        distance_km = distance_rad * 6371

        nearest_lat = sat_coords.iloc[idx]['lat']
        nearest_lon = sat_coords.iloc[idx]['lon']

        # Calcular distancia exacta con Haversine
        exact_distance = haversine_distance(
            station_lat, station_lon,
            nearest_lat, nearest_lon
        )

        if exact_distance <= max_distance_km:
            results.append({
                'estacion': station_name,
                'station_lat': station_lat,
                'station_lon': station_lon,
                'satellite_lat': nearest_lat,
                'satellite_lon': nearest_lon,
                'distance_km': exact_distance
            })

            logger.info(f"  ✓ {station_name}: {exact_distance:.2f} km")
        else:
            logger.warning(f"  ✗ {station_name}: {exact_distance:.2f} km (> {max_distance_km} km)")

    return pd.DataFrame(results)


def aggregate_satellite_to_station(satellite_df, mapping_df, date_col='date'):
    """
    Agrega datos satelitales por estación usando el mapping espacial.

    Args:
        satellite_df: DataFrame con datos satelitales
        mapping_df: DataFrame con mapping estación → coordenadas satélite
        date_col: Nombre de la columna de fecha

    Returns:
        DataFrame agregado por estación y fecha
    """
    logger.info("\nAgregando datos satelitales por estación...")

    # Merge con mapping para asociar cada píxel con estaciones
    merged = satellite_df.merge(
        mapping_df[['estacion', 'satellite_lat', 'satellite_lon']],
        left_on=['lat', 'lon'],
        right_on=['satellite_lat', 'satellite_lon'],
        how='inner'
    )

    logger.info(f"  Registros después de matching: {len(merged):,}")

    # Parsear fecha si es necesario
    if merged[date_col].dtype == 'object':
        merged[date_col] = pd.to_datetime(merged[date_col])

    # Extraer año-mes para agregación mensual
    merged['year_month'] = merged[date_col].dt.to_period('M')

    # Identificar columnas de valor
    value_cols = [col for col in merged.columns if col not in [
        'lat', 'lon', 'satellite_lat', 'satellite_lon',
        'estacion', date_col, 'year_month', 'variable', 'date'
    ]]

    # Agregar por estación y mes (promedio)
    agg_dict = {col: 'mean' for col in value_cols}

    if 'value' in merged.columns:
        # Formato largo (MODIS, Sentinel)
        aggregated = merged.groupby(['estacion', 'year_month']).agg({'value': 'mean'}).reset_index()
    else:
        # Formato ancho (ERA5)
        aggregated = merged.groupby(['estacion', 'year_month']).agg(agg_dict).reset_index()

    # Convertir year_month de vuelta a datetime
    aggregated['date'] = aggregated['year_month'].dt.to_timestamp()
    aggregated = aggregated.drop('year_month', axis=1)

    logger.info(f"  Registros agregados: {len(aggregated):,}")
    logger.info(f"  Estaciones únicas: {aggregated['estacion'].nunique()}")

    return aggregated


def main():
    logger.info("\n" + "="*70)
    logger.info("INTEGRACIÓN DE DATOS SATELITALES CON SINCA")
    logger.info("="*70)

    # Directorios
    sinca_file = Path('data/processed/sinca_pm25_master.csv')
    output_dir = Path('data/processed')

    # 1. Cargar datos SINCA
    logger.info("\n" + "="*60)
    logger.info("CARGANDO DATOS SINCA")
    logger.info("="*60)

    sinca = pd.read_csv(sinca_file, parse_dates=['datetime', 'date'])
    logger.info(f"Registros SINCA: {len(sinca):,}")
    logger.info(f"Rango temporal: {sinca['datetime'].min()} → {sinca['datetime'].max()}")

    # Filtrar por periodo de overlap (2019-2025)
    sinca_overlap = sinca[sinca['year'] >= 2019].copy()
    logger.info(f"Registros en periodo overlap (2019+): {len(sinca_overlap):,}")

    # Obtener estaciones únicas
    stations = sinca[['estacion', 'lat', 'lon']].drop_duplicates()
    logger.info(f"Estaciones únicas: {len(stations)}")

    # 2. Consolidar ERA5
    era5 = consolidate_satellite_files('data/raw/era5/*.csv', 'ERA5')
    if era5 is not None:
        era5['date'] = pd.to_datetime(era5['date'])

        # Matching espacial
        era5_mapping = find_nearest_pixels(stations, era5)

        # Agregar por estación
        era5_agg = aggregate_satellite_to_station(era5, era5_mapping)

        # Renombrar columnas con prefijo
        value_cols = [col for col in era5_agg.columns if col not in ['estacion', 'date']]
        rename_dict = {col: f'era5_{col}' for col in value_cols}
        era5_agg = era5_agg.rename(columns=rename_dict)

        logger.info(f"\nERA5 agregado: {len(era5_agg):,} registros")
    else:
        era5_agg = None

    # 3. Consolidar MODIS
    modis = consolidate_satellite_files('data/raw/modis/*.csv', 'MODIS')
    if modis is not None:
        modis['date'] = pd.to_datetime(modis['date'])

        # Matching espacial
        modis_mapping = find_nearest_pixels(stations, modis)

        # Agregar por estación
        modis_agg = aggregate_satellite_to_station(modis, modis_mapping)
        modis_agg = modis_agg.rename(columns={'value': 'modis_aod'})

        logger.info(f"\nMODIS agregado: {len(modis_agg):,} registros")
    else:
        modis_agg = None

    # 4. Consolidar Sentinel-5P
    s5p = consolidate_satellite_files('data/raw/sentinel5p/*.csv', 'Sentinel-5P')
    if s5p is not None:
        s5p['date'] = pd.to_datetime(s5p['date'])

        # Matching espacial
        s5p_mapping = find_nearest_pixels(stations, s5p)

        # Agregar por estación
        s5p_agg = aggregate_satellite_to_station(s5p, s5p_mapping)
        s5p_agg = s5p_agg.rename(columns={'value': 's5p_no2'})

        logger.info(f"\nSentinel-5P agregado: {len(s5p_agg):,} registros")
    else:
        s5p_agg = None

    # 5. Integrar con SINCA
    logger.info("\n" + "="*60)
    logger.info("INTEGRANDO DATASETS")
    logger.info("="*60)

    # Agregar columna año-mes a SINCA para join
    sinca_overlap['year_month'] = pd.to_datetime(sinca_overlap['date']).dt.to_period('M')
    sinca_overlap['satellite_date'] = sinca_overlap['year_month'].dt.to_timestamp()

    # Merge con datos satelitales
    integrated = sinca_overlap.copy()

    if era5_agg is not None:
        logger.info("\nMerging con ERA5...")
        integrated = integrated.merge(
            era5_agg,
            left_on=['estacion', 'satellite_date'],
            right_on=['estacion', 'date'],
            how='left',
            suffixes=('', '_era5')
        )
        integrated = integrated.drop('date_era5', axis=1, errors='ignore')
        logger.info(f"  Registros con ERA5: {integrated['era5_temperature_2m'].notna().sum():,}")

    if modis_agg is not None:
        logger.info("\nMerging con MODIS...")
        integrated = integrated.merge(
            modis_agg,
            left_on=['estacion', 'satellite_date'],
            right_on=['estacion', 'date'],
            how='left',
            suffixes=('', '_modis')
        )
        integrated = integrated.drop('date_modis', axis=1, errors='ignore')
        logger.info(f"  Registros con MODIS: {integrated['modis_aod'].notna().sum():,}")

    if s5p_agg is not None:
        logger.info("\nMerging con Sentinel-5P...")
        integrated = integrated.merge(
            s5p_agg,
            left_on=['estacion', 'satellite_date'],
            right_on=['estacion', 'date'],
            how='left',
            suffixes=('', '_s5p')
        )
        integrated = integrated.drop('date_s5p', axis=1, errors='ignore')
        logger.info(f"  Registros con Sentinel-5P: {integrated['s5p_no2'].notna().sum():,}")

    # Limpiar columnas temporales
    integrated = integrated.drop(['year_month', 'satellite_date'], axis=1, errors='ignore')

    # 6. Validación y estadísticas
    logger.info("\n" + "="*60)
    logger.info("VALIDACIÓN DEL DATASET INTEGRADO")
    logger.info("="*60)

    logger.info(f"\nRegistros totales: {len(integrated):,}")
    logger.info(f"Registros con PM2.5: {integrated['pm25'].notna().sum():,}")

    # Completitud por fuente
    logger.info("\nCompletitud por fuente satelital:")
    if 'era5_temperature_2m' in integrated.columns:
        era5_pct = integrated['era5_temperature_2m'].notna().sum() / len(integrated) * 100
        logger.info(f"  ERA5: {era5_pct:.1f}%")

    if 'modis_aod' in integrated.columns:
        modis_pct = integrated['modis_aod'].notna().sum() / len(integrated) * 100
        logger.info(f"  MODIS: {modis_pct:.1f}%")

    if 's5p_no2' in integrated.columns:
        s5p_pct = integrated['s5p_no2'].notna().sum() / len(integrated) * 100
        logger.info(f"  Sentinel-5P: {s5p_pct:.1f}%")

    # Registros completos (con todas las fuentes)
    complete_mask = integrated['pm25'].notna()
    if 'era5_temperature_2m' in integrated.columns:
        complete_mask &= integrated['era5_temperature_2m'].notna()
    if 'modis_aod' in integrated.columns:
        complete_mask &= integrated['modis_aod'].notna()
    if 's5p_no2' in integrated.columns:
        complete_mask &= integrated['s5p_no2'].notna()

    complete_count = complete_mask.sum()
    complete_pct = complete_count / len(integrated) * 100

    logger.info(f"\nRegistros con TODAS las variables: {complete_count:,} ({complete_pct:.1f}%)")

    # 7. Guardar dataset integrado
    output_file = output_dir / 'sinca_satellite_integrated.csv'
    logger.info(f"\n" + "="*60)
    logger.info(f"GUARDANDO DATASET INTEGRADO")
    logger.info(f"="*60)
    logger.info(f"Archivo: {output_file}")

    integrated.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Guardado exitosamente")
    logger.info(f"  Tamaño: {size_mb:.2f} MB")
    logger.info(f"  Columnas: {len(integrated.columns)}")
    logger.info(f"  Registros: {len(integrated):,}")

    # Guardar también versión completa (sin NaNs)
    if complete_count > 0:
        complete_file = output_dir / 'sinca_satellite_complete.csv'
        integrated[complete_mask].to_csv(complete_file, index=False)
        logger.info(f"\n✓ Dataset completo guardado: {complete_file}")
        logger.info(f"  Registros completos: {complete_count:,}")

    # Resumen de columnas
    logger.info(f"\nColumnas en dataset integrado:")
    for col in integrated.columns:
        logger.info(f"  • {col}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ INTEGRACIÓN COMPLETADA EXITOSAMENTE ✓✓✓")
    logger.info("="*70)

    return integrated


if __name__ == "__main__":
    df = main()
