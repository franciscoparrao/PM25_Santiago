#!/usr/bin/env python3
"""
Agrega metadatos de estaciones (lat, lon, elevación) al dataset consolidado.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_station_name(name):
    """Normaliza nombres de estaciones para matching."""
    # Mapeo manual de nombres
    mapping = {
        'Cerrillos II': 'Cerrillos',
        'Parque O\'Higgins': 'Parque O\'Higgins',
        'El Bosque': 'El Bosque',
        'Independencia': 'Independencia',
        'Las Condes': 'Las Condes',
        'Cerro Navia': 'Cerro Navia',
        'Pudahuel': 'Pudahuel',
        'Puente Alto': 'Puente Alto',
        'Talagante': 'Talagante'
    }
    return mapping.get(name, name)

def main():
    logger.info("="*70)
    logger.info("AGREGANDO METADATOS DE ESTACIONES")
    logger.info("="*70)

    # Cargar dataset consolidado
    consolidated_file = Path('data/processed/sinca_pm25_consolidated.csv')
    logger.info(f"\nCargando: {consolidated_file}")
    df = pd.read_csv(consolidated_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")

    # Cargar metadatos
    metadata_file = Path('data/external/sinca_stations_metadata.csv')
    logger.info(f"\nCargando metadatos: {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    logger.info(f"  Estaciones con metadatos: {len(metadata)}")

    # Normalizar nombres
    df['station_normalized'] = df['estacion'].apply(normalize_station_name)
    metadata['station_normalized'] = metadata['station']

    # Merge con metadatos
    logger.info("\nAgregando coordenadas...")
    df_with_coords = df.merge(
        metadata[['station_normalized', 'lat', 'lon', 'elevation']],
        on='station_normalized',
        how='left'
    )

    # Verificar matching
    matched = df_with_coords['lat'].notna().sum()
    total = len(df_with_coords)
    logger.info(f"  Registros con coordenadas: {matched:,}/{total:,} ({matched/total*100:.1f}%)")

    # Revisar estaciones sin match
    no_match = df_with_coords[df_with_coords['lat'].isna()]['estacion'].unique()
    if len(no_match) > 0:
        logger.warning(f"\nEstaciones sin coordenadas: {list(no_match)}")

    # Eliminar columna temporal
    df_with_coords = df_with_coords.drop('station_normalized', axis=1)

    # Reordenar columnas
    cols = ['datetime', 'date', 'year', 'month', 'day', 'estacion',
            'lat', 'lon', 'elevation', 'pm25', 'validado',
            'pm25_validado', 'pm25_preliminar', 'archivo']

    df_final = df_with_coords[cols]

    # Guardar
    output_file = Path('data/processed/sinca_pm25_master.csv')
    logger.info(f"\nGuardando dataset maestro: {output_file}")
    df_final.to_csv(output_file, index=False)

    logger.info(f"✓ Dataset guardado")
    logger.info(f"  Tamaño: {output_file.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  Columnas: {list(df_final.columns)}")

    # Estadísticas finales
    logger.info("\n" + "="*70)
    logger.info("DATASET MAESTRO - ESTADÍSTICAS FINALES")
    logger.info("="*70)
    logger.info(f"Total de registros: {len(df_final):,}")
    logger.info(f"Rango temporal: {df_final['datetime'].min()} a {df_final['datetime'].max()}")
    logger.info(f"Estaciones: {df_final['estacion'].nunique()}")
    logger.info(f"Registros con PM2.5: {df_final['pm25'].notna().sum():,}")
    logger.info(f"Registros validados: {df_final['validado'].sum():,}")

    logger.info("\nEstaciones incluidas:")
    for station in sorted(df_final['estacion'].unique()):
        count = len(df_final[df_final['estacion'] == station])
        lat = df_final[df_final['estacion'] == station]['lat'].iloc[0]
        lon = df_final[df_final['estacion'] == station]['lon'].iloc[0]
        logger.info(f"  • {station}: {count:,} registros (lat={lat}, lon={lon})")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ DATASET MAESTRO COMPLETADO ✓✓✓")
    logger.info("="*70)

if __name__ == "__main__":
    main()
