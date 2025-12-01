#!/usr/bin/env python3
"""
Exporta mapa de PM2.5 a GeoTIFF para usar en QGIS/ArcGIS.

Lee grilla de predicciones y genera archivo raster GeoTIFF.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_grid_to_geotiff(grid_df, output_file, resolution=0.01):
    """
    Exportar grilla de predicciones a GeoTIFF.

    Args:
        grid_df: DataFrame con lon, lat, pm25_pred
        output_file: Ruta del archivo GeoTIFF de salida
        resolution: Resolución en grados (default: 0.01° ≈ 1.1 km)
    """
    logger.info("\n" + "="*70)
    logger.info("EXPORTANDO A GEOTIFF")
    logger.info("="*70)

    logger.info(f"\nPuntos de datos: {len(grid_df):,}")

    # Obtener bounds
    min_lon = grid_df['lon'].min()
    max_lon = grid_df['lon'].max()
    min_lat = grid_df['lat'].min()
    max_lat = grid_df['lat'].max()

    logger.info(f"\nBounds:")
    logger.info(f"  Longitud: [{min_lon:.4f}, {max_lon:.4f}]")
    logger.info(f"  Latitud: [{min_lat:.4f}, {max_lat:.4f}]")

    # Crear grilla regular para rasterización
    logger.info(f"\nResolución: {resolution}° (~{resolution*111:.1f} km)")

    lon_grid = np.arange(min_lon, max_lon + resolution, resolution)
    lat_grid = np.arange(min_lat, max_lat + resolution, resolution)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    logger.info(f"Dimensiones grilla: {len(lat_grid)} x {len(lon_grid)}")

    # Interpolar valores a grilla regular
    logger.info("\nInterpolando valores a grilla regular...")

    points = grid_df[['lon', 'lat']].values
    values = grid_df['pm25_pred'].values

    # Interpolar usando griddata (método cubic para suavizar)
    pm25_grid = griddata(
        points,
        values,
        (lon_mesh, lat_mesh),
        method='cubic',
        fill_value=np.nan
    )

    logger.info(f"  Valores interpolados: {np.sum(~np.isnan(pm25_grid)):,}")
    logger.info(f"  PM2.5 min: {np.nanmin(pm25_grid):.2f} μg/m³")
    logger.info(f"  PM2.5 max: {np.nanmax(pm25_grid):.2f} μg/m³")
    logger.info(f"  PM2.5 promedio: {np.nanmean(pm25_grid):.2f} μg/m³")

    # Voltear grilla (rasterio espera Y invertido)
    pm25_grid = np.flipud(pm25_grid)

    # Definir transformación geoespacial
    transform = from_bounds(
        min_lon, min_lat, max_lon, max_lat,
        pm25_grid.shape[1], pm25_grid.shape[0]
    )

    # Escribir GeoTIFF
    logger.info(f"\nEscribiendo GeoTIFF: {output_file}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=pm25_grid.shape[0],
        width=pm25_grid.shape[1],
        count=1,
        dtype=pm25_grid.dtype,
        crs='EPSG:4326',  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(pm25_grid, 1)

        # Agregar metadata
        dst.update_tags(
            DESCRIPTION='PM2.5 Predictions for Santiago Metropolitan Region',
            UNITS='μg/m³',
            SOURCE='SINCA + Satellite Data (ERA5, MODIS, Sentinel-5P)',
            MODEL='XGBoost Regression',
            DATE_CREATED='2025-11-16'
        )

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  Tamaño: {size_mb:.2f} MB")

    logger.info("\n" + "="*70)
    logger.info("✓ GeoTIFF EXPORTADO EXITOSAMENTE")
    logger.info("="*70)

    logger.info("\nPara usar en QGIS:")
    logger.info(f"  1. Abrir QGIS")
    logger.info(f"  2. Layer → Add Layer → Add Raster Layer")
    logger.info(f"  3. Seleccionar: {output_file.absolute()}")
    logger.info(f"  4. Aplicar estilo de color (amarillo-rojo para PM2.5)")

    return output_file


def main():
    logger.info("\n" + "="*70)
    logger.info("EXPORTACIÓN A GEOTIFF PARA QGIS")
    logger.info("="*70)

    # Cargar grilla de predicciones
    grid_file = Path('data/processed/pm25_grid_predictions.csv')

    logger.info(f"\nCargando grilla: {grid_file}")
    grid_df = pd.read_csv(grid_file)

    logger.info(f"  Registros: {len(grid_df):,}")
    logger.info(f"  Columnas: {list(grid_df.columns)}")

    # Exportar a GeoTIFF
    output_file = Path('data/processed/pm25_map_santiago.tif')

    export_grid_to_geotiff(
        grid_df,
        output_file,
        resolution=0.01  # ~1.1 km
    )

    logger.info("\n✓✓✓ EXPORTACIÓN COMPLETADA ✓✓✓")

    return output_file


if __name__ == "__main__":
    output = main()
