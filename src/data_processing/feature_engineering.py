#!/usr/bin/env python3
"""
Script para crear features derivadas del dataset integrado SINCA + Satélite.

Features creadas:
1. Wind-derived: velocidad y dirección del viento
2. Temporal: día de semana, fin de semana, estación del año
3. Lag features: PM2.5 rezagado, promedios móviles
4. Meteorological: temperatura en Celsius, humedad relativa
5. Interaction features: combinaciones de variables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_wind_features(df):
    """
    Crea features derivadas del viento.

    Args:
        df: DataFrame con componentes u y v del viento

    Returns:
        DataFrame con nuevas columnas de viento
    """
    logger.info("\nCreando features de viento...")

    # Velocidad del viento (magnitud del vector)
    df['wind_speed'] = np.sqrt(
        df['era5_u_component_of_wind_10m']**2 +
        df['era5_v_component_of_wind_10m']**2
    )

    # Dirección del viento (en radianes, luego convertir a grados)
    df['wind_direction_rad'] = np.arctan2(
        df['era5_v_component_of_wind_10m'],
        df['era5_u_component_of_wind_10m']
    )

    # Convertir a grados (0-360)
    df['wind_direction_deg'] = (df['wind_direction_rad'] * 180 / np.pi) % 360

    logger.info(f"  ✓ wind_speed (m/s)")
    logger.info(f"  ✓ wind_direction_deg (0-360°)")

    return df


def create_temporal_features(df):
    """
    Crea features temporales basadas en la fecha.

    Args:
        df: DataFrame con columna datetime

    Returns:
        DataFrame con nuevas columnas temporales
    """
    logger.info("\nCreando features temporales...")

    # Día de la semana (0=Lunes, 6=Domingo)
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Fin de semana (sábado y domingo)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Estación del año (1=verano, 2=otoño, 3=invierno, 4=primavera)
    # Para hemisferio sur: Dic-Feb=verano, Mar-May=otoño, Jun-Ago=invierno, Sep-Nov=primavera
    df['season'] = df['month'].map({
        12: 1, 1: 1, 2: 1,  # Verano
        3: 2, 4: 2, 5: 2,   # Otoño
        6: 3, 7: 3, 8: 3,   # Invierno
        9: 4, 10: 4, 11: 4  # Primavera
    })

    # Día del año (1-365/366)
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Trimestre
    df['quarter'] = df['datetime'].dt.quarter

    logger.info(f"  ✓ day_of_week (0-6)")
    logger.info(f"  ✓ is_weekend (0/1)")
    logger.info(f"  ✓ season (1-4)")
    logger.info(f"  ✓ day_of_year (1-365)")
    logger.info(f"  ✓ quarter (1-4)")

    return df


def create_lag_features(df):
    """
    Crea features de rezago (lag) de PM2.5 por estación.

    Args:
        df: DataFrame con PM2.5 y estación

    Returns:
        DataFrame con nuevas columnas de lag
    """
    logger.info("\nCreando features de lag (por estación)...")

    # Ordenar por estación y fecha para lags correctos
    df = df.sort_values(['estacion', 'datetime'])

    # Lag 1 día (valor de ayer)
    df['pm25_lag1'] = df.groupby('estacion')['pm25'].shift(1)

    # Lag 7 días (valor de hace una semana)
    df['pm25_lag7'] = df.groupby('estacion')['pm25'].shift(7)

    # Promedio móvil 7 días
    df['pm25_ma7'] = df.groupby('estacion')['pm25'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Promedio móvil 30 días
    df['pm25_ma30'] = df.groupby('estacion')['pm25'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # Desviación estándar móvil 7 días (volatilidad)
    df['pm25_std7'] = df.groupby('estacion')['pm25'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    # Diferencia de PM2.5 respecto al día anterior
    df['pm25_diff1'] = df.groupby('estacion')['pm25'].diff(1)

    logger.info(f"  ✓ pm25_lag1 (ayer)")
    logger.info(f"  ✓ pm25_lag7 (hace 7 días)")
    logger.info(f"  ✓ pm25_ma7 (promedio 7 días)")
    logger.info(f"  ✓ pm25_ma30 (promedio 30 días)")
    logger.info(f"  ✓ pm25_std7 (volatilidad 7 días)")
    logger.info(f"  ✓ pm25_diff1 (cambio diario)")

    return df


def create_meteorological_features(df):
    """
    Crea features meteorológicas derivadas.

    Args:
        df: DataFrame con variables meteorológicas ERA5

    Returns:
        DataFrame con nuevas columnas meteorológicas
    """
    logger.info("\nCreando features meteorológicas...")

    # Temperatura en Celsius (desde Kelvin)
    df['temperature_celsius'] = df['era5_temperature_2m'] - 273.15
    df['dewpoint_celsius'] = df['era5_dewpoint_temperature_2m'] - 273.15

    # Humedad relativa (aproximación usando Magnus formula)
    # RH = 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
    def calculate_rh(temp_c, dewpoint_c):
        numerator = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
        denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
        return 100 * (numerator / denominator)

    df['relative_humidity'] = calculate_rh(
        df['temperature_celsius'],
        df['dewpoint_celsius']
    )

    # Presión en hPa (desde Pa)
    df['surface_pressure_hpa'] = df['era5_surface_pressure'] / 100

    # Precipitación acumulada 7 días
    df['precipitation_sum7'] = df.groupby('estacion')['era5_total_precipitation_hourly'].transform(
        lambda x: x.rolling(window=7, min_periods=1).sum()
    )

    logger.info(f"  ✓ temperature_celsius (°C)")
    logger.info(f"  ✓ dewpoint_celsius (°C)")
    logger.info(f"  ✓ relative_humidity (%)")
    logger.info(f"  ✓ surface_pressure_hpa (hPa)")
    logger.info(f"  ✓ precipitation_sum7 (acum. 7 días)")

    return df


def create_interaction_features(df):
    """
    Crea features de interacción entre variables.

    Args:
        df: DataFrame con todas las features

    Returns:
        DataFrame con nuevas columnas de interacción
    """
    logger.info("\nCreando features de interacción...")

    # Temperatura × AOD (aerosoles con temperatura)
    df['temp_aod_interaction'] = df['temperature_celsius'] * df['modis_aod']

    # Velocidad viento × NO2 (dispersión de contaminantes)
    df['wind_no2_interaction'] = df['wind_speed'] * df['s5p_no2']

    # Humedad × AOD
    df['humidity_aod_interaction'] = df['relative_humidity'] * df['modis_aod']

    # Estabilidad atmosférica (presión × temperatura)
    df['atmospheric_stability'] = df['surface_pressure_hpa'] * df['temperature_celsius']

    logger.info(f"  ✓ temp_aod_interaction")
    logger.info(f"  ✓ wind_no2_interaction")
    logger.info(f"  ✓ humidity_aod_interaction")
    logger.info(f"  ✓ atmospheric_stability")

    return df


def create_spatial_features(df):
    """
    Crea features espaciales basadas en ubicación de estaciones.

    Args:
        df: DataFrame con lat, lon, elevation

    Returns:
        DataFrame con nuevas columnas espaciales
    """
    logger.info("\nCreando features espaciales...")

    # Distancia al centro de Santiago (Plaza de Armas: -33.4372, -70.6506)
    centro_lat, centro_lon = -33.4372, -70.6506

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radio Tierra en km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['distance_to_center_km'] = df.apply(
        lambda row: haversine_distance(row['lat'], row['lon'], centro_lat, centro_lon),
        axis=1
    )

    # Elevación normalizada (útil para modelos)
    df['elevation_normalized'] = (df['elevation'] - df['elevation'].mean()) / df['elevation'].std()

    logger.info(f"  ✓ distance_to_center_km")
    logger.info(f"  ✓ elevation_normalized")

    return df


def main():
    logger.info("\n" + "="*70)
    logger.info("FEATURE ENGINEERING - SINCA + SATÉLITE")
    logger.info("="*70)

    # Cargar dataset integrado (versión completa)
    input_file = Path('data/processed/sinca_satellite_complete.csv')
    logger.info(f"\nCargando dataset: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Columnas originales: {len(df.columns)}")
    logger.info(f"  Periodo: {df['datetime'].min()} → {df['datetime'].max()}")

    # Aplicar transformaciones
    df = create_wind_features(df)
    df = create_temporal_features(df)
    df = create_meteorological_features(df)
    df = create_lag_features(df)
    df = create_interaction_features(df)
    df = create_spatial_features(df)

    # Resumen de features creadas
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE FEATURES")
    logger.info("="*60)

    new_features = [col for col in df.columns if col not in pd.read_csv(input_file, nrows=0).columns]
    logger.info(f"\nFeatures nuevas creadas: {len(new_features)}")
    logger.info(f"Total de columnas: {len(df.columns)}")

    logger.info("\nNuevas features:")
    for feature in sorted(new_features):
        logger.info(f"  • {feature}")

    # Verificar valores faltantes en nuevas features
    logger.info("\n" + "="*60)
    logger.info("VALORES FALTANTES")
    logger.info("="*60)

    missing_counts = df[new_features].isnull().sum()
    if missing_counts.sum() > 0:
        logger.info("\nFeatures con valores faltantes:")
        for feature, count in missing_counts[missing_counts > 0].items():
            pct = count / len(df) * 100
            logger.info(f"  • {feature}: {count:,} ({pct:.1f}%)")
    else:
        logger.info("✓ No hay valores faltantes en features nuevas")

    # Eliminar filas con NaNs solo en features críticas (lag features)
    # pm25_validado y pm25_preliminar pueden tener NaNs (son mutuamente exclusivas)
    critical_features = ['pm25_lag1', 'pm25_lag7', 'pm25_ma7', 'pm25_ma30',
                         'pm25_std7', 'pm25_diff1']

    original_len = len(df)
    df_clean = df.dropna(subset=critical_features)
    removed = original_len - len(df_clean)

    if removed > 0:
        logger.info(f"\nFilas eliminadas por NaNs en lag features: {removed:,}")
        logger.info(f"Dataset final: {len(df_clean):,} registros")
    else:
        logger.info(f"\n✓ No se eliminaron filas")
        logger.info(f"Dataset final: {len(df_clean):,} registros")
        df_clean = df

    # Guardar dataset con features
    output_file = Path('data/processed/sinca_features_engineered.csv')
    logger.info("\n" + "="*60)
    logger.info("GUARDANDO DATASET")
    logger.info("="*60)
    logger.info(f"Archivo: {output_file}")

    df_clean.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Guardado exitosamente")
    logger.info(f"  Tamaño: {size_mb:.2f} MB")
    logger.info(f"  Registros: {len(df_clean):,}")
    logger.info(f"  Columnas: {len(df_clean.columns)}")

    # Estadísticas finales
    logger.info("\n" + "="*60)
    logger.info("ESTADÍSTICAS FINALES")
    logger.info("="*60)

    logger.info(f"\nTarget (PM2.5):")
    logger.info(f"  Media: {df_clean['pm25'].mean():.2f} μg/m³")
    logger.info(f"  Mediana: {df_clean['pm25'].median():.2f} μg/m³")
    logger.info(f"  Std: {df_clean['pm25'].std():.2f} μg/m³")
    logger.info(f"  Min: {df_clean['pm25'].min():.2f} μg/m³")
    logger.info(f"  Max: {df_clean['pm25'].max():.2f} μg/m³")

    logger.info(f"\nEstaciones: {df_clean['estacion'].nunique()}")
    logger.info(f"Periodo temporal: {df_clean['datetime'].min()} → {df_clean['datetime'].max()}")
    logger.info(f"Total días: {(df_clean['datetime'].max() - df_clean['datetime'].min()).days}")

    # Distribución por estación
    logger.info("\nRegistros por estación:")
    station_counts = df_clean['estacion'].value_counts().sort_index()
    for station, count in station_counts.items():
        logger.info(f"  • {station}: {count:,}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ FEATURE ENGINEERING COMPLETADO ✓✓✓")
    logger.info("="*70)

    return df_clean


if __name__ == "__main__":
    df = main()
