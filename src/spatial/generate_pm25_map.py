#!/usr/bin/env python3
"""
Genera mapa completo de PM2.5 para la Región Metropolitana.

Enfoque:
1. Entrenar modelo con TODAS las estaciones (no LOSO-CV)
2. Generar grilla de predicción para toda la región
3. Exportar mapa a GeoTIFF y visualización

Nota: Este es el enfoque de PRODUCCIÓN, no validación.
Para validación usar regression_kriging.py con LOSO-CV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_prediction_grid(bbox, resolution_km=1.0):
    """
    Crear grilla de predicción para la región.

    Args:
        bbox: (west, south, east, north) en grados
        resolution_km: Resolución en kilómetros

    Returns:
        DataFrame con lat, lon para cada punto de la grilla
    """
    west, south, east, north = bbox

    # Convertir resolución km a grados (aproximado para Santiago: ~111km por grado)
    res_deg = resolution_km / 111.0

    # Crear grilla
    lons = np.arange(west, east, res_deg)
    lats = np.arange(south, north, res_deg)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten
    grid_df = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten()
    })

    logger.info(f"  Grilla creada: {len(grid_df):,} puntos")
    logger.info(f"  Resolución: ~{resolution_km:.1f} km ({res_deg:.4f}°)")
    logger.info(f"  Dimensiones: {len(lats)} x {len(lons)}")

    return grid_df


def calculate_grid_features(grid_df, center_lat=-33.4489, center_lon=-70.6693):
    """
    Calcular features espaciales para cada punto de la grilla.

    Features básicas:
    - lat, lon
    - distance_to_center_km
    - elevation (aproximada por topografía simple)
    """
    logger.info("\\nCalculando features para grilla...")

    # Distancia al centro (aproximación esférica)
    R = 6371  # Radio Tierra en km

    dlat = np.radians(grid_df['lat'] - center_lat)
    dlon = np.radians(grid_df['lon'] - center_lon)

    a = np.sin(dlat/2)**2 + np.cos(np.radians(center_lat)) * \
        np.cos(np.radians(grid_df['lat'])) * np.sin(dlon/2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    grid_df['distance_to_center_km'] = R * c

    # Elevación aproximada (basada en distancia y dirección desde centro)
    # Santiago: centro ~500m, cerros al este ~800-1000m, oeste ~400m
    # Simplificación: elevación aumenta hacia el este

    lon_normalized = (grid_df['lon'] - center_lon) / 0.5  # Normalizar
    lat_normalized = (grid_df['lat'] - center_lat) / 0.5

    # Elevación base + componente este (cerros Andes al este)
    grid_df['elevation'] = 500 + lon_normalized * 200  # Mayor elevación al este

    # Clip a rangos razonables
    grid_df['elevation'] = grid_df['elevation'].clip(300, 1200)

    logger.info(f"  Features calculadas: {len(grid_df.columns) - 2}")
    logger.info(f"  Distancia al centro: [{grid_df['distance_to_center_km'].min():.2f}, {grid_df['distance_to_center_km'].max():.2f}] km")
    logger.info(f"  Elevación: [{grid_df['elevation'].min():.0f}, {grid_df['elevation'].max():.0f}] m")

    return grid_df


def train_full_model(df, feature_cols):
    """
    Entrenar modelo con TODAS las estaciones.

    Args:
        df: DataFrame con features y PM2.5
        feature_cols: Lista de columnas de features

    Returns:
        model: Modelo entrenado
        scaler: Scaler fit
    """
    logger.info("\\n" + "="*70)
    logger.info("ENTRENANDO MODELO COMPLETO (TODAS LAS ESTACIONES)")
    logger.info("="*70)

    X = df[feature_cols].values
    y = df['pm25'].values

    logger.info(f"\\nMuestras de entrenamiento: {len(y):,}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"PM2.5 promedio: {y.mean():.2f} μg/m³")
    logger.info(f"PM2.5 std: {y.std():.2f} μg/m³")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    if HAS_XGBOOST:
        logger.info("\\nModelo: XGBoost")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        logger.info("\\nModelo: Random Forest")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

    logger.info("Entrenando...")
    model.fit(X_scaled, y)

    # Training performance
    y_pred_train = model.predict(X_scaled)
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    r2_train = r2_score(y, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
    mae_train = mean_absolute_error(y, y_pred_train)

    logger.info("\\nPerformance en entrenamiento:")
    logger.info(f"  R²: {r2_train:.4f}")
    logger.info(f"  RMSE: {rmse_train:.2f} μg/m³")
    logger.info(f"  MAE: {mae_train:.2f} μg/m³")

    if HAS_XGBOOST:
        # Feature importance
        importance = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info("\\nTop 5 features más importantes:")
        for idx, row in feat_imp_df.head(5).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

    logger.info("\\n✓ Modelo entrenado exitosamente")

    return model, scaler


def predict_grid(model, scaler, grid_df, feature_cols):
    """
    Predecir PM2.5 en toda la grilla.

    Args:
        model: Modelo entrenado
        scaler: Scaler fit
        grid_df: DataFrame con features de grilla
        feature_cols: Columnas de features

    Returns:
        grid_df: DataFrame con predicciones agregadas
    """
    logger.info("\\n" + "="*70)
    logger.info("PREDICIENDO PM2.5 EN GRILLA")
    logger.info("="*70)

    logger.info(f"\\nPuntos de predicción: {len(grid_df):,}")

    # Preparar features
    X_grid = grid_df[feature_cols].values
    X_grid_scaled = scaler.transform(X_grid)

    # Predict
    logger.info("Generando predicciones...")
    pm25_pred = model.predict(X_grid_scaled)

    grid_df['pm25_pred'] = pm25_pred

    logger.info("\\nEstadísticas de predicciones:")
    logger.info(f"  PM2.5 min: {pm25_pred.min():.2f} μg/m³")
    logger.info(f"  PM2.5 max: {pm25_pred.max():.2f} μg/m³")
    logger.info(f"  PM2.5 promedio: {pm25_pred.mean():.2f} μg/m³")
    logger.info(f"  PM2.5 std: {pm25_pred.std():.2f} μg/m³")

    logger.info("\\n✓ Predicciones completadas")

    return grid_df


def visualize_map(grid_df, stations_df, output_dir):
    """
    Visualizar mapa de PM2.5 predicho.

    Args:
        grid_df: DataFrame con predicciones
        stations_df: DataFrame con estaciones de medición
        output_dir: Directorio de salida
    """
    logger.info("\\n" + "="*70)
    logger.info("GENERANDO VISUALIZACIÓN")
    logger.info("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 12))

    # Scatter plot de predicciones
    scatter = ax.scatter(
        grid_df['lon'],
        grid_df['lat'],
        c=grid_df['pm25_pred'],
        cmap='YlOrRd',
        s=50,
        alpha=0.6,
        edgecolors='none'
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PM2.5 Predicho (μg/m³)', fontsize=12)

    # Estaciones
    ax.scatter(
        stations_df['lon'],
        stations_df['lat'],
        c='blue',
        s=200,
        marker='^',
        edgecolors='black',
        linewidths=2,
        label='Estaciones SINCA',
        zorder=10
    )

    # Anotar estaciones
    for idx, row in stations_df.iterrows():
        ax.annotate(
            row['estacion'],
            (row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.set_title('Mapa de PM2.5 Predicho - Región Metropolitana', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Guardar
    output_file = output_dir / 'pm25_map_predictions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"\\n✓ Mapa guardado: {output_file}")

    plt.close()

    return output_file


def main():
    logger.info("\\n" + "="*70)
    logger.info("GENERACIÓN DE MAPA COMPLETO DE PM2.5")
    logger.info("="*70)

    # Cargar datos de entrenamiento
    input_file = Path('data/processed/sinca_features_spatial.csv')

    logger.info(f"\\nCargando datos: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")

    # Agregar a medias diarias
    logger.info("\\nAgregando a medias diarias por estación...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['year', 'month', 'day']
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_daily = df.groupby(['estacion', 'date'])[agg_cols].mean().reset_index()
    logger.info(f"  Registros agregados: {len(df_daily):,}")

    # Features para entrenamiento
    exclude_for_features = ['pm25', 'lat.1']  # lat.1 parece duplicado
    all_cols = df_daily.columns.tolist()

    feature_cols = [col for col in all_cols
                    if col not in ['estacion', 'date'] + exclude_for_features
                    and col in agg_cols]

    logger.info(f"\\nFeatures de entrenamiento: {len(feature_cols)}")
    for feat in feature_cols:
        logger.info(f"  • {feat}")

    # Entrenar modelo completo
    model, scaler = train_full_model(df_daily, feature_cols)

    # Crear grilla de predicción
    logger.info("\\n" + "="*70)
    logger.info("CREANDO GRILLA DE PREDICCIÓN")
    logger.info("="*70)

    # Bounding box de Santiago
    bbox = (-70.85, -33.65, -70.45, -33.25)  # (west, south, east, north)
    logger.info(f"\\nBounding box: {bbox}")

    grid_df = create_prediction_grid(bbox, resolution_km=2.0)

    # Calcular features para grilla
    grid_df = calculate_grid_features(grid_df)

    # Features mínimas (las que tenemos para grilla)
    grid_feature_cols = ['lat', 'lon', 'elevation', 'distance_to_center_km']

    logger.info(f"\\nFeatures disponibles en grilla: {grid_feature_cols}")

    # Nota: Faltan features meteorológicas/satelitales
    logger.info("\\n⚠️  ADVERTENCIA: Grilla solo tiene features espaciales básicas")
    logger.info("   Para mejores predicciones, agregar:")
    logger.info("   - Datos meteorológicos (ERA5)")
    logger.info("   - Datos satelitales (MODIS, Sentinel-5P)")
    logger.info("   - Features OSM (densidad vial, etc.)")

    # Por ahora, predecir solo con features espaciales
    # Necesitamos rellenar features faltantes con valores promedio

    logger.info("\\nRellenando features faltantes con valores promedio...")

    for feat in feature_cols:
        if feat not in grid_df.columns:
            mean_val = df_daily[feat].mean()
            grid_df[feat] = mean_val
            logger.info(f"  {feat}: {mean_val:.4f}")

    # Predecir
    grid_df = predict_grid(model, scaler, grid_df, feature_cols)

    # Estaciones
    stations = df[['estacion', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)

    # Visualizar
    output_dir = Path('reports/figures')
    visualize_map(grid_df, stations, output_dir)

    # Guardar grilla con predicciones
    output_file = Path('data/processed/pm25_grid_predictions.csv')
    grid_df.to_csv(output_file, index=False)
    logger.info(f"\\n✓ Grilla con predicciones guardada: {output_file}")

    logger.info("\\n" + "="*70)
    logger.info("✓✓✓ MAPA GENERADO EXITOSAMENTE ✓✓✓")
    logger.info("="*70)

    logger.info("\\nPróximos pasos:")
    logger.info("  1. Agregar features meteorológicas/satelitales a la grilla")
    logger.info("  2. Exportar a GeoTIFF para GIS")
    logger.info("  3. Crear mapa interactivo con Folium")
    logger.info("  4. Validar predicciones con LOSO-CV (regression_kriging.py)")

    return grid_df, model, scaler


if __name__ == "__main__":
    grid_df, model, scaler = main()
