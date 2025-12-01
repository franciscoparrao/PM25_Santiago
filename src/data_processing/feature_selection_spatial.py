#!/usr/bin/env python3
"""
Feature Selection para Modelado ESPACIAL (sin lags).

Este análisis evalúa qué features son importantes cuando NO tenemos
datos históricos de PM2.5 (escenario de predicción en nuevas ubicaciones).

Simula Leave-One-Station-Out Cross-Validation para medir capacidad
de generalización espacial.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_spatial_features(df):
    """
    Identifica features válidas para predicción espacial (sin lags).

    Args:
        df: DataFrame completo

    Returns:
        Lista de features espaciales
    """
    # Excluir metadatos, target, y LAG FEATURES
    exclude_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'archivo', 'validado',
        'pm25', 'pm25_validado', 'pm25_preliminar',
        # LAG FEATURES (no disponibles en nuevas ubicaciones)
        'pm25_lag1', 'pm25_lag7', 'pm25_ma7', 'pm25_ma30',
        'pm25_std7', 'pm25_diff1'
    ]

    spatial_features = [col for col in df.columns if col not in exclude_cols]

    return spatial_features


def evaluate_spatial_generalization(df, feature_cols, target='pm25'):
    """
    Evalúa generalización espacial usando Leave-One-Station-Out CV.

    Args:
        df: DataFrame
        feature_cols: Features a evaluar
        target: Target variable

    Returns:
        Dict con resultados por estación
    """
    logger.info("\n" + "="*70)
    logger.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
    logger.info("="*70)

    X = df[feature_cols].copy()
    y = df[target]
    groups = df['estacion']

    stations = df['estacion'].unique()
    results = []

    logger.info(f"\nEvaluando {len(stations)} estaciones...")

    for i, test_station in enumerate(stations, 1):
        # Split: entrenar en todas las estaciones EXCEPTO la de test
        train_mask = df['estacion'] != test_station
        test_mask = df['estacion'] == test_station

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'station': test_station,
            'n_test': len(y_test),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })

        logger.info(f"  [{i}/{len(stations)}] {test_station:20s} "
                   f"R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
                   f"(n={len(y_test)})")

    results_df = pd.DataFrame(results)

    # Promedio ponderado por n_test
    total_samples = results_df['n_test'].sum()
    avg_r2 = (results_df['r2'] * results_df['n_test']).sum() / total_samples
    avg_rmse = (results_df['rmse'] * results_df['n_test']).sum() / total_samples
    avg_mae = (results_df['mae'] * results_df['n_test']).sum() / total_samples

    logger.info("\n" + "="*70)
    logger.info(f"PROMEDIO (ponderado por n_test):")
    logger.info(f"  R² = {avg_r2:.3f}")
    logger.info(f"  RMSE = {avg_rmse:.2f} μg/m³")
    logger.info(f"  MAE = {avg_mae:.2f} μg/m³")

    return results_df, avg_r2, avg_rmse, avg_mae


def calculate_spatial_feature_importance(df, feature_cols, target='pm25'):
    """
    Calcula feature importance entrenando en todas las estaciones
    (sin usar lags).

    Args:
        df: DataFrame
        feature_cols: Features espaciales
        target: Target

    Returns:
        DataFrame con importancias
    """
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE - SPATIAL MODEL (sin lags)")
    logger.info("="*60)

    X = df[feature_cols].copy()
    y = df[target]

    logger.info(f"\nEntrenando Random Forest (100 árboles)...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 15 features para predicción espacial:")
    for i, row in importances.head(15).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.6f}")

    return importances


def compare_with_temporal(spatial_importance, temporal_rankings_file):
    """
    Compara importancias espaciales vs temporales.

    Args:
        spatial_importance: DataFrame con importancias espaciales
        temporal_rankings_file: Path a feature_rankings.csv (con lags)

    Returns:
        DataFrame comparativo
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARACIÓN: ESPACIAL vs TEMPORAL")
    logger.info("="*60)

    # Cargar rankings temporales
    temporal = pd.read_csv(temporal_rankings_file)

    # Filtrar solo features compartidas (sin lags)
    shared_features = set(spatial_importance['feature']) & set(temporal['feature'])

    comparison = []
    for feat in shared_features:
        spatial_imp = spatial_importance[spatial_importance['feature'] == feat]['importance'].iloc[0]
        temporal_imp = temporal[temporal['feature'] == feat]['importance'].iloc[0]

        comparison.append({
            'feature': feat,
            'spatial_importance': spatial_imp,
            'temporal_importance': temporal_imp,
            'ratio': spatial_imp / temporal_imp if temporal_imp > 0 else np.inf
        })

    comp_df = pd.DataFrame(comparison).sort_values('spatial_importance', ascending=False)

    logger.info("\nFeatures con MAYOR importancia en modelo espacial:")
    logger.info("(ratio > 5 = mucho más importante espacialmente)")
    high_spatial = comp_df[comp_df['ratio'] > 5].head(10)
    for _, row in high_spatial.iterrows():
        logger.info(f"  {row['feature']:40s} "
                   f"Spatial={row['spatial_importance']:.6f}  "
                   f"Temporal={row['temporal_importance']:.6f}  "
                   f"Ratio={row['ratio']:.1f}x")

    return comp_df


def create_visualizations(spatial_importance, spatial_results, output_dir):
    """
    Crea visualizaciones para análisis espacial.

    Args:
        spatial_importance: DataFrame con importancias
        spatial_results: DataFrame con resultados LOSO-CV
        output_dir: Directorio para guardar figuras
    """
    logger.info("\n" + "="*60)
    logger.info("CREANDO VISUALIZACIONES ESPACIALES")
    logger.info("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Importance Espacial (top 20)
    plt.figure(figsize=(12, 8))
    top_features = spatial_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Spatial Model - No Lags)')
    plt.title('Top 20 Features - Spatial Prediction (Leave-One-Station-Out)')
    plt.tight_layout()

    output_file = output_dir / 'spatial_feature_importance_top20.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 2. R² por estación
    plt.figure(figsize=(10, 6))
    spatial_results_sorted = spatial_results.sort_values('r2', ascending=False)
    colors = ['green' if r2 > 0.5 else 'orange' if r2 > 0.3 else 'red'
              for r2 in spatial_results_sorted['r2']]

    plt.barh(range(len(spatial_results_sorted)), spatial_results_sorted['r2'], color=colors)
    plt.yticks(range(len(spatial_results_sorted)), spatial_results_sorted['station'])
    plt.xlabel('R² Score')
    plt.title('Spatial Generalization - R² per Station (LOSO-CV)')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='R²=0.5')
    plt.legend()
    plt.tight_layout()

    output_file = output_dir / 'spatial_r2_by_station.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION - MODELADO ESPACIAL (sin lags)")
    logger.info("="*70)

    # Cargar dataset
    input_file = Path('data/processed/sinca_features_engineered.csv')
    logger.info(f"\nCargando dataset: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")

    # Identificar features espaciales (sin lags)
    spatial_features = get_spatial_features(df)
    logger.info(f"\nFeatures para predicción espacial: {len(spatial_features)}")
    logger.info("(excluye lags de PM2.5)")

    # 1. Feature Importance espacial
    spatial_importance = calculate_spatial_feature_importance(
        df, spatial_features, target='pm25'
    )

    # 2. Evaluación LOSO-CV
    spatial_results, avg_r2, avg_rmse, avg_mae = evaluate_spatial_generalization(
        df, spatial_features, target='pm25'
    )

    # 3. Comparación con temporal
    temporal_rankings = Path('data/processed/feature_rankings.csv')
    if temporal_rankings.exists():
        comparison = compare_with_temporal(spatial_importance, temporal_rankings)
    else:
        logger.warning(f"No se encontró {temporal_rankings} - omitiendo comparación")
        comparison = None

    # 4. Visualizaciones
    create_visualizations(
        spatial_importance, spatial_results,
        output_dir=Path('reports/figures')
    )

    # 5. Selección de features espaciales (top 10)
    top_spatial_features = spatial_importance.head(10)['feature'].tolist()

    logger.info("\n" + "="*60)
    logger.info("FEATURES SELECCIONADAS PARA MODELADO ESPACIAL")
    logger.info("="*60)
    logger.info(f"\nTop 10 features espaciales:")
    for i, feat in enumerate(top_spatial_features, 1):
        imp = spatial_importance[spatial_importance['feature'] == feat]['importance'].iloc[0]
        logger.info(f"  {i:2d}. {feat:40s} {imp:.6f}")

    # 6. Guardar dataset espacial
    spatial_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'lat', 'lon', 'elevation',
        'pm25', 'validado', 'archivo'
    ] + top_spatial_features

    df_spatial = df[spatial_cols].copy()

    output_file = Path('data/processed/sinca_features_spatial.csv')
    df_spatial.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"\n✓ Dataset espacial guardado: {output_file}")
    logger.info(f"  Tamaño: {size_mb:.2f} MB")
    logger.info(f"  Features: {len(top_spatial_features)}")
    logger.info(f"  R² promedio (LOSO-CV): {avg_r2:.3f}")

    # 7. Guardar resultados
    results_file = Path('data/processed/spatial_loso_results.csv')
    spatial_results.to_csv(results_file, index=False)
    logger.info(f"✓ Resultados LOSO-CV guardados: {results_file}")

    spatial_importance_file = Path('data/processed/spatial_feature_importance.csv')
    spatial_importance.to_csv(spatial_importance_file, index=False)
    logger.info(f"✓ Importancias espaciales guardadas: {spatial_importance_file}")

    if comparison is not None:
        comparison_file = Path('data/processed/spatial_vs_temporal_comparison.csv')
        comparison.to_csv(comparison_file, index=False)
        logger.info(f"✓ Comparación guardada: {comparison_file}")

    # Resumen final
    logger.info("\n" + "="*70)
    logger.info("RESUMEN - PREDICCIÓN ESPACIAL")
    logger.info("="*70)
    logger.info(f"\nPerformance LOSO-CV (sin lags):")
    logger.info(f"  R² promedio:  {avg_r2:.3f}")
    logger.info(f"  RMSE promedio: {avg_rmse:.2f} μg/m³")
    logger.info(f"  MAE promedio:  {avg_mae:.2f} μg/m³")

    logger.info(f"\nTop 3 features espaciales:")
    for i, row in spatial_importance.head(3).iterrows():
        logger.info(f"  {i+1}. {row['feature']:40s} {row['importance']:.4f}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ ANÁLISIS ESPACIAL COMPLETADO ✓✓✓")
    logger.info("="*70)

    return df_spatial, spatial_results, spatial_importance


if __name__ == "__main__":
    df_spatial, results, importance = main()
