#!/usr/bin/env python3
"""
Análisis estacional de PM2.5 en Santiago.

Evalúa performance del modelo por estación del año:
- Verano (Dic-Feb): Baja contaminación, alta variabilidad
- Otoño (Mar-May): Transición
- Invierno (Jun-Ago): Alta contaminación, episodios críticos
- Primavera (Sep-Nov): Transición

Análisis incluye:
1. Performance por estación (R², RMSE, MAE)
2. Feature importance por estación
3. Episodios críticos (PM2.5 > 80 μg/m³)
4. Autocorrelación por estación
5. Train/test cross-season (train invierno → test verano)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapeo de estaciones (hemisferio sur)
SEASON_MAP = {
    0: 'Verano',   # Dic-Feb
    1: 'Otoño',    # Mar-May
    2: 'Invierno', # Jun-Ago
    3: 'Primavera' # Sep-Nov
}

SEASON_MONTHS = {
    'Verano': [12, 1, 2],
    'Otoño': [3, 4, 5],
    'Invierno': [6, 7, 8],
    'Primavera': [9, 10, 11]
}


def add_season_labels(df):
    """Agregar etiquetas de estación."""

    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Verano
        elif month in [3, 4, 5]:
            return 1  # Otoño
        elif month in [6, 7, 8]:
            return 2  # Invierno
        else:
            return 3  # Primavera

    df['season'] = df['month'].apply(get_season)
    df['season_name'] = df['season'].map(SEASON_MAP)

    return df


def analyze_pm25_by_season(df):
    """
    Análisis descriptivo de PM2.5 por estación.

    Args:
        df: DataFrame con pm25 y season_name

    Returns:
        DataFrame con estadísticas por estación
    """
    logger.info("\n" + "="*70)
    logger.info("ESTADÍSTICAS DE PM2.5 POR ESTACIÓN")
    logger.info("="*70)

    stats_by_season = df.groupby('season_name')['pm25'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('p25', lambda x: x.quantile(0.25)),
        ('p50', lambda x: x.quantile(0.50)),
        ('p75', lambda x: x.quantile(0.75)),
        ('p95', lambda x: x.quantile(0.95))
    ]).round(2)

    # Ordenar por estación del año
    season_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    stats_by_season = stats_by_season.reindex(season_order)

    logger.info("\n" + str(stats_by_season))

    # Episodios críticos (PM2.5 > 80)
    logger.info("\n" + "="*70)
    logger.info("EPISODIOS CRÍTICOS (PM2.5 > 80 μg/m³)")
    logger.info("="*70)

    for season in season_order:
        season_data = df[df['season_name'] == season]
        n_critical = (season_data['pm25'] > 80).sum()
        pct_critical = (n_critical / len(season_data)) * 100

        logger.info(f"{season:12s}: {n_critical:4d} episodios ({pct_critical:5.2f}%)")

    return stats_by_season


def train_seasonal_models(df, feature_cols):
    """
    Entrenar modelos separados por estación.

    Args:
        df: DataFrame con features, pm25 y season_name
        feature_cols: Lista de features

    Returns:
        Dict con resultados por estación
    """
    logger.info("\n" + "="*70)
    logger.info("MODELOS POR ESTACIÓN (TRAIN/TEST TEMPORAL)")
    logger.info("="*70)

    season_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    results = {}

    for season in season_order:
        logger.info(f"\n{'='*70}")
        logger.info(f"ESTACIÓN: {season.upper()}")
        logger.info(f"{'='*70}")

        # Filtrar datos de la estación
        season_data = df[df['season_name'] == season].copy()

        logger.info(f"\nRegistros totales: {len(season_data):,}")

        if len(season_data) < 100:
            logger.warning(f"⚠️  Pocos datos para {season}, saltando...")
            continue

        # Train/test split temporal (80/20)
        season_data = season_data.sort_values('date').reset_index(drop=True)
        split_idx = int(len(season_data) * 0.8)

        train_data = season_data.iloc[:split_idx]
        test_data = season_data.iloc[split_idx:]

        logger.info(f"  Train: {len(train_data):,} ({len(train_data)/len(season_data)*100:.0f}%)")
        logger.info(f"  Test:  {len(test_data):,} ({len(test_data)/len(season_data)*100:.0f}%)")
        logger.info(f"  Train dates: {train_data['date'].min()} → {train_data['date'].max()}")
        logger.info(f"  Test dates:  {test_data['date'].min()} → {test_data['date'].max()}")

        # Preparar X, y
        X_train = train_data[feature_cols].values
        y_train = train_data['pm25'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['pm25'].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        if HAS_XGBOOST:
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
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        logger.info("\nEntrenando modelo...")
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        metrics = {
            'season': season,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }

        logger.info("\nResultados:")
        logger.info(f"  Train R²:   {metrics['train_r2']:.4f}")
        logger.info(f"  Train RMSE: {metrics['train_rmse']:.2f} μg/m³")
        logger.info(f"  Test R²:    {metrics['test_r2']:.4f}")
        logger.info(f"  Test RMSE:  {metrics['test_rmse']:.2f} μg/m³")
        logger.info(f"  Test MAE:   {metrics['test_mae']:.2f} μg/m³")

        overfit = metrics['train_r2'] - metrics['test_r2']
        logger.info(f"  Overfitting: {overfit:.4f}")

        # Feature importance
        if HAS_XGBOOST:
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)

            metrics['top_features'] = feat_imp.head(5)

            logger.info("\nTop 5 features:")
            for idx, row in feat_imp.head(5).iterrows():
                logger.info(f"  {row['feature']:35s}: {row['importance']:.4f}")

        results[season] = {
            'metrics': metrics,
            'model': model,
            'scaler': scaler,
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred_test,
                'dates': test_data['date'].values
            }
        }

    return results


def cross_season_validation(df, feature_cols):
    """
    Validación cruzada entre estaciones.

    Train en una estación, test en otra para ver transferibilidad.

    Args:
        df: DataFrame con features
        feature_cols: Lista de features

    Returns:
        DataFrame con resultados cruzados
    """
    logger.info("\n" + "="*70)
    logger.info("VALIDACIÓN CRUZADA ENTRE ESTACIONES")
    logger.info("="*70)
    logger.info("\nEntrenar en una estación → Predecir en otra")

    seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    cross_results = []

    for train_season in seasons:
        for test_season in seasons:
            if train_season == test_season:
                continue  # Skip same season

            logger.info(f"\nTrain: {train_season} → Test: {test_season}")

            # Data
            train_data = df[df['season_name'] == train_season]
            test_data = df[df['season_name'] == test_season]

            if len(train_data) < 100 or len(test_data) < 100:
                logger.info("  Datos insuficientes, saltando...")
                continue

            # Preparar
            X_train = train_data[feature_cols].values
            y_train = train_data['pm25'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['pm25'].values

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            if HAS_XGBOOST:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )

            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            logger.info(f"  R²: {r2:.4f}, RMSE: {rmse:.2f} μg/m³")

            cross_results.append({
                'train_season': train_season,
                'test_season': test_season,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })

    cross_df = pd.DataFrame(cross_results)

    return cross_df


def analyze_critical_episodes(df, threshold=80):
    """
    Analizar detección de episodios críticos por estación.

    Args:
        df: DataFrame con pm25 y predicciones por estación
        threshold: Umbral de PM2.5 crítico (default: 80 μg/m³)

    Returns:
        Dict con métricas de clasificación
    """
    logger.info("\n" + "="*70)
    logger.info(f"DETECCIÓN DE EPISODIOS CRÍTICOS (PM2.5 > {threshold})")
    logger.info("="*70)

    # Nota: Este análisis requiere predicciones almacenadas
    # Por ahora, solo mostramos distribución de episodios críticos

    for season in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
        season_data = df[df['season_name'] == season]

        n_total = len(season_data)
        n_critical = (season_data['pm25'] > threshold).sum()
        pct_critical = (n_critical / n_total) * 100 if n_total > 0 else 0

        mean_pm25 = season_data['pm25'].mean()
        max_pm25 = season_data['pm25'].max()

        logger.info(f"\n{season}:")
        logger.info(f"  Total observaciones: {n_total:,}")
        logger.info(f"  Episodios críticos:  {n_critical:,} ({pct_critical:.2f}%)")
        logger.info(f"  PM2.5 promedio:      {mean_pm25:.2f} μg/m³")
        logger.info(f"  PM2.5 máximo:        {max_pm25:.2f} μg/m³")


def plot_seasonal_analysis(stats_df, seasonal_results, cross_season_df, output_dir):
    """
    Visualizar análisis estacional.

    Args:
        stats_df: Estadísticas por estación
        seasonal_results: Resultados de modelos por estación
        cross_season_df: Resultados cross-season
        output_dir: Directorio de salida
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: PM2.5 por estación (boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Extraer datos para boxplot (necesitamos acceso al df original)
    # Por ahora, usamos stats_df

    ax = axes[0]
    seasons = stats_df.index.tolist()
    means = stats_df['mean'].values
    stds = stats_df['std'].values

    colors = ['#FFD700', '#FF8C00', '#4169E1', '#32CD32']  # Verano, Otoño, Invierno, Primavera

    ax.bar(seasons, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
    ax.set_title('PM2.5 Promedio por Estación', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Performance del modelo por estación
    ax = axes[1]

    # Extraer R² test de seasonal_results
    season_names = []
    r2_values = []

    for season, result in seasonal_results.items():
        season_names.append(season)
        r2_values.append(result['metrics']['test_r2'])

    ax.bar(season_names, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R² (Test)', fontsize=12)
    ax.set_title('Performance del Modelo por Estación', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Línea de referencia
    ax.axhline(y=0.76, color='red', linestyle='--', linewidth=2, label='R² Global (0.76)')
    ax.legend()

    plt.tight_layout()
    output_file = output_dir / 'seasonal_pm25_and_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Gráfico guardado: {output_file}")
    plt.close()

    # Plot 3: Cross-season validation heatmap
    if len(cross_season_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Pivot para heatmap
        pivot = cross_season_df.pivot(index='train_season', columns='test_season', values='r2')

        # Reordenar
        season_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
        pivot = pivot.reindex(index=season_order, columns=season_order)

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                   vmin=0, vmax=1, cbar_kws={'label': 'R²'},
                   linewidths=0.5, linecolor='black', ax=ax)

        ax.set_xlabel('Test Season', fontsize=12)
        ax.set_ylabel('Train Season', fontsize=12)
        ax.set_title('Cross-Season Validation (R²)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_file = output_dir / 'cross_season_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Gráfico guardado: {output_file}")
        plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS ESTACIONAL DE PM2.5")
    logger.info("="*70)

    # Cargar datos
    input_file = Path('data/processed/sinca_features_spatial.csv')

    logger.info(f"\nCargando datos: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")

    # Agregar a medias diarias
    logger.info("\nAgregando a medias diarias...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['year', 'month', 'day']
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_daily = df.groupby(['estacion', 'date'])[agg_cols].mean().reset_index()

    # Agregar columnas temporales
    df_daily['year'] = df_daily['date'].dt.year
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day'] = df_daily['date'].dt.day
    df_daily['day_of_year'] = df_daily['date'].dt.dayofyear
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek

    logger.info(f"  Registros diarios: {len(df_daily):,}")

    # Agregar estaciones
    df_daily = add_season_labels(df_daily)

    # Estadísticas descriptivas
    stats_df = analyze_pm25_by_season(df_daily)

    # Episodios críticos
    analyze_critical_episodes(df_daily, threshold=80)

    # Crear lag features (del script temporal_models.py)
    from temporal_models import create_lag_features, create_temporal_features

    df_daily = create_lag_features(df_daily, target_col='pm25', lags=[1, 2, 3, 7, 14])
    df_daily = create_temporal_features(df_daily)

    # Drop NaN
    df_clean = df_daily.dropna().reset_index(drop=True)
    logger.info(f"\nDespués de drop NaN: {len(df_clean):,} registros")

    # Features
    exclude_for_features = ['estacion', 'date', 'datetime', 'archivo', 'validado', 'pm25', 'season', 'season_name']
    all_cols = df_clean.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_for_features]

    logger.info(f"\nFeatures: {len(feature_cols)}")

    # Modelos por estación
    seasonal_results = train_seasonal_models(df_clean, feature_cols)

    # Cross-season validation
    cross_season_df = cross_season_validation(df_clean, feature_cols)

    # Guardar resultados
    logger.info("\n" + "="*70)
    logger.info("GUARDANDO RESULTADOS")
    logger.info("="*70)

    # Métricas por estación
    metrics_list = []
    for season, result in seasonal_results.items():
        metrics_list.append(result['metrics'])

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = Path('data/processed/seasonal_model_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"\n✓ Métricas por estación: {metrics_file}")

    # Cross-season
    if len(cross_season_df) > 0:
        cross_file = Path('data/processed/cross_season_validation.csv')
        cross_season_df.to_csv(cross_file, index=False)
        logger.info(f"✓ Validación cross-season: {cross_file}")

    # Estadísticas descriptivas
    stats_file = Path('data/processed/seasonal_pm25_statistics.csv')
    stats_df.to_csv(stats_file)
    logger.info(f"✓ Estadísticas por estación: {stats_file}")

    # Visualizar
    plot_seasonal_analysis(stats_df, seasonal_results, cross_season_df, Path('reports/figures'))

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ ANÁLISIS ESTACIONAL COMPLETADO ✓✓✓")
    logger.info("="*70)

    # Resumen ejecutivo
    logger.info("\n" + "="*70)
    logger.info("RESUMEN EJECUTIVO")
    logger.info("="*70)

    logger.info("\nPerformance por Estación (R² Test):")
    for season, result in seasonal_results.items():
        r2 = result['metrics']['test_r2']
        logger.info(f"  {season:12s}: R² = {r2:.4f}")

    return seasonal_results, cross_season_df, stats_df


if __name__ == "__main__":
    seasonal_results, cross_season_df, stats_df = main()
