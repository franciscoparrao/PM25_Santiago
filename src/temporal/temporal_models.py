#!/usr/bin/env python3
"""
Modelado temporal de PM2.5 usando lag features.

Enfoque:
- Predecir PM2.5 en el TIEMPO (no en el espacio)
- Usar valores pasados de PM2.5 (lags) como features
- Validación temporal (train/test split cronológico)
- Mejor performance esperado: R² 0.60-0.85

Features:
1. Lag features: PM2.5 de 1, 2, 3, 7 días anteriores
2. Rolling statistics: Media móvil 3, 7, 14 días
3. Temporales: día semana, mes, estación del año
4. Meteorológicas: temperatura, viento, precipitación
5. Satelitales: NO2, AOD
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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


def create_lag_features(df, target_col='pm25', lags=[1, 2, 3, 7], group_col='estacion'):
    """
    Crear lag features por estación.

    Args:
        df: DataFrame con datos temporales
        target_col: Columna target
        lags: Lista de lags (días)
        group_col: Columna para agrupar (estación)

    Returns:
        DataFrame con lag features
    """
    logger.info("\nCreando lag features...")

    df = df.sort_values(['estacion', 'date']).copy()

    for lag in lags:
        col_name = f'{target_col}_lag_{lag}d'
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        logger.info(f"  • {col_name}")

    # Rolling statistics
    windows = [3, 7, 14]

    for window in windows:
        col_name = f'{target_col}_rolling_mean_{window}d'
        df[col_name] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        logger.info(f"  • {col_name}")

        col_name = f'{target_col}_rolling_std_{window}d'
        df[col_name] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        logger.info(f"  • {col_name}")

    # Diff features (cambio día a día)
    df[f'{target_col}_diff_1d'] = df.groupby(group_col)[target_col].diff(1)
    logger.info(f"  • {target_col}_diff_1d")

    logger.info(f"\nLag features creadas: {len(lags) + len(windows)*2 + 1}")

    return df


def create_temporal_features(df):
    """
    Crear features temporales cíclicas.

    Args:
        df: DataFrame con columna 'date'

    Returns:
        DataFrame con features temporales
    """
    logger.info("\nCreando features temporales cíclicas...")

    # Ya existen: year, month, day, day_of_year, day_of_week

    # Mes cíclico (sin, cos)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    logger.info("  • month_sin, month_cos")

    # Día del año cíclico
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    logger.info("  • day_of_year_sin, day_of_year_cos")

    # Día de la semana cíclico
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    logger.info("  • day_of_week_sin, day_of_week_cos")

    # Estación del año (simplificado para hemisferio sur)
    # Verano: Dic-Feb (12, 1, 2)
    # Otoño: Mar-May (3, 4, 5)
    # Invierno: Jun-Ago (6, 7, 8)
    # Primavera: Sep-Nov (9, 10, 11)

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
    logger.info("  • season (0=Verano, 1=Otoño, 2=Invierno, 3=Primavera)")

    # One-hot encoding de estación
    for season_val in range(4):
        df[f'season_{season_val}'] = (df['season'] == season_val).astype(int)

    logger.info("  • season_0, season_1, season_2, season_3 (one-hot)")

    # Fin de semana
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    logger.info("  • is_weekend")

    logger.info("\nFeatures temporales creadas: 13")

    return df


def temporal_train_test_split(df, test_size=0.2):
    """
    Split temporal: últimos test_size% de datos como test.

    Args:
        df: DataFrame ordenado por fecha
        test_size: Proporción de test (default: 0.2 = 20%)

    Returns:
        df_train, df_test
    """
    df = df.sort_values('date').reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    logger.info(f"\nTemporal train/test split:")
    logger.info(f"  Train: {len(df_train):,} ({(1-test_size)*100:.0f}%)")
    logger.info(f"  Test:  {len(df_test):,} ({test_size*100:.0f}%)")
    logger.info(f"  Train dates: {df_train['date'].min()} → {df_train['date'].max()}")
    logger.info(f"  Test dates:  {df_test['date'].min()} → {df_test['date'].max()}")

    return df_train, df_test


def train_temporal_model(X_train, y_train, X_test, y_test, model_name='XGBoost'):
    """
    Entrenar modelo temporal.

    Args:
        X_train, y_train: Train set
        X_test, y_test: Test set
        model_name: 'XGBoost' o 'RandomForest'

    Returns:
        model, scaler, metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"ENTRENANDO MODELO: {model_name}")
    logger.info(f"{'='*70}")

    logger.info(f"\nTrain samples: {len(y_train):,}")
    logger.info(f"Test samples:  {len(y_test):,}")
    logger.info(f"Features: {X_train.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    if model_name == 'XGBoost' and HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=300,
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
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

    logger.info("\nEntrenando...")
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    metrics = {}

    # Train
    metrics['train_r2'] = r2_score(y_train, y_pred_train)
    metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)

    # Test
    metrics['test_r2'] = r2_score(y_test, y_pred_test)
    metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)

    logger.info("\n" + "="*70)
    logger.info("RESULTADOS")
    logger.info("="*70)

    logger.info("\nTrain:")
    logger.info(f"  R²:   {metrics['train_r2']:.4f}")
    logger.info(f"  RMSE: {metrics['train_rmse']:.2f} μg/m³")
    logger.info(f"  MAE:  {metrics['train_mae']:.2f} μg/m³")

    logger.info("\nTest:")
    logger.info(f"  R²:   {metrics['test_r2']:.4f}")
    logger.info(f"  RMSE: {metrics['test_rmse']:.2f} μg/m³")
    logger.info(f"  MAE:  {metrics['test_mae']:.2f} μg/m³")

    # Overfitting check
    overfit = metrics['train_r2'] - metrics['test_r2']
    logger.info(f"\nOverfitting (Train R² - Test R²): {overfit:.4f}")

    if overfit > 0.1:
        logger.warning("  ⚠️  Modelo puede estar overfitting")
    else:
        logger.info("  ✓ Modelo generaliza bien")

    return model, scaler, metrics, y_pred_test


def plot_predictions(y_test, y_pred, dates_test, output_dir):
    """
    Visualizar predicciones vs observaciones.

    Args:
        y_test: Valores reales
        y_pred: Predicciones
        dates_test: Fechas de test
        output_dir: Directorio de salida
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Time series
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(dates_test, y_test, 'o-', label='Real', alpha=0.6, markersize=3)
    ax.plot(dates_test, y_pred, 'o-', label='Predicho', alpha=0.6, markersize=3)

    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
    ax.set_title('PM2.5: Real vs Predicho (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'temporal_predictions_timeseries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Gráfico guardado: {output_file}")
    plt.close()

    # Figure 2: Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_test, y_pred, alpha=0.3, s=20)

    # Diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfecto')

    ax.set_xlabel('PM2.5 Real (μg/m³)', fontsize=12)
    ax.set_ylabel('PM2.5 Predicho (μg/m³)', fontsize=12)
    ax.set_title('Real vs Predicho - Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_file = output_dir / 'temporal_predictions_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Gráfico guardado: {output_file}")
    plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("MODELADO TEMPORAL DE PM2.5")
    logger.info("="*70)

    # Cargar datos
    input_file = Path('data/processed/sinca_features_spatial.csv')

    logger.info(f"\nCargando datos: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")
    logger.info(f"  Fechas: {df['date'].min()} → {df['date'].max()}")

    # Agregar a medias diarias
    logger.info("\nAgregando a medias diarias...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['year', 'month', 'day']
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_daily = df.groupby(['estacion', 'date'])[agg_cols].mean().reset_index()

    # Agregar columnas temporales de nuevo
    df_daily['year'] = df_daily['date'].dt.year
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day'] = df_daily['date'].dt.day
    df_daily['day_of_year'] = df_daily['date'].dt.dayofyear
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek

    logger.info(f"  Registros diarios: {len(df_daily):,}")

    # Crear lag features
    df_daily = create_lag_features(df_daily, target_col='pm25', lags=[1, 2, 3, 7, 14])

    # Crear features temporales
    df_daily = create_temporal_features(df_daily)

    # Drop NaN (por lags)
    df_clean = df_daily.dropna().reset_index(drop=True)
    logger.info(f"\nDespués de drop NaN: {len(df_clean):,} registros")

    # Features para modelo temporal
    exclude_for_features = ['estacion', 'date', 'datetime', 'archivo', 'validado', 'pm25', 'season']

    all_cols = df_clean.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_for_features]

    logger.info(f"\nFeatures totales: {len(feature_cols)}")

    # Categorizar features
    lag_features = [col for col in feature_cols if 'lag' in col or 'rolling' in col or 'diff' in col]
    temporal_features = [col for col in feature_cols if any(x in col for x in ['month', 'day', 'year', 'season', 'weekend'])]
    meteo_features = [col for col in feature_cols if any(x in col for x in ['era5', 'wind', 'precipitation'])]
    sat_features = [col for col in feature_cols if 's5p' in col]
    spatial_features = [col for col in feature_cols if col in ['lat', 'lon', 'elevation', 'distance_to_center_km']]

    logger.info(f"\nCategorías de features:")
    logger.info(f"  Lag features:      {len(lag_features)}")
    logger.info(f"  Temporal features: {len(temporal_features)}")
    logger.info(f"  Meteo features:    {len(meteo_features)}")
    logger.info(f"  Satelital features:{len(sat_features)}")
    logger.info(f"  Spatial features:  {len(spatial_features)}")

    # Temporal train/test split
    df_train, df_test = temporal_train_test_split(df_clean, test_size=0.2)

    # Preparar X, y
    X_train = df_train[feature_cols].values
    y_train = df_train['pm25'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['pm25'].values
    dates_test = df_test['date'].values

    # Entrenar modelo
    model, scaler, metrics, y_pred_test = train_temporal_model(
        X_train, y_train, X_test, y_test,
        model_name='XGBoost'
    )

    # Feature importance
    if HAS_XGBOOST:
        importance = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info("\n" + "="*70)
        logger.info("TOP 15 FEATURES MÁS IMPORTANTES")
        logger.info("="*70)

        for idx, row in feat_imp_df.head(15).iterrows():
            logger.info(f"  {row['feature']:35s}: {row['importance']:.4f}")

        # Guardar feature importance
        feat_imp_file = Path('data/processed/temporal_feature_importance.csv')
        feat_imp_df.to_csv(feat_imp_file, index=False)
        logger.info(f"\n✓ Feature importance guardado: {feat_imp_file}")

    # Visualizar
    plot_predictions(y_test, y_pred_test, dates_test, Path('reports/figures'))

    # Guardar resultados
    results_df = pd.DataFrame([metrics])
    results_file = Path('data/processed/temporal_model_results.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n✓ Resultados guardados: {results_file}")

    # Guardar predicciones de test
    predictions_df = pd.DataFrame({
        'date': dates_test,
        'pm25_real': y_test,
        'pm25_pred': y_pred_test,
        'error': y_test - y_pred_test
    })
    predictions_file = Path('data/processed/temporal_predictions_test.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"✓ Predicciones guardadas: {predictions_file}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ MODELADO TEMPORAL COMPLETADO ✓✓✓")
    logger.info("="*70)

    logger.info(f"\n🎯 RESULTADO FINAL: R² Test = {metrics['test_r2']:.4f}")

    if metrics['test_r2'] > 0.60:
        logger.info("✓ Excelente! Modelo temporal funciona mucho mejor que espacial")
    elif metrics['test_r2'] > 0.30:
        logger.info("✓ Bueno! Modelo captura patrones temporales")
    else:
        logger.info("⚠️  Modelo necesita mejoras")

    return model, scaler, metrics


if __name__ == "__main__":
    model, scaler, metrics = main()
