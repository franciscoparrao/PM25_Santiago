#!/usr/bin/env python3
"""
Forecasting REAL de PM2.5 con walk-forward validation.

Diferencia clave con temporal_models.py:
- temporal_models.py: Usa pm25_lag_1d CONOCIDO (R² = 0.9984)
- forecasting.py: Predice SIN conocer valores futuros (R² ~0.60-0.80)

Métodos:
1. Direct forecasting: Entrenar modelo específico para cada horizonte
2. Recursive forecasting: Usar predicciones pasadas como lags
3. Walk-forward validation: Rolling window en tiempo real

Horizontes: 1, 3, 7 días adelante
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


def create_forecast_features(df, target_col='pm25', max_lag=14, group_col='estacion'):
    """
    Crear features para forecasting (similar a temporal_models.py).

    Args:
        df: DataFrame con datos temporales
        target_col: Columna target
        max_lag: Máximo lag a usar
        group_col: Columna para agrupar

    Returns:
        DataFrame con features
    """
    logger.info("\nCreando features para forecasting...")

    df = df.sort_values([group_col, 'date']).copy()

    # Lag features
    lags = [1, 2, 3, 7, 14]
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}d'
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14]:
        df[f'{target_col}_rolling_mean_{window}d'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_std_{window}d'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Diff
    df[f'{target_col}_diff_1d'] = df.groupby(group_col)[target_col].diff(1)

    # Temporal features cíclicas
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Estación
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
    for season_val in range(4):
        df[f'season_{season_val}'] = (df['season'] == season_val).astype(int)

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    logger.info(f"  Features creadas: {len([c for c in df.columns if 'lag' in c or 'rolling' in c or 'sin' in c or 'season' in c])}")

    return df


def walk_forward_validation(df, feature_cols, horizon=1, train_size=365*3, step_size=7):
    """
    Walk-forward validation para forecasting.

    Simula predicción en tiempo real:
    1. Entrenar con ventana de train_size días
    2. Predecir horizon días adelante
    3. Avanzar step_size días
    4. Repetir

    Args:
        df: DataFrame con features y target
        feature_cols: Lista de features
        horizon: Días adelante a predecir (1, 3, 7)
        train_size: Tamaño ventana de entrenamiento (días)
        step_size: Días a avanzar cada iteración

    Returns:
        DataFrame con predicciones y métricas
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"WALK-FORWARD VALIDATION: {horizon}-DAY AHEAD FORECAST")
    logger.info(f"{'='*70}")

    logger.info(f"\nParámetros:")
    logger.info(f"  Horizon: {horizon} días adelante")
    logger.info(f"  Train window: {train_size} días (~{train_size/365:.1f} años)")
    logger.info(f"  Step size: {step_size} días")

    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)

    # Crear target desplazado (horizon días adelante)
    df[f'pm25_target_{horizon}d'] = df.groupby('estacion')['pm25'].shift(-horizon)

    # Drop NaN
    df_clean = df.dropna(subset=[f'pm25_target_{horizon}d'] + feature_cols).reset_index(drop=True)

    logger.info(f"\nDatos disponibles: {len(df_clean):,} registros")
    logger.info(f"  Fechas: {df_clean['date'].min()} → {df_clean['date'].max()}")

    # Walk-forward loop
    predictions = []
    start_idx = train_size
    end_idx = len(df_clean) - horizon

    n_iterations = (end_idx - start_idx) // step_size
    logger.info(f"\nIteraciones de validación: {n_iterations}")

    for i, test_start in enumerate(range(start_idx, end_idx, step_size)):
        train_end = test_start
        train_start = train_end - train_size

        # Asegurar que train_start >= 0
        if train_start < 0:
            train_start = 0

        test_end = min(test_start + step_size, end_idx)

        # Split
        df_train = df_clean.iloc[train_start:train_end]
        df_test = df_clean.iloc[test_start:test_end]

        if len(df_test) == 0:
            break

        X_train = df_train[feature_cols].values
        y_train = df_train[f'pm25_target_{horizon}d'].values
        X_test = df_test[feature_cols].values
        y_test = df_test[f'pm25_target_{horizon}d'].values

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
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Store predictions
        for j in range(len(df_test)):
            predictions.append({
                'date': df_test.iloc[j]['date'],
                'estacion': df_test.iloc[j]['estacion'],
                'pm25_real': y_test[j],
                'pm25_pred': y_pred[j],
                'horizon': horizon,
                'iteration': i
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Iteración {i+1}/{n_iterations} completada")

    predictions_df = pd.DataFrame(predictions)

    logger.info(f"\nPredicciones totales: {len(predictions_df):,}")

    return predictions_df


def evaluate_forecast(predictions_df, horizon):
    """
    Evaluar performance de forecasting.

    Args:
        predictions_df: DataFrame con predicciones
        horizon: Horizonte de predicción

    Returns:
        Dict con métricas
    """
    y_real = predictions_df['pm25_real'].values
    y_pred = predictions_df['pm25_pred'].values

    metrics = {
        'horizon': horizon,
        'n_predictions': len(predictions_df),
        'r2': r2_score(y_real, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_real, y_pred)),
        'mae': mean_absolute_error(y_real, y_pred),
        'mape': np.mean(np.abs((y_real - y_pred) / y_real)) * 100  # Mean Absolute Percentage Error
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"MÉTRICAS: {horizon}-DAY AHEAD FORECAST")
    logger.info(f"{'='*70}")
    logger.info(f"  Predicciones: {metrics['n_predictions']:,}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} μg/m³")
    logger.info(f"  MAE:  {metrics['mae']:.2f} μg/m³")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics


def baseline_persistence(df, horizon=1):
    """
    Baseline: Persistencia (asumir que PM2.5 futuro = PM2.5 de hoy).

    Args:
        df: DataFrame con pm25
        horizon: Días adelante

    Returns:
        DataFrame con predicciones
    """
    logger.info(f"\nCalculando baseline: Persistencia ({horizon} días)")

    df = df.sort_values('date').reset_index(drop=True)

    # Predicción = valor actual
    df['pm25_pred_persistence'] = df['pm25']

    # Target = valor horizon días adelante
    df['pm25_target'] = df.groupby('estacion')['pm25'].shift(-horizon)

    # Drop NaN
    df_clean = df.dropna(subset=['pm25_target']).copy()

    # Evaluar
    y_real = df_clean['pm25_target'].values
    y_pred = df_clean['pm25_pred_persistence'].values

    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)

    logger.info(f"  Persistencia R²: {r2:.4f}")
    logger.info(f"  Persistencia RMSE: {rmse:.2f} μg/m³")
    logger.info(f"  Persistencia MAE: {mae:.2f} μg/m³")

    return {
        'horizon': horizon,
        'model': 'Persistencia',
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def baseline_historical_mean(df, horizon=1, window=30):
    """
    Baseline: Media histórica (promedio de últimos window días).

    Args:
        df: DataFrame con pm25
        horizon: Días adelante
        window: Ventana para calcular media

    Returns:
        Dict con métricas
    """
    logger.info(f"\nCalculando baseline: Media histórica ({window} días)")

    df = df.sort_values('date').reset_index(drop=True)

    # Media móvil
    df['pm25_pred_mean'] = df.groupby('estacion')['pm25'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # Target
    df['pm25_target'] = df.groupby('estacion')['pm25'].shift(-horizon)

    # Drop NaN
    df_clean = df.dropna(subset=['pm25_target']).copy()

    # Evaluar
    y_real = df_clean['pm25_target'].values
    y_pred = df_clean['pm25_pred_mean'].values

    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)

    logger.info(f"  Media histórica R²: {r2:.4f}")
    logger.info(f"  Media histórica RMSE: {rmse:.2f} μg/m³")
    logger.info(f"  Media histórica MAE: {mae:.2f} μg/m³")

    return {
        'horizon': horizon,
        'model': f'Media {window}d',
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def plot_forecast_results(all_predictions, all_metrics, output_dir):
    """
    Visualizar resultados de forecasting.

    Args:
        all_predictions: Dict con predicciones por horizonte
        all_metrics: DataFrame con métricas
        output_dir: Directorio de salida
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Métricas por horizonte
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    horizons = all_metrics['horizon'].unique()

    for idx, metric in enumerate(['r2', 'rmse', 'mae']):
        ax = axes[idx]

        for model in all_metrics['model'].unique():
            model_data = all_metrics[all_metrics['model'] == model]
            ax.plot(model_data['horizon'], model_data[metric], 'o-',
                   label=model, markersize=8, linewidth=2)

        ax.set_xlabel('Horizonte (días)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} por Horizonte', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons)

    plt.tight_layout()
    output_file = output_dir / 'forecast_metrics_by_horizon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Gráfico guardado: {output_file}")
    plt.close()

    # Plot 2: Time series para cada horizonte
    for horizon in [1, 3, 7]:
        if horizon not in all_predictions:
            continue

        pred_df = all_predictions[horizon]

        # Tomar solo últimos 60 días para visualización
        pred_df = pred_df.sort_values('date')
        pred_df_recent = pred_df.tail(60)

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(pred_df_recent['date'], pred_df_recent['pm25_real'],
               'o-', label='Real', alpha=0.6, markersize=4)
        ax.plot(pred_df_recent['date'], pred_df_recent['pm25_pred'],
               'o-', label='Predicho', alpha=0.6, markersize=4)

        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
        ax.set_title(f'Forecast {horizon}-Day Ahead (últimos 60 días)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / f'forecast_{horizon}d_timeseries.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Gráfico guardado: {output_file}")
        plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("FORECASTING REAL DE PM2.5")
    logger.info("="*70)

    # Cargar datos
    input_file = Path('data/processed/sinca_features_spatial.csv')

    logger.info(f"\nCargando datos: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")

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

    # Crear features
    df_daily = create_forecast_features(df_daily)

    # Features
    exclude_for_features = ['estacion', 'date', 'datetime', 'archivo', 'validado', 'pm25', 'season']
    all_cols = df_daily.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_for_features
                   and col in df_daily.columns]

    logger.info(f"\nFeatures totales: {len(feature_cols)}")

    # Horizontes a evaluar
    horizons = [1, 3, 7]

    all_predictions = {}
    all_metrics = []

    # Walk-forward validation para cada horizonte
    for horizon in horizons:
        logger.info(f"\n{'='*70}")
        logger.info(f"HORIZON: {horizon} DÍAS ADELANTE")
        logger.info(f"{'='*70}")

        # Walk-forward validation
        predictions_df = walk_forward_validation(
            df_daily,
            feature_cols,
            horizon=horizon,
            train_size=365*3,  # 3 años de entrenamiento
            step_size=7  # Avanzar 1 semana cada vez
        )

        # Evaluar
        metrics = evaluate_forecast(predictions_df, horizon)
        metrics['model'] = 'XGBoost'
        all_metrics.append(metrics)

        all_predictions[horizon] = predictions_df

        # Guardar predicciones
        pred_file = Path(f'data/processed/forecast_{horizon}d_predictions.csv')
        predictions_df.to_csv(pred_file, index=False)
        logger.info(f"\n✓ Predicciones guardadas: {pred_file}")

        # Baselines
        baseline_pers = baseline_persistence(df_daily, horizon=horizon)
        all_metrics.append(baseline_pers)

        baseline_mean = baseline_historical_mean(df_daily, horizon=horizon, window=30)
        all_metrics.append(baseline_mean)

    # Consolidar métricas
    metrics_df = pd.DataFrame(all_metrics)

    logger.info("\n" + "="*70)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*70)
    print("\n", metrics_df.to_string(index=False))

    # Guardar métricas
    metrics_file = Path('data/processed/forecast_metrics_comparison.csv')
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"\n✓ Métricas guardadas: {metrics_file}")

    # Visualizar
    plot_forecast_results(all_predictions, metrics_df, Path('reports/figures'))

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ FORECASTING COMPLETADO ✓✓✓")
    logger.info("="*70)

    # Análisis de degradación
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS DE DEGRADACIÓN POR HORIZONTE")
    logger.info("="*70)

    xgb_metrics = metrics_df[metrics_df['model'] == 'XGBoost'].sort_values('horizon')

    logger.info("\nXGBoost Performance:")
    for idx, row in xgb_metrics.iterrows():
        logger.info(f"  {row['horizon']}d ahead: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.2f} μg/m³")

    if len(xgb_metrics) > 1:
        r2_1d = xgb_metrics[xgb_metrics['horizon'] == 1]['r2'].values[0]
        r2_7d = xgb_metrics[xgb_metrics['horizon'] == 7]['r2'].values[0]
        degradation = ((r2_1d - r2_7d) / r2_1d) * 100

        logger.info(f"\nDegradación 1d → 7d: {degradation:.1f}%")

    return metrics_df, all_predictions


if __name__ == "__main__":
    metrics_df, all_predictions = main()
