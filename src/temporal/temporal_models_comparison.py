"""
Comparación Completa de Modelos Temporales para Predicción de PM2.5

Este script compara XGBoost con modelos clásicos de series de tiempo:
1. ARIMA/SARIMA (Auto ARIMA)
2. Prophet (Facebook)
3. XGBoost (con intervalos de confianza via quantile regression)

Todos los modelos usan walk-forward validation para evaluación honesta.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorios
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports' / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Cargar datos de forecasting y preparar para análisis."""
    logging.info("\n" + "="*70)
    logging.info("CARGANDO DATOS")
    logging.info("="*70)

    # Cargar predicciones de XGBoost (ya calculadas)
    df_1d = pd.read_csv(DATA_DIR / 'forecast_1d_predictions.csv', parse_dates=['date'])

    logging.info(f"\n✓ XGBoost predictions: {len(df_1d):,} registros")
    logging.info(f"  Periodo: {df_1d['date'].min()} → {df_1d['date'].max()}")

    return df_1d


def fit_arima_model(train_data, horizon=1, max_samples=1000):
    """
    Entrenar modelo ARIMA usando pmdarima (auto_arima).

    Args:
        train_data: Serie temporal de entrenamiento
        horizon: Horizonte de predicción (días)
        max_samples: Máximo de samples para ARIMA (por performance)

    Returns:
        Modelo ARIMA entrenado
    """
    try:
        from pmdarima import auto_arima
    except ImportError:
        logging.warning("⚠ pmdarima no instalado. Instalando...")
        import subprocess
        subprocess.run(['pip3', 'install', 'pmdarima', '--quiet'])
        from pmdarima import auto_arima

    # Limitar muestras para performance
    if len(train_data) > max_samples:
        train_subset = train_data[-max_samples:]
    else:
        train_subset = train_data

    # Auto ARIMA con configuración razonable
    model = auto_arima(
        train_subset,
        seasonal=True,
        m=7,  # Periodo semanal
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        max_d=2, max_D=1,
        start_p=1, start_q=1,
        start_P=1, start_Q=1,
        suppress_warnings=True,
        stepwise=True,
        error_action='ignore',
        trace=False,
        n_jobs=-1
    )

    return model


def fit_prophet_model(train_df):
    """
    Entrenar modelo Prophet (Facebook).

    Args:
        train_df: DataFrame con columnas 'date' y 'pm25'

    Returns:
        Modelo Prophet entrenado
    """
    try:
        from prophet import Prophet
    except ImportError:
        logging.warning("⚠ prophet no instalado. Instalando...")
        import subprocess
        subprocess.run(['pip3', 'install', 'prophet', '--quiet'])
        from prophet import Prophet

    # Preparar datos en formato Prophet
    prophet_df = pd.DataFrame({
        'ds': train_df['date'],
        'y': train_df['pm25']
    })

    # Configurar modelo con estacionalidad
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )

    model.fit(prophet_df)

    return model


def walk_forward_arima(df, horizon=1, train_size=365*3, step_size=30, n_iterations=50):
    """
    Walk-forward validation para ARIMA.

    NOTA: ARIMA es MUY lento, limitamos a 50 iteraciones (vs 2161 de XGBoost)
    y step_size=30 días (vs 7 de XGBoost).
    """
    logging.info("\n" + "="*70)
    logging.info(f"WALK-FORWARD VALIDATION: ARIMA ({horizon}-DAY AHEAD)")
    logging.info("="*70)
    logging.info(f"\n⚠ ARIMA es computacionalmente costoso:")
    logging.info(f"  Limitando a {n_iterations} iteraciones (vs 2,161 de XGBoost)")
    logging.info(f"  Step size: {step_size} días (vs 7 de XGBoost)")

    # Agrupar por fecha (promedio de todas las estaciones)
    df_daily = df.groupby('date').agg({'pm25': 'mean'}).reset_index()
    df_daily = df_daily.sort_values('date').reset_index(drop=True)

    predictions = []
    actual_values = []
    dates = []

    start_idx = train_size + horizon
    end_idx = len(df_daily) - horizon

    # Limitar iteraciones
    step_indices = list(range(start_idx, end_idx, step_size))[:n_iterations]
    n_iterations_actual = len(step_indices)

    logging.info(f"\nIteraciones a realizar: {n_iterations_actual}")

    for i, test_start in enumerate(step_indices):
        train_end = test_start
        train_start = train_end - train_size

        # Train data
        train_series = df_daily.loc[train_start:train_end-1, 'pm25'].values

        # Test data (horizon días adelante)
        test_idx = test_start + horizon - 1
        if test_idx >= len(df_daily):
            break

        y_true = df_daily.loc[test_idx, 'pm25']
        test_date = df_daily.loc[test_idx, 'date']

        try:
            # Entrenar ARIMA
            model = fit_arima_model(train_series, horizon=horizon)

            # Predecir
            y_pred = model.predict(n_periods=horizon)[-1]

            predictions.append(y_pred)
            actual_values.append(y_true)
            dates.append(test_date)

        except Exception as e:
            logging.warning(f"  Iteración {i+1} falló: {str(e)[:50]}")
            continue

        if (i + 1) % 10 == 0:
            logging.info(f"  Iteración {i+1}/{n_iterations_actual} completada")

    # Resultados
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual_values,
        'predicted': predictions,
        'horizon': horizon
    })

    return results_df


def walk_forward_prophet(df, horizon=1, train_size=365*3, step_size=30, n_iterations=50):
    """
    Walk-forward validation para Prophet.

    NOTA: Prophet también es lento, limitamos iteraciones.
    """
    logging.info("\n" + "="*70)
    logging.info(f"WALK-FORWARD VALIDATION: PROPHET ({horizon}-DAY AHEAD)")
    logging.info("="*70)
    logging.info(f"\n⚠ Prophet es computacionalmente costoso:")
    logging.info(f"  Limitando a {n_iterations} iteraciones")
    logging.info(f"  Step size: {step_size} días")

    # Agrupar por fecha
    df_daily = df.groupby('date').agg({'pm25': 'mean'}).reset_index()
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    df_daily.columns = ['date', 'pm25']

    predictions = []
    actual_values = []
    dates = []

    start_idx = train_size + horizon
    end_idx = len(df_daily) - horizon

    step_indices = list(range(start_idx, end_idx, step_size))[:n_iterations]
    n_iterations_actual = len(step_indices)

    logging.info(f"\nIteraciones a realizar: {n_iterations_actual}")

    for i, test_start in enumerate(step_indices):
        train_end = test_start
        train_start = train_end - train_size

        # Train data
        train_df = df_daily.loc[train_start:train_end-1].copy()

        # Test data
        test_idx = test_start + horizon - 1
        if test_idx >= len(df_daily):
            break

        y_true = df_daily.loc[test_idx, 'pm25']
        test_date = df_daily.loc[test_idx, 'date']

        try:
            # Entrenar Prophet
            model = fit_prophet_model(train_df)

            # Crear futuro
            future = pd.DataFrame({'ds': [test_date]})

            # Predecir
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values[0]

            predictions.append(y_pred)
            actual_values.append(y_true)
            dates.append(test_date)

        except Exception as e:
            logging.warning(f"  Iteración {i+1} falló: {str(e)[:50]}")
            continue

        if (i + 1) % 10 == 0:
            logging.info(f"  Iteración {i+1}/{n_iterations_actual} completada")

    # Resultados
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual_values,
        'predicted': predictions,
        'horizon': horizon
    })

    return results_df


def calculate_metrics(y_true, y_pred):
    """Calcular métricas de evaluación."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }


def xgboost_quantile_regression(df, feature_cols, horizon=1, quantiles=[0.05, 0.5, 0.95],
                                train_size=365*3, step_size=7, max_iterations=100):
    """
    XGBoost con intervalos de confianza usando quantile regression.

    Args:
        quantiles: Cuantiles a predecir (0.05, 0.5, 0.95 = intervalo 90%)
    """
    logging.info("\n" + "="*70)
    logging.info(f"XGBOOST QUANTILE REGRESSION ({horizon}-DAY AHEAD)")
    logging.info("="*70)
    logging.info(f"\nCuantiles: {quantiles}")
    logging.info(f"Intervalo de confianza: {(quantiles[-1] - quantiles[0])*100:.0f}%")

    # Preparar datos diarios
    df_daily = df.groupby(['date', 'estacion']).agg({
        **{col: 'mean' for col in feature_cols},
        'pm25': 'mean'
    }).reset_index()
    df_daily = df_daily.sort_values(['estacion', 'date']).reset_index(drop=True)

    # Resultados por cuantil
    results_by_quantile = {q: {'predictions': [], 'actual': [], 'dates': []}
                           for q in quantiles}

    start_idx = train_size + horizon
    end_idx = len(df_daily) - horizon
    step_indices = list(range(start_idx, end_idx, step_size))[:max_iterations]

    logging.info(f"\nIteraciones: {len(step_indices)}")

    for i, test_start in enumerate(step_indices):
        train_end = test_start
        train_start = train_end - train_size

        # Split
        train_data = df_daily.iloc[train_start:train_end]
        test_idx = test_start + horizon - 1

        if test_idx >= len(df_daily):
            break

        test_data = df_daily.iloc[[test_idx]]

        X_train = train_data[feature_cols].values
        y_train = train_data['pm25'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['pm25'].values[0]
        test_date = test_data['date'].values[0]

        # Entrenar modelo para cada cuantil
        for q in quantiles:
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[0]

            results_by_quantile[q]['predictions'].append(y_pred)
            results_by_quantile[q]['actual'].append(y_test)
            results_by_quantile[q]['dates'].append(pd.Timestamp(test_date))

        if (i + 1) % 20 == 0:
            logging.info(f"  Iteración {i+1}/{len(step_indices)} completada")

    # Crear DataFrame con intervalos
    results_df = pd.DataFrame({
        'date': results_by_quantile[0.5]['dates'],
        'actual': results_by_quantile[0.5]['actual'],
        'predicted': results_by_quantile[0.5]['predictions'],
        'lower_bound': results_by_quantile[quantiles[0]]['predictions'],
        'upper_bound': results_by_quantile[quantiles[-1]]['predictions'],
        'horizon': horizon
    })

    # Calcular cobertura del intervalo
    coverage = ((results_df['actual'] >= results_df['lower_bound']) &
                (results_df['actual'] <= results_df['upper_bound'])).mean()

    logging.info(f"\n✓ Cobertura del intervalo: {coverage*100:.2f}%")
    logging.info(f"  Esperado: {(quantiles[-1] - quantiles[0])*100:.0f}%")

    return results_df


def plot_models_comparison(xgb_results, arima_results, prophet_results, quantile_results):
    """Crear visualizaciones comparando todos los modelos."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Comparación de R² por modelo
    ax1 = fig.add_subplot(gs[0, 0])

    models_metrics = []
    for name, df in [('XGBoost', xgb_results), ('ARIMA', arima_results),
                     ('Prophet', prophet_results)]:
        if df is not None and len(df) > 0:
            metrics = calculate_metrics(df['actual'].values, df['predicted'].values)
            models_metrics.append({'Model': name, **metrics})

    metrics_df = pd.DataFrame(models_metrics)

    bars = ax1.bar(metrics_df['Model'], metrics_df['R²'],
                   color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(metrics_df['R²']) * 1.1])

    for bar, val in zip(bars, metrics_df['R²']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Comparación de RMSE
    ax2 = fig.add_subplot(gs[0, 1])

    bars = ax2.bar(metrics_df['Model'], metrics_df['RMSE'],
                   color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('RMSE (μg/m³)', fontsize=11)
    ax2.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, metrics_df['RMSE']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Comparación de MAE
    ax3 = fig.add_subplot(gs[0, 2])

    bars = ax3.bar(metrics_df['Model'], metrics_df['MAE'],
                   color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('MAE (μg/m³)', fontsize=11)
    ax3.set_title('MAE Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, metrics_df['MAE']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Serie temporal con intervalos de confianza (últimos 180 días)
    ax4 = fig.add_subplot(gs[1, :])

    if quantile_results is not None and len(quantile_results) > 180:
        plot_df = quantile_results.tail(180).copy()

        ax4.fill_between(plot_df['date'], plot_df['lower_bound'], plot_df['upper_bound'],
                        alpha=0.2, color='#3498db', label='90% Confidence Interval')
        ax4.plot(plot_df['date'], plot_df['actual'], 'o-',
                label='Actual', color='black', markersize=2, linewidth=1.5, alpha=0.7)
        ax4.plot(plot_df['date'], plot_df['predicted'], 's-',
                label='XGBoost Predicted', color='#3498db', markersize=2, linewidth=1.5)

        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
        ax4.set_title('XGBoost Predictions with 90% Confidence Interval (Last 180 days)',
                     fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 5. Scatter plot: Predicted vs Actual (XGBoost)
    ax5 = fig.add_subplot(gs[2, 0])

    if len(xgb_results) > 0:
        ax5.scatter(xgb_results['actual'], xgb_results['predicted'],
                   alpha=0.3, s=10, color='#3498db')

        # Línea diagonal (predicción perfecta)
        min_val = min(xgb_results['actual'].min(), xgb_results['predicted'].min())
        max_val = max(xgb_results['actual'].max(), xgb_results['predicted'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax5.set_xlabel('Actual PM2.5 (μg/m³)', fontsize=11)
        ax5.set_ylabel('Predicted PM2.5 (μg/m³)', fontsize=11)
        ax5.set_title('XGBoost: Predicted vs Actual', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)

    # 6. Residuals distribution (XGBoost)
    ax6 = fig.add_subplot(gs[2, 1])

    if len(xgb_results) > 0:
        residuals = xgb_results['actual'] - xgb_results['predicted']
        ax6.hist(residuals, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax6.set_xlabel('Residual (μg/m³)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title(f'XGBoost Residuals (μ={residuals.mean():.2f}, σ={residuals.std():.2f})',
                     fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)

    # 7. Coverage analysis (intervalos de confianza)
    ax7 = fig.add_subplot(gs[2, 2])

    if quantile_results is not None and len(quantile_results) > 0:
        coverage = ((quantile_results['actual'] >= quantile_results['lower_bound']) &
                   (quantile_results['actual'] <= quantile_results['upper_bound'])).mean()

        labels = ['Inside\nInterval', 'Outside\nInterval']
        sizes = [coverage * 100, (1 - coverage) * 100]
        colors = ['#2ecc71', '#e74c3c']

        wedges, texts, autotexts = ax7.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax7.set_title('Confidence Interval\nCoverage', fontsize=11, fontweight='bold')

    plt.suptitle('Temporal Models Comparison: XGBoost vs ARIMA vs Prophet',
                fontsize=14, fontweight='bold', y=0.995)

    return fig, metrics_df


def main():
    """Ejecutar comparación completa de modelos temporales."""

    # 1. Cargar datos
    df_full = pd.read_csv(DATA_DIR / 'sinca_features_spatial.csv', parse_dates=['date'])
    df_full = df_full.rename(columns={'pm25_mean': 'pm25'})

    logging.info(f"\n✓ Datos cargados: {len(df_full):,} registros")

    # 2. Cargar resultados de XGBoost (ya calculados)
    xgb_results = load_data()

    logging.info("\n" + "="*70)
    logging.info("EVALUANDO XGBoost (Resultados Previos)")
    logging.info("="*70)
    xgb_metrics = calculate_metrics(xgb_results['pm25_real'].values,
                                     xgb_results['pm25_pred'].values)
    for metric, value in xgb_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # 3. ARIMA
    logging.info("\n" + "="*70)
    logging.info("ENTRENANDO Y EVALUANDO ARIMA")
    logging.info("="*70)

    try:
        arima_results = walk_forward_arima(df_full, horizon=1,
                                          train_size=365*3, step_size=30, n_iterations=50)

        if len(arima_results) > 0:
            arima_metrics = calculate_metrics(arima_results['actual'].values,
                                             arima_results['predicted'].values)
            logging.info("\nMétricas ARIMA:")
            for metric, value in arima_metrics.items():
                logging.info(f"  {metric}: {value:.4f}")

            arima_results.to_csv(DATA_DIR / 'arima_predictions_1d.csv', index=False)
        else:
            logging.warning("⚠ ARIMA no produjo resultados")
            arima_results = None
    except Exception as e:
        logging.error(f"❌ ARIMA falló: {str(e)}")
        arima_results = None

    # 4. Prophet
    logging.info("\n" + "="*70)
    logging.info("ENTRENANDO Y EVALUANDO PROPHET")
    logging.info("="*70)

    try:
        prophet_results = walk_forward_prophet(df_full, horizon=1,
                                              train_size=365*3, step_size=30, n_iterations=50)

        if len(prophet_results) > 0:
            prophet_metrics = calculate_metrics(prophet_results['actual'].values,
                                               prophet_results['predicted'].values)
            logging.info("\nMétricas Prophet:")
            for metric, value in prophet_metrics.items():
                logging.info(f"  {metric}: {value:.4f}")

            prophet_results.to_csv(DATA_DIR / 'prophet_predictions_1d.csv', index=False)
        else:
            logging.warning("⚠ Prophet no produjo resultados")
            prophet_results = None
    except Exception as e:
        logging.error(f"❌ Prophet falló: {str(e)}")
        prophet_results = None

    # 5. XGBoost con intervalos de confianza
    logging.info("\n" + "="*70)
    logging.info("XGBOOST CON INTERVALOS DE CONFIANZA")
    logging.info("="*70)

    # Definir features (sin lags para ser justo con ARIMA/Prophet)
    feature_cols = [
        'day_of_year', 'day_of_week', 'month',
        'distance_to_center_km', 'lat', 'lon', 'elevation',
        'era5_u_component_of_wind_10m', 'era5_total_precipitation_hourly',
        's5p_no2'
    ]

    # Verificar que features existen
    available_features = [f for f in feature_cols if f in df_full.columns]

    try:
        quantile_results = xgboost_quantile_regression(
            df_full, available_features, horizon=1,
            quantiles=[0.05, 0.5, 0.95],
            train_size=365*3, step_size=7, max_iterations=100
        )

        quantile_results.to_csv(DATA_DIR / 'xgboost_quantile_predictions_1d.csv', index=False)
    except Exception as e:
        logging.error(f"❌ XGBoost quantile falló: {str(e)}")
        quantile_results = None

    # 6. Generar visualizaciones
    logging.info("\n" + "="*70)
    logging.info("GENERANDO VISUALIZACIONES")
    logging.info("="*70)

    # Preparar resultados de XGBoost en formato correcto
    xgb_results_formatted = pd.DataFrame({
        'date': xgb_results['date'],
        'actual': xgb_results['pm25_real'],
        'predicted': xgb_results['pm25_pred']
    })

    fig, metrics_comparison = plot_models_comparison(
        xgb_results_formatted, arima_results, prophet_results, quantile_results
    )

    fig_path = REPORTS_DIR / 'temporal_models_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"\n✓ Gráfico guardado: {fig_path}")
    plt.close()

    # 7. Guardar resumen de métricas
    metrics_comparison.to_csv(DATA_DIR / 'temporal_models_metrics_comparison.csv', index=False)
    logging.info(f"✓ Métricas guardadas: {DATA_DIR / 'temporal_models_metrics_comparison.csv'}")

    # 8. Resumen ejecutivo
    logging.info("\n" + "="*70)
    logging.info("✓✓✓ COMPARACIÓN DE MODELOS COMPLETADA ✓✓✓")
    logging.info("="*70)

    logging.info("\n" + "="*70)
    logging.info("RESUMEN EJECUTIVO")
    logging.info("="*70)

    for _, row in metrics_comparison.iterrows():
        logging.info(f"\n{row['Model']}:")
        logging.info(f"  R²:   {row['R²']:.4f}")
        logging.info(f"  RMSE: {row['RMSE']:.2f} μg/m³")
        logging.info(f"  MAE:  {row['MAE']:.2f} μg/m³")
        logging.info(f"  MAPE: {row['MAPE']:.2f}%")

    # Determinar mejor modelo
    best_model = metrics_comparison.loc[metrics_comparison['R²'].idxmax()]
    logging.info(f"\n🏆 Mejor modelo: {best_model['Model']} (R² = {best_model['R²']:.4f})")

    logging.info("\n" + "="*70)


if __name__ == '__main__':
    main()
