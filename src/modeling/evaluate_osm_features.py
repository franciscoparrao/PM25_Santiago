#!/usr/bin/env python3
"""
Evalúa mejora de modelos espaciales con features OSM.

Compara:
- Baseline (sin OSM): 13 features originales
- Enhanced (con OSM): 18 features (13 originales + 5 OSM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def loso_cv_evaluate(X, y, groups, model, model_name):
    """
    Leave-One-Station-Out Cross-Validation.

    Args:
        X: Features
        y: Target
        groups: Estaciones
        model: Modelo sklearn
        model_name: Nombre del modelo

    Returns:
        Dict con resultados
    """
    unique_stations = np.unique(groups)
    results = []

    scaler = StandardScaler()

    for station in unique_stations:
        # Split
        train_mask = groups != station
        test_mask = groups == station

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Scale
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'model': model_name,
            'station': station,
            'n_test': len(y_test),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })

    results_df = pd.DataFrame(results)

    # Promedio ponderado
    total_samples = results_df['n_test'].sum()
    avg_r2 = (results_df['r2'] * results_df['n_test']).sum() / total_samples
    avg_rmse = (results_df['rmse'] * results_df['n_test']).sum() / total_samples
    avg_mae = (results_df['mae'] * results_df['n_test']).sum() / total_samples

    return {
        'results_df': results_df,
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae
    }


def main():
    logger.info("\n" + "="*70)
    logger.info("EVALUACIÓN: MODELOS CON FEATURES OSM")
    logger.info("="*70)

    # Cargar datasets
    baseline_file = Path('data/processed/sinca_features_spatial.csv')
    enhanced_file = Path('data/processed/sinca_features_spatial_enhanced.csv')

    logger.info(f"\nBaseline: {baseline_file}")
    logger.info(f"Enhanced: {enhanced_file}")

    df_baseline = pd.read_csv(baseline_file, parse_dates=['datetime', 'date'])
    df_enhanced = pd.read_csv(enhanced_file, parse_dates=['datetime', 'date'])

    logger.info(f"\nBaseline features: {len(df_baseline.columns)}")
    logger.info(f"Enhanced features: {len(df_enhanced.columns)}")

    # Features
    exclude_cols = ['datetime', 'date', 'year', 'month', 'day',
                    'estacion', 'archivo', 'validado', 'pm25']

    baseline_features = [col for col in df_baseline.columns if col not in exclude_cols]
    enhanced_features = [col for col in df_enhanced.columns if col not in exclude_cols]

    osm_features = [col for col in enhanced_features if col not in baseline_features]

    logger.info(f"\nBaseline features: {len(baseline_features)}")
    logger.info(f"Enhanced features: {len(enhanced_features)}")
    logger.info(f"New OSM features: {len(osm_features)}")

    logger.info(f"\nOSM features agregadas:")
    for feat in osm_features:
        logger.info(f"  • {feat}")

    # Preparar datos
    X_baseline = df_baseline[baseline_features].values
    X_enhanced = df_enhanced[enhanced_features].values
    y = df_enhanced['pm25'].values
    groups = df_enhanced['estacion'].values

    logger.info(f"\nSamples: {len(y):,}")
    logger.info(f"Stations: {len(np.unique(groups))}")

    # Modelos a evaluar
    models = {
        'Lasso': Lasso(alpha=1.0, max_iter=5000, random_state=42)
    }

    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    # Evaluar cada modelo
    all_results = []

    for model_name, model in models.items():
        logger.info("\n" + "="*70)
        logger.info(f"MODELO: {model_name}")
        logger.info("="*70)

        # Baseline
        logger.info(f"\nEvaluando BASELINE ({len(baseline_features)} features)...")
        baseline_results = loso_cv_evaluate(
            X_baseline, y, groups,
            model, f"{model_name}_Baseline"
        )

        logger.info(f"  Baseline R²: {baseline_results['avg_r2']:.4f}")
        logger.info(f"  Baseline RMSE: {baseline_results['avg_rmse']:.2f}")
        logger.info(f"  Baseline MAE: {baseline_results['avg_mae']:.2f}")

        # Enhanced
        logger.info(f"\nEvaluando ENHANCED ({len(enhanced_features)} features)...")
        enhanced_results = loso_cv_evaluate(
            X_enhanced, y, groups,
            model, f"{model_name}_Enhanced"
        )

        logger.info(f"  Enhanced R²: {enhanced_results['avg_r2']:.4f}")
        logger.info(f"  Enhanced RMSE: {enhanced_results['avg_rmse']:.2f}")
        logger.info(f"  Enhanced MAE: {enhanced_results['avg_mae']:.2f}")

        # Mejora
        improvement_r2 = enhanced_results['avg_r2'] - baseline_results['avg_r2']
        improvement_rmse = baseline_results['avg_rmse'] - enhanced_results['avg_rmse']  # Menor es mejor
        improvement_mae = baseline_results['avg_mae'] - enhanced_results['avg_mae']

        logger.info("\n" + "-"*70)
        logger.info(f"MEJORA con OSM:")
        logger.info(f"  ΔR²:   {improvement_r2:+.4f} ({improvement_r2/abs(baseline_results['avg_r2'])*100:+.1f}%)")
        logger.info(f"  ΔRMSE: {improvement_rmse:+.2f} μg/m³")
        logger.info(f"  ΔMAE:  {improvement_mae:+.2f} μg/m³")

        # Guardar resultados
        all_results.append({
            'model': model_name,
            'version': 'Baseline',
            'n_features': len(baseline_features),
            'r2': baseline_results['avg_r2'],
            'rmse': baseline_results['avg_rmse'],
            'mae': baseline_results['avg_mae']
        })

        all_results.append({
            'model': model_name,
            'version': 'Enhanced',
            'n_features': len(enhanced_features),
            'r2': enhanced_results['avg_r2'],
            'rmse': enhanced_results['avg_rmse'],
            'mae': enhanced_results['avg_mae']
        })

        # Resultados por estación
        combined_station_results = pd.concat([
            baseline_results['results_df'],
            enhanced_results['results_df']
        ])

        # Guardar resultados detallados
        station_results_file = Path(f'data/processed/{model_name.lower()}_station_results_osm.csv')
        combined_station_results.to_csv(station_results_file, index=False)
        logger.info(f"\n✓ Resultados por estación guardados: {station_results_file}")

    # Resumen final
    logger.info("\n" + "="*70)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*70)

    summary_df = pd.DataFrame(all_results)
    print("\n", summary_df.to_string(index=False))

    # Guardar resumen
    summary_file = Path('data/processed/osm_features_evaluation.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\n✓ Resumen guardado: {summary_file}")

    # Análisis de mejora
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS DE MEJORA")
    logger.info("="*70)

    for model_name in models.keys():
        baseline_row = summary_df[(summary_df['model'] == model_name) &
                                   (summary_df['version'] == 'Baseline')].iloc[0]
        enhanced_row = summary_df[(summary_df['model'] == model_name) &
                                   (summary_df['version'] == 'Enhanced')].iloc[0]

        logger.info(f"\n{model_name}:")
        logger.info(f"  Baseline R²: {baseline_row['r2']:.4f}")
        logger.info(f"  Enhanced R²: {enhanced_row['r2']:.4f}")

        improvement = enhanced_row['r2'] - baseline_row['r2']
        logger.info(f"  Mejora absoluta: {improvement:+.4f}")

        if baseline_row['r2'] != 0:
            pct_improvement = improvement / abs(baseline_row['r2']) * 100
            logger.info(f"  Mejora relativa: {pct_improvement:+.1f}%")

        # ¿Alcanzamos R² > 0?
        if enhanced_row['r2'] > 0:
            logger.info(f"  🎯 ¡R² > 0 alcanzado! (modelo mejor que baseline)")
        else:
            r2_to_zero = 0 - enhanced_row['r2']
            logger.info(f"  📊 Falta {r2_to_zero:.4f} para alcanzar R² = 0")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ EVALUACIÓN COMPLETADA ✓✓✓")
    logger.info("="*70)

    return summary_df


if __name__ == "__main__":
    summary = main()
