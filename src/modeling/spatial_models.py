#!/usr/bin/env python3
"""
Modelado Espacial de PM2.5 - Predicción en nuevas ubicaciones sin historial.

Estrategia:
1. Leave-One-Station-Out Cross-Validation (evaluar generalización espacial)
2. Múltiples modelos:
   - Baseline: Linear Regression
   - Ridge/Lasso (regularización)
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Gaussian Process Regression (geoestadística)
3. Ensamble de modelos
4. Análisis de feature importance real
5. Predicciones espaciales (mapa continuo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    HAS_GPR = True
except ImportError:
    HAS_GPR = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpatialPM25Model:
    """
    Clase para modelado espacial de PM2.5.
    """

    def __init__(self, model_type='ridge', model_params=None):
        """
        Inicializa el modelo espacial.

        Args:
            model_type: Tipo de modelo ('linear', 'ridge', 'lasso', 'rf',
                       'gb', 'xgb', 'lgb', 'gpr')
            model_params: Parámetros del modelo (dict)
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        self._init_model()

    def _init_model(self):
        """Inicializa el modelo según el tipo."""
        if self.model_type == 'linear':
            self.model = LinearRegression(**self.model_params)

        elif self.model_type == 'ridge':
            default_params = {'alpha': 1.0, 'random_state': 42}
            default_params.update(self.model_params)
            self.model = Ridge(**default_params)

        elif self.model_type == 'lasso':
            default_params = {'alpha': 1.0, 'random_state': 42}
            default_params.update(self.model_params)
            self.model = Lasso(**default_params)

        elif self.model_type == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = RandomForestRegressor(**default_params)

        elif self.model_type == 'gb':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
            default_params.update(self.model_params)
            self.model = GradientBoostingRegressor(**default_params)

        elif self.model_type == 'xgb':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost no disponible. Instalar: pip install xgboost")
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBRegressor(**default_params)

        elif self.model_type == 'lgb':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM no disponible. Instalar: pip install lightgbm")
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            default_params.update(self.model_params)
            self.model = lgb.LGBMRegressor(**default_params)

        elif self.model_type == 'gpr':
            if not HAS_GPR:
                raise ImportError("Gaussian Process no disponible")
            # Kernel: RBF (espacial) + WhiteKernel (ruido)
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            default_params = {
                'kernel': kernel,
                'n_restarts_optimizer': 10,
                'random_state': 42
            }
            default_params.update(self.model_params)
            self.model = GaussianProcessRegressor(**default_params)

        else:
            raise ValueError(f"Modelo desconocido: {self.model_type}")

    def fit(self, X, y):
        """Entrena el modelo."""
        self.feature_names = X.columns.tolist()

        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar
        self.model.fit(X_scaled, y)

        return self

    def predict(self, X):
        """Predice valores."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        """Obtiene importancia de features (si el modelo lo soporta)."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        elif hasattr(self.model, 'coef_'):
            # Para modelos lineales, usar coeficientes absolutos
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)

        else:
            return None


def evaluate_loso_cv(df, feature_cols, target='pm25', models_dict=None):
    """
    Evalúa múltiples modelos usando Leave-One-Station-Out CV.

    Args:
        df: DataFrame
        feature_cols: Features a usar
        target: Variable target
        models_dict: Dict de modelos {nombre: tipo}

    Returns:
        DataFrame con resultados por modelo y estación
    """
    logger.info("\n" + "="*70)
    logger.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
    logger.info("="*70)

    if models_dict is None:
        models_dict = {
            'Linear': 'linear',
            'Ridge': 'ridge',
            'Lasso': 'lasso',
            'Random Forest': 'rf',
            'Gradient Boosting': 'gb'
        }

        if HAS_XGBOOST:
            models_dict['XGBoost'] = 'xgb'
        if HAS_LIGHTGBM:
            models_dict['LightGBM'] = 'lgb'

    X = df[feature_cols].copy()
    y = df[target]
    stations = df['estacion'].unique()

    all_results = []

    logger.info(f"\nModelos a evaluar: {list(models_dict.keys())}")
    logger.info(f"Estaciones: {len(stations)}")

    for model_name, model_type in models_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"MODELO: {model_name}")
        logger.info(f"{'='*60}")

        station_results = []

        for i, test_station in enumerate(stations, 1):
            # Split
            train_mask = df['estacion'] != test_station
            test_mask = df['estacion'] == test_station

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Entrenar
            try:
                model = SpatialPM25Model(model_type=model_type)
                model.fit(X_train, y_train)

                # Predecir
                y_pred = model.predict(X_test)

                # Métricas
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                station_results.append({
                    'model': model_name,
                    'station': test_station,
                    'n_test': len(y_test),
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                })

                logger.info(f"  [{i}/{len(stations)}] {test_station:20s} "
                           f"R²={r2:6.3f}  RMSE={rmse:6.2f}  MAE={mae:6.2f}")

            except Exception as e:
                logger.error(f"  Error en {test_station}: {e}")
                station_results.append({
                    'model': model_name,
                    'station': test_station,
                    'n_test': len(y_test),
                    'r2': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan
                })

        # Promedio ponderado
        results_df = pd.DataFrame(station_results)
        valid_results = results_df.dropna()

        if len(valid_results) > 0:
            total_samples = valid_results['n_test'].sum()
            avg_r2 = (valid_results['r2'] * valid_results['n_test']).sum() / total_samples
            avg_rmse = (valid_results['rmse'] * valid_results['n_test']).sum() / total_samples
            avg_mae = (valid_results['mae'] * valid_results['n_test']).sum() / total_samples

            logger.info(f"\n  PROMEDIO {model_name}:")
            logger.info(f"    R² = {avg_r2:.3f}")
            logger.info(f"    RMSE = {avg_rmse:.2f} μg/m³")
            logger.info(f"    MAE = {avg_mae:.2f} μg/m³")

        all_results.extend(station_results)

    return pd.DataFrame(all_results)


def train_final_models(df, feature_cols, target='pm25', models_dict=None):
    """
    Entrena modelos finales usando TODOS los datos.

    Args:
        df: DataFrame completo
        feature_cols: Features
        target: Target
        models_dict: Modelos a entrenar

    Returns:
        Dict de modelos entrenados
    """
    logger.info("\n" + "="*70)
    logger.info("ENTRENAMIENTO DE MODELOS FINALES (todos los datos)")
    logger.info("="*70)

    if models_dict is None:
        models_dict = {
            'Ridge': 'ridge',
            'Random Forest': 'rf',
            'Gradient Boosting': 'gb'
        }

    X = df[feature_cols].copy()
    y = df[target]

    trained_models = {}

    for model_name, model_type in models_dict.items():
        logger.info(f"\nEntrenando {model_name}...")

        model = SpatialPM25Model(model_type=model_type)
        model.fit(X, y)

        trained_models[model_name] = model

        # Feature importance
        importance = model.get_feature_importance()
        if importance is not None:
            logger.info(f"  Top 5 features:")
            for _, row in importance.head(5).iterrows():
                logger.info(f"    {row['feature']:40s} {row['importance']:.4f}")

    return trained_models


def create_visualizations(results_df, output_dir):
    """
    Crea visualizaciones de resultados.

    Args:
        results_df: DataFrame con resultados LOSO-CV
        output_dir: Directorio para guardar figuras
    """
    logger.info("\n" + "="*60)
    logger.info("CREANDO VISUALIZACIONES")
    logger.info("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. R² por modelo (boxplot)
    plt.figure(figsize=(12, 6))

    # Ordenar modelos por mediana de R²
    model_order = results_df.groupby('model')['r2'].median().sort_values(ascending=False).index

    sns.boxplot(data=results_df, x='model', y='r2', order=model_order)
    plt.axhline(y=0, color='red', linestyle='--', label='R²=0 (baseline)')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Modelo')
    plt.ylabel('R² Score')
    plt.title('Spatial Generalization Performance - R² by Model (LOSO-CV)')
    plt.legend()
    plt.tight_layout()

    output_file = output_dir / 'spatial_models_r2_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 2. RMSE por modelo
    plt.figure(figsize=(12, 6))

    model_order = results_df.groupby('model')['rmse'].median().sort_values().index

    sns.boxplot(data=results_df, x='model', y='rmse', order=model_order)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Modelo')
    plt.ylabel('RMSE (μg/m³)')
    plt.title('Spatial Generalization Performance - RMSE by Model (LOSO-CV)')
    plt.tight_layout()

    output_file = output_dir / 'spatial_models_rmse_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 3. Heatmap: R² por modelo × estación
    plt.figure(figsize=(14, 8))

    pivot = results_df.pivot(index='station', columns='model', values='r2')

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R²'}, linewidths=0.5)
    plt.title('R² Score by Model and Station (LOSO-CV)')
    plt.xlabel('Model')
    plt.ylabel('Test Station (held out)')
    plt.tight_layout()

    output_file = output_dir / 'spatial_models_r2_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 4. Scatter: Predicted vs Actual (mejor modelo)
    best_model = results_df.groupby('model')['r2'].mean().idxmax()

    logger.info(f"\n  Mejor modelo (promedio R²): {best_model}")


def main():
    logger.info("\n" + "="*70)
    logger.info("MODELADO ESPACIAL - PM2.5 SANTIAGO")
    logger.info("="*70)

    # Cargar dataset espacial
    input_file = Path('data/processed/sinca_features_spatial.csv')
    logger.info(f"\nCargando dataset: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")

    # Features espaciales
    exclude_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'archivo', 'validado',
        'pm25'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"\nFeatures espaciales: {len(feature_cols)}")
    for feat in feature_cols:
        logger.info(f"  • {feat}")

    # 1. LOSO-CV con múltiples modelos
    results_df = evaluate_loso_cv(df, feature_cols, target='pm25')

    # 2. Resumen de resultados
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("="*70)

    summary = results_df.groupby('model').agg({
        'r2': ['mean', 'std', 'min', 'max'],
        'rmse': ['mean', 'std', 'min', 'max'],
        'mae': ['mean', 'std', 'min', 'max']
    }).round(3)

    logger.info("\n" + str(summary))

    # Mejor modelo
    best_model_r2 = results_df.groupby('model')['r2'].mean().idxmax()
    best_r2 = results_df.groupby('model')['r2'].mean().max()

    logger.info(f"\n🏆 MEJOR MODELO: {best_model_r2}")
    logger.info(f"   R² promedio: {best_r2:.3f}")

    # 3. Visualizaciones
    create_visualizations(results_df, output_dir=Path('reports/figures'))

    # 4. Guardar resultados
    output_file = Path('data/processed/spatial_models_results.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Resultados guardados: {output_file}")

    summary_file = Path('data/processed/spatial_models_summary.csv')
    summary.to_csv(summary_file)
    logger.info(f"✓ Resumen guardado: {summary_file}")

    # 5. Entrenar modelos finales (top 3)
    top_models = results_df.groupby('model')['r2'].mean().nlargest(3).index.tolist()
    models_dict = {name: results_df[results_df['model'] == name].iloc[0]['model'].lower().replace(' ', '_')
                   for name in top_models}

    # Mapeo manual para modelos conocidos
    model_type_map = {
        'Linear': 'linear',
        'Ridge': 'ridge',
        'Lasso': 'lasso',
        'Random Forest': 'rf',
        'Gradient Boosting': 'gb',
        'XGBoost': 'xgb',
        'LightGBM': 'lgb'
    }

    final_models_dict = {name: model_type_map.get(name, 'ridge') for name in top_models}

    logger.info(f"\nEntrenando modelos finales (top 3): {list(final_models_dict.keys())}")
    trained_models = train_final_models(df, feature_cols, models_dict=final_models_dict)

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ MODELADO ESPACIAL COMPLETADO ✓✓✓")
    logger.info("="*70)

    logger.info(f"\n📊 PERFORMANCE FINAL:")
    logger.info(f"   Mejor modelo: {best_model_r2}")
    logger.info(f"   R² promedio: {best_r2:.3f}")

    best_model_results = results_df[results_df['model'] == best_model_r2]
    best_rmse = best_model_results['rmse'].mean()
    best_mae = best_model_results['mae'].mean()

    logger.info(f"   RMSE: {best_rmse:.2f} μg/m³")
    logger.info(f"   MAE: {best_mae:.2f} μg/m³")

    logger.info(f"\n📁 ARCHIVOS GENERADOS:")
    logger.info(f"   • spatial_models_results.csv (resultados detallados)")
    logger.info(f"   • spatial_models_summary.csv (resumen por modelo)")
    logger.info(f"   • spatial_models_r2_comparison.png")
    logger.info(f"   • spatial_models_rmse_comparison.png")
    logger.info(f"   • spatial_models_r2_heatmap.png")

    return results_df, trained_models


if __name__ == "__main__":
    results, models = main()
