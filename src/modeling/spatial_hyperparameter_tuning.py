#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Modelado Espacial de PM2.5.

Estrategia:
1. Grid Search / Random Search para cada modelo
2. Leave-One-Station-Out CV como métrica
3. Bayesian Optimization para modelos complejos
4. Análisis de sensibilidad de hiperparámetros
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LOSOGroupCV:
    """
    Custom CV que implementa Leave-One-Station-Out.
    Compatible con scikit-learn GridSearchCV.
    """

    def __init__(self, groups):
        self.groups = groups
        self.unique_groups = np.unique(groups)

    def split(self, X, y=None, groups=None):
        """Genera splits LOSO."""
        if groups is None:
            groups = self.groups

        for test_group in self.unique_groups:
            train_idx = np.where(groups != test_group)[0]
            test_idx = np.where(groups == test_group)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Número de splits = número de estaciones."""
        return len(self.unique_groups)


def tune_lasso(X, y, groups, n_iter=50):
    """
    Optimiza Lasso con Random Search.

    Args:
        X: Features
        y: Target
        groups: Estaciones (para LOSO-CV)
        n_iter: Iteraciones de búsqueda

    Returns:
        Mejor modelo y resultados
    """
    logger.info("\n" + "="*70)
    logger.info("LASSO - HYPERPARAMETER TUNING")
    logger.info("="*70)

    # Grid de hiperparámetros (exploración logarítmica)
    param_distributions = {
        'alpha': np.logspace(-4, 2, 100),  # 0.0001 a 100
        'max_iter': [5000, 10000],
        'tol': [1e-4, 1e-3]
    }

    # Custom scorer (queremos MAXIMIZAR R²)
    scorer = make_scorer(r2_score)

    # LOSO CV
    cv = LOSOGroupCV(groups)

    # Random Search
    logger.info(f"Buscando en {n_iter} combinaciones...")
    logger.info(f"Alpha range: {param_distributions['alpha'].min():.4f} a {param_distributions['alpha'].max():.4f}")

    random_search = RandomizedSearchCV(
        Lasso(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar
    start_time = time.time()
    random_search.fit(X_scaled, y, groups=groups)
    elapsed = time.time() - start_time

    logger.info(f"\n✓ Búsqueda completada en {elapsed:.1f}s")
    logger.info(f"\nMejor R²: {random_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  • {param}: {value}")

    # Análisis de resultados
    results_df = pd.DataFrame(random_search.cv_results_)

    return random_search.best_estimator_, random_search.best_params_, results_df


def tune_random_forest(X, y, groups, n_iter=30):
    """
    Optimiza Random Forest con Random Search.

    Args:
        X: Features
        y: Target
        groups: Estaciones
        n_iter: Iteraciones

    Returns:
        Mejor modelo y resultados
    """
    logger.info("\n" + "="*70)
    logger.info("RANDOM FOREST - HYPERPARAMETER TUNING")
    logger.info("="*70)

    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [5, 10, 20, 50],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5, 0.7]
    }

    scorer = make_scorer(r2_score)
    cv = LOSOGroupCV(groups)

    logger.info(f"Buscando en {n_iter} combinaciones...")

    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=1,  # RF ya usa n_jobs=-1
        verbose=1,
        random_state=42
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start_time = time.time()
    random_search.fit(X_scaled, y, groups=groups)
    elapsed = time.time() - start_time

    logger.info(f"\n✓ Búsqueda completada en {elapsed:.1f}s")
    logger.info(f"\nMejor R²: {random_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  • {param}: {value}")

    results_df = pd.DataFrame(random_search.cv_results_)

    return random_search.best_estimator_, random_search.best_params_, results_df


def tune_xgboost(X, y, groups, n_iter=50):
    """
    Optimiza XGBoost con Bayesian Optimization (si disponible) o Random Search.

    Args:
        X: Features
        y: Target
        groups: Estaciones
        n_iter: Iteraciones

    Returns:
        Mejor modelo y resultados
    """
    if not HAS_XGBOOST:
        logger.warning("XGBoost no disponible")
        return None, None, None

    logger.info("\n" + "="*70)
    logger.info("XGBOOST - HYPERPARAMETER TUNING")
    logger.info("="*70)

    scorer = make_scorer(r2_score)
    cv = LOSOGroupCV(groups)

    if HAS_SKOPT:
        # Bayesian Optimization
        logger.info("Usando Bayesian Optimization (skopt)...")

        search_spaces = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma': Real(0, 5),
            'reg_alpha': Real(0, 10),
            'reg_lambda': Real(0, 10)
        }

        bayes_search = BayesSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=1,
            verbose=1,
            random_state=42
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        start_time = time.time()
        bayes_search.fit(X_scaled, y, groups=groups)
        elapsed = time.time() - start_time

        logger.info(f"\n✓ Búsqueda completada en {elapsed:.1f}s")
        logger.info(f"\nMejor R²: {bayes_search.best_score_:.4f}")
        logger.info(f"Mejores parámetros:")
        for param, value in bayes_search.best_params_.items():
            logger.info(f"  • {param}: {value}")

        results_df = pd.DataFrame(bayes_search.cv_results_)

        return bayes_search.best_estimator_, bayes_search.best_params_, results_df

    else:
        # Fallback: Random Search
        logger.info("Usando Random Search (skopt no disponible)...")

        param_distributions = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': np.logspace(-2, -0.5, 20),
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.5, 1, 2, 5],
            'reg_alpha': [0, 0.1, 1, 5, 10],
            'reg_lambda': [0, 0.1, 1, 5, 10]
        }

        random_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=1,
            verbose=1,
            random_state=42
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        start_time = time.time()
        random_search.fit(X_scaled, y, groups=groups)
        elapsed = time.time() - start_time

        logger.info(f"\n✓ Búsqueda completada en {elapsed:.1f}s")
        logger.info(f"\nMejor R²: {random_search.best_score_:.4f}")
        logger.info(f"Mejores parámetros:")
        for param, value in random_search.best_params_.items():
            logger.info(f"  • {param}: {value}")

        results_df = pd.DataFrame(random_search.cv_results_)

        return random_search.best_estimator_, random_search.best_params_, results_df


def analyze_hyperparameter_sensitivity(results_df, model_name, output_dir):
    """
    Analiza sensibilidad de R² a cada hiperparámetro.

    Args:
        results_df: Resultados de GridSearchCV/RandomSearchCV
        model_name: Nombre del modelo
        output_dir: Directorio para guardar gráficos
    """
    logger.info(f"\nAnalizando sensibilidad de hiperparámetros ({model_name})...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraer columnas de parámetros
    param_cols = [col for col in results_df.columns if col.startswith('param_')]

    if len(param_cols) == 0:
        logger.warning("No hay columnas de parámetros")
        return

    # Plot: R² vs cada hiperparámetro
    n_params = len(param_cols)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, param_col in enumerate(param_cols):
        param_name = param_col.replace('param_', '')

        # Datos
        x = results_df[param_col]
        y = results_df['mean_test_score']

        # Si es numérico, scatter; si categórico, boxplot
        if pd.api.types.is_numeric_dtype(x):
            axes[i].scatter(x, y, alpha=0.6)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('R² (CV)')
            axes[i].set_title(f'Sensitivity: {param_name}')

            # Línea de tendencia
            if len(x.dropna()) > 3:
                z = np.polyfit(x.dropna(), y[x.notna()], 1)
                p = np.poly1d(z)
                axes[i].plot(x, p(x), "r--", alpha=0.8)
        else:
            # Boxplot para categóricos
            df_plot = pd.DataFrame({'param': x, 'r2': y})
            df_plot.boxplot(column='r2', by='param', ax=axes[i])
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('R² (CV)')
            axes[i].set_title(f'Sensitivity: {param_name}')
            plt.sca(axes[i])
            plt.xticks(rotation=45)

    # Ocultar ejes vacíos
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    output_file = output_dir / f'{model_name}_hyperparameter_sensitivity.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()


def compare_tuned_vs_default(X, y, groups, tuned_models, default_params):
    """
    Compara modelos optimizados vs parámetros default.

    Args:
        X: Features
        y: Target
        groups: Estaciones
        tuned_models: Dict de modelos optimizados
        default_params: Dict de parámetros default

    Returns:
        DataFrame con comparación
    """
    logger.info("\n" + "="*70)
    logger.info("COMPARACIÓN: TUNED vs DEFAULT")
    logger.info("="*70)

    from sklearn.model_selection import cross_val_score

    cv = LOSOGroupCV(groups)
    scorer = make_scorer(r2_score)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    for model_name, tuned_model in tuned_models.items():
        if tuned_model is None:
            continue

        # Modelo optimizado
        logger.info(f"\nEvaluando {model_name}...")

        tuned_scores = cross_val_score(
            tuned_model, X_scaled, y,
            cv=cv, scoring=scorer, groups=groups
        )

        tuned_r2_mean = tuned_scores.mean()
        tuned_r2_std = tuned_scores.std()

        logger.info(f"  Tuned:   R² = {tuned_r2_mean:.4f} ± {tuned_r2_std:.4f}")

        # Modelo default
        if model_name in default_params:
            default_model = default_params[model_name]

            default_scores = cross_val_score(
                default_model, X_scaled, y,
                cv=cv, scoring=scorer, groups=groups
            )

            default_r2_mean = default_scores.mean()
            default_r2_std = default_scores.std()

            logger.info(f"  Default: R² = {default_r2_mean:.4f} ± {default_r2_std:.4f}")

            improvement = tuned_r2_mean - default_r2_mean
            logger.info(f"  Mejora:  {improvement:+.4f} ({improvement/abs(default_r2_mean)*100:+.1f}%)")

            results.append({
                'model': model_name,
                'tuned_r2_mean': tuned_r2_mean,
                'tuned_r2_std': tuned_r2_std,
                'default_r2_mean': default_r2_mean,
                'default_r2_std': default_r2_std,
                'improvement': improvement,
                'improvement_pct': improvement / abs(default_r2_mean) * 100
            })

    return pd.DataFrame(results)


def main():
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER TUNING - MODELADO ESPACIAL PM2.5")
    logger.info("="*70)

    # Cargar dataset
    input_file = Path('data/processed/sinca_features_spatial.csv')
    logger.info(f"\nCargando: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")

    # Features
    exclude_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'archivo', 'validado', 'pm25'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['pm25'].values
    groups = df['estacion'].values

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Estaciones: {len(np.unique(groups))}")

    # 1. Tune Lasso
    lasso_tuned, lasso_params, lasso_results = tune_lasso(X, y, groups, n_iter=50)

    # 2. Tune Random Forest
    rf_tuned, rf_params, rf_results = tune_random_forest(X, y, groups, n_iter=30)

    # 3. Tune XGBoost
    if HAS_XGBOOST:
        xgb_tuned, xgb_params, xgb_results = tune_xgboost(X, y, groups, n_iter=50)
    else:
        xgb_tuned, xgb_params, xgb_results = None, None, None

    # 4. Análisis de sensibilidad
    output_dir = Path('reports/figures')

    if lasso_results is not None:
        analyze_hyperparameter_sensitivity(lasso_results, 'Lasso', output_dir)

    if rf_results is not None:
        analyze_hyperparameter_sensitivity(rf_results, 'RandomForest', output_dir)

    if xgb_results is not None:
        analyze_hyperparameter_sensitivity(xgb_results, 'XGBoost', output_dir)

    # 5. Comparación tuned vs default
    tuned_models = {
        'Lasso': lasso_tuned,
        'Random Forest': rf_tuned
    }

    if xgb_tuned is not None:
        tuned_models['XGBoost'] = xgb_tuned

    default_models = {
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
    }

    if HAS_XGBOOST:
        default_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42, n_jobs=-1
        )

    comparison_df = compare_tuned_vs_default(X, y, groups, tuned_models, default_models)

    # 6. Guardar resultados
    logger.info("\n" + "="*70)
    logger.info("GUARDANDO RESULTADOS")
    logger.info("="*70)

    # Mejores parámetros
    best_params_file = Path('data/processed/spatial_best_hyperparameters.txt')
    with open(best_params_file, 'w') as f:
        f.write("MEJORES HIPERPARÁMETROS - MODELADO ESPACIAL PM2.5\n")
        f.write("="*70 + "\n\n")

        if lasso_params:
            f.write("LASSO:\n")
            for param, value in lasso_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

        if rf_params:
            f.write("RANDOM FOREST:\n")
            for param, value in rf_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

        if xgb_params:
            f.write("XGBOOST:\n")
            for param, value in xgb_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

    logger.info(f"✓ Parámetros guardados: {best_params_file}")

    # Comparación
    comparison_file = Path('data/processed/spatial_tuning_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"✓ Comparación guardada: {comparison_file}")

    # Resumen
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE MEJORAS")
    logger.info("="*70)

    print("\n", comparison_df.to_string(index=False))

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ HYPERPARAMETER TUNING COMPLETADO ✓✓✓")
    logger.info("="*70)

    return tuned_models, comparison_df


if __name__ == "__main__":
    tuned_models, comparison = main()
