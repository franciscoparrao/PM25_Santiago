#!/usr/bin/env python3
"""
Regression Kriging para mapeo espacial completo de PM2.5.

Combina:
1. Modelo ML (XGBoost) para predecir tendencia usando features espaciales
2. Ordinary Kriging para interpolar residuales
3. Predicción final = Tendencia ML + Residuales Kriging

Genera mapas de alta resolución para toda la Región Metropolitana.
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

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False
    raise ImportError("PyKrige requerido: pip install --user --break-system-packages pykrige")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegressionKriging:
    """
    Regression Kriging: ML + Geostatistics.

    Workflow:
    1. Entrenar modelo ML para predecir tendencia (drift)
    2. Calcular residuales = observado - predicho
    3. Kriging de residuales (captura estructura espacial)
    4. Predicción final = ML + Kriging_residuales
    """

    def __init__(self, ml_model=None, variogram_model='spherical', verbose=True):
        """
        Args:
            ml_model: Modelo sklearn (default: XGBoost si disponible, sino RF)
            variogram_model: Modelo de variograma ('spherical', 'exponential', 'gaussian', 'linear')
            verbose: Logging detallado
        """
        self.verbose = verbose
        self.variogram_model = variogram_model

        # Modelo ML para tendencia
        if ml_model is None:
            if HAS_XGBOOST:
                self.ml_model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                if self.verbose:
                    logger.info("Usando XGBoost para tendencia ML")
            else:
                self.ml_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                if self.verbose:
                    logger.info("Usando Random Forest para tendencia ML")
        else:
            self.ml_model = ml_model

        self.scaler = StandardScaler()
        self.kriging_model = None
        self.residuals = None

    def fit(self, X, y, coords):
        """
        Entrenar Regression Kriging.

        Args:
            X: Features (n_samples, n_features)
            y: Target PM2.5 (n_samples,)
            coords: Coordenadas (n_samples, 2) - [lon, lat]
        """
        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("ENTRENANDO REGRESSION KRIGING")
            logger.info("="*70)

        # 1. Entrenar modelo ML
        if self.verbose:
            logger.info("\n[1/3] Entrenando modelo ML para tendencia...")

        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)

        # Predicciones ML
        y_pred_ml = self.ml_model.predict(X_scaled)

        r2_ml = r2_score(y, y_pred_ml)
        rmse_ml = np.sqrt(mean_squared_error(y, y_pred_ml))

        if self.verbose:
            logger.info(f"  ML R²: {r2_ml:.4f}")
            logger.info(f"  ML RMSE: {rmse_ml:.2f} μg/m³")

        # 2. Calcular residuales
        if self.verbose:
            logger.info("\n[2/3] Calculando residuales...")

        self.residuals = y - y_pred_ml

        if self.verbose:
            logger.info(f"  Residuales media: {self.residuals.mean():.4f}")
            logger.info(f"  Residuales std: {self.residuals.std():.2f}")
            logger.info(f"  Residuales min/max: [{self.residuals.min():.2f}, {self.residuals.max():.2f}]")

        # 3. Kriging de residuales
        if self.verbose:
            logger.info(f"\n[3/3] Kriging de residuales (modelo: {self.variogram_model})...")

        try:
            self.kriging_model = OrdinaryKriging(
                x=coords[:, 0],  # lon
                y=coords[:, 1],  # lat
                z=self.residuals,
                variogram_model=self.variogram_model,
                verbose=False,
                enable_plotting=False,
                nlags=min(6, len(coords) - 1)  # Nlags menor que n_samples
            )

            if self.verbose:
                logger.info("  ✓ Kriging model fitted")
                logger.info(f"  Variogram sill: {self.kriging_model.variogram_model_parameters[0]:.4f}")
                logger.info(f"  Variogram range: {self.kriging_model.variogram_model_parameters[1]:.4f}")
                logger.info(f"  Variogram nugget: {self.kriging_model.variogram_model_parameters[2]:.4f}")

        except Exception as e:
            logger.warning(f"  Kriging falló: {e}")
            logger.warning("  Continuando sin kriging de residuales (solo ML)")
            self.kriging_model = None

        if self.verbose:
            logger.info("\n" + "="*70)
            logger.info("✓ REGRESSION KRIGING ENTRENADO")
            logger.info("="*70)

        return self

    def predict(self, X, coords):
        """
        Predecir PM2.5 usando Regression Kriging.

        Args:
            X: Features (n_samples, n_features)
            coords: Coordenadas (n_samples, 2) - [lon, lat]

        Returns:
            predictions: PM2.5 predicho (n_samples,)
            variance: Varianza de predicción (n_samples,) - solo si kriging disponible
        """
        # 1. Predicción ML (tendencia)
        X_scaled = self.scaler.transform(X)
        y_pred_ml = self.ml_model.predict(X_scaled)

        # 2. Kriging de residuales (si disponible)
        if self.kriging_model is not None:
            try:
                z_pred_residuals, ss_pred = self.kriging_model.execute(
                    style='points',
                    xpoints=coords[:, 0],
                    ypoints=coords[:, 1]
                )

                # Predicción final = ML + Kriging_residuales
                predictions = y_pred_ml + z_pred_residuals
                variance = ss_pred

            except Exception as e:
                logger.warning(f"Kriging prediction falló: {e}")
                logger.warning("Usando solo predicción ML")
                predictions = y_pred_ml
                variance = None
        else:
            predictions = y_pred_ml
            variance = None

        return predictions, variance

    def predict_grid(self, grid_lon, grid_lat, features_grid):
        """
        Predecir PM2.5 en grilla regular.

        Args:
            grid_lon: Longitudes de grilla (n_points,)
            grid_lat: Latitudes de grilla (n_points,)
            features_grid: Features para cada punto (n_points, n_features)

        Returns:
            predictions: PM2.5 en grilla (n_points,)
            variance: Varianza (n_points,) o None
        """
        coords = np.column_stack([grid_lon, grid_lat])
        return self.predict(features_grid, coords)


def aggregate_to_daily_station_means(df):
    """
    Agregar datos a medias diarias por estación.

    Para Regression Kriging necesitamos un valor por estación-día
    (no múltiples observaciones por día).
    """
    logger.info("\nAgregando a medias diarias por estación...")

    # Seleccionar solo columnas numéricas para agregación
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas de identificación
    exclude_cols = ['year', 'month', 'day']
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Agrupar por estación y fecha
    df_daily = df.groupby(['estacion', 'date'])[agg_cols].mean().reset_index()

    # Agregar año, mes, día
    df_daily['year'] = df_daily['date'].dt.year
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day'] = df_daily['date'].dt.day

    logger.info(f"  Registros originales: {len(df):,}")
    logger.info(f"  Registros agregados: {len(df_daily):,}")
    logger.info(f"  Estaciones: {df_daily['estacion'].nunique()}")

    return df_daily


def evaluate_regression_kriging(df, feature_cols, exclude_temporal=True):
    """
    Evaluar Regression Kriging con validación LOSO-CV.

    Args:
        df: DataFrame con features y PM2.5
        feature_cols: Lista de columnas de features
        exclude_temporal: Si True, excluir features temporales (año, mes, día, hora)
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUACIÓN: REGRESSION KRIGING con LOSO-CV")
    logger.info("="*70)

    # Filtrar features (excluir temporales si es espacial)
    if exclude_temporal:
        temporal_features = ['year', 'month', 'day', 'hour']
        feature_cols = [col for col in feature_cols if col not in temporal_features]
        logger.info(f"\nExcluyendo features temporales para predicción espacial")

    logger.info(f"\nFeatures utilizadas: {len(feature_cols)}")
    for feat in feature_cols:
        logger.info(f"  • {feat}")

    # Preparar datos
    X = df[feature_cols].values
    y = df['pm25'].values
    coords = df[['lon', 'lat']].values
    groups = df['estacion'].values

    unique_stations = np.unique(groups)

    logger.info(f"\nDatos:")
    logger.info(f"  Muestras: {len(y):,}")
    logger.info(f"  Estaciones: {len(unique_stations)}")
    logger.info(f"  Features: {X.shape[1]}")

    # LOSO-CV
    logger.info("\n" + "="*70)
    logger.info("LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
    logger.info("="*70)

    results = []

    for i, test_station in enumerate(unique_stations):
        logger.info(f"\n[{i+1}/{len(unique_stations)}] Test: {test_station}")
        logger.info("-" * 70)

        # Split
        train_mask = groups != test_station
        test_mask = groups == test_station

        X_train = X[train_mask]
        y_train = y[train_mask]
        coords_train = coords[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]
        coords_test = coords[test_mask]

        logger.info(f"  Train samples: {len(y_train):,}")
        logger.info(f"  Test samples: {len(y_test):,}")

        # Train Regression Kriging
        rk = RegressionKriging(variogram_model='spherical', verbose=False)

        try:
            rk.fit(X_train, y_train, coords_train)

            # Predict
            y_pred, variance = rk.predict(X_test, coords_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  RMSE: {rmse:.2f} μg/m³")
            logger.info(f"  MAE: {mae:.2f} μg/m³")

            if variance is not None:
                logger.info(f"  Varianza media: {variance.mean():.2f}")

            results.append({
                'station': test_station,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'has_kriging': rk.kriging_model is not None
            })

        except Exception as e:
            logger.error(f"  Error en {test_station}: {e}")
            results.append({
                'station': test_station,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'r2': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'has_kriging': False
            })

    # Resumen
    results_df = pd.DataFrame(results)

    logger.info("\n" + "="*70)
    logger.info("RESULTADOS POR ESTACIÓN")
    logger.info("="*70)
    print("\n", results_df.to_string(index=False))

    # Promedios ponderados
    valid_results = results_df[~results_df['r2'].isna()]

    if len(valid_results) > 0:
        total_samples = valid_results['n_test'].sum()
        avg_r2 = (valid_results['r2'] * valid_results['n_test']).sum() / total_samples
        avg_rmse = (valid_results['rmse'] * valid_results['n_test']).sum() / total_samples
        avg_mae = (valid_results['mae'] * valid_results['n_test']).sum() / total_samples

        logger.info("\n" + "="*70)
        logger.info("MÉTRICAS PROMEDIO (LOSO-CV)")
        logger.info("="*70)
        logger.info(f"  R² promedio: {avg_r2:.4f}")
        logger.info(f"  RMSE promedio: {avg_rmse:.2f} μg/m³")
        logger.info(f"  MAE promedio: {avg_mae:.2f} μg/m³")
        logger.info(f"  Estaciones con R² > 0: {(valid_results['r2'] > 0).sum()}/{len(valid_results)}")
        logger.info(f"  Kriging exitoso: {valid_results['has_kriging'].sum()}/{len(valid_results)} estaciones")

    return results_df


def main():
    logger.info("\n" + "="*70)
    logger.info("REGRESSION KRIGING PARA MAPEO ESPACIAL DE PM2.5")
    logger.info("="*70)

    # Cargar dataset
    input_file = Path('data/processed/sinca_features_spatial.csv')

    logger.info(f"\nCargando dataset: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])

    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Columnas: {len(df.columns)}")
    logger.info(f"  Estaciones: {df['estacion'].nunique()}")

    # Agregar a medias diarias
    df_daily = aggregate_to_daily_station_means(df)

    # Features espaciales (excluir temporales)
    exclude_cols = ['datetime', 'date', 'year', 'month', 'day',
                    'estacion', 'archivo', 'validado', 'pm25']

    all_features = [col for col in df_daily.columns if col not in exclude_cols]

    logger.info(f"\nFeatures disponibles: {len(all_features)}")

    # Evaluar Regression Kriging
    results = evaluate_regression_kriging(
        df_daily,
        feature_cols=all_features,
        exclude_temporal=True
    )

    # Guardar resultados
    output_file = Path('data/processed/regression_kriging_results.csv')
    results.to_csv(output_file, index=False)
    logger.info(f"\n✓ Resultados guardados: {output_file}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ EVALUACIÓN COMPLETADA ✓✓✓")
    logger.info("="*70)

    return results


if __name__ == "__main__":
    results = main()
