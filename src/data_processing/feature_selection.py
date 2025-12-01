#!/usr/bin/env python3
"""
Script para selección de features del dataset engineered.

Métodos aplicados:
1. Análisis de correlación (eliminar features altamente correlacionadas)
2. Feature importance con Random Forest
3. Variance Inflation Factor (VIF) para multicolinealidad
4. Mutual Information Score
5. Selección final basada en combinación de métodos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# VIF es opcional (requiere statsmodels)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning("statsmodels no disponible - VIF será omitido")

import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_feature_columns(df):
    """
    Identifica columnas que son features (no metadatos ni target).

    Args:
        df: DataFrame completo

    Returns:
        Lista de nombres de features
    """
    # Excluir metadatos y target
    exclude_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'lat', 'lon', 'elevation',
        'pm25',  # Target
        'validado', 'pm25_validado', 'pm25_preliminar',
        'archivo'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def analyze_correlations(df, feature_cols, threshold=0.90):
    """
    Analiza correlaciones entre features y encuentra pares altamente correlacionados.

    Args:
        df: DataFrame
        feature_cols: Lista de columnas de features
        threshold: Umbral de correlación (default 0.90)

    Returns:
        Dict con correlaciones altas y features a eliminar
    """
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS DE CORRELACIÓN")
    logger.info("="*60)

    # Calcular matriz de correlación
    corr_matrix = df[feature_cols].corr().abs()

    # Encontrar pares con correlación > threshold
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                high_corr_pairs.append((col_i, col_j, corr_val))

    logger.info(f"\nPares con correlación > {threshold}:")

    if high_corr_pairs:
        # Ordenar por correlación descendente
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        for col1, col2, corr in high_corr_pairs:
            logger.info(f"  • {col1} ↔ {col2}: {corr:.3f}")

        # Decidir cuál feature eliminar de cada par
        # Estrategia: eliminar la feature derivada, mantener la original
        features_to_drop = set()

        derived_priority = [
            # Más derivadas (eliminar primero)
            'wind_direction_rad',  # Mantener deg
            'era5_temperature_2m',  # Mantener celsius
            'era5_dewpoint_temperature_2m',  # Mantener celsius
            'era5_surface_pressure',  # Mantener hpa
            # Lag features redundantes
            'pm25_ma30',  # ma7 es más reciente
        ]

        for col1, col2, corr in high_corr_pairs:
            # Si una está en la lista de derivadas, eliminarla
            if col1 in derived_priority and col2 not in features_to_drop:
                features_to_drop.add(col1)
            elif col2 in derived_priority and col1 not in features_to_drop:
                features_to_drop.add(col2)
            elif col1 not in features_to_drop:
                # Si ninguna está en la lista, eliminar la primera
                features_to_drop.add(col1)

        logger.info(f"\nFeatures a eliminar por alta correlación ({len(features_to_drop)}):")
        for feat in sorted(features_to_drop):
            logger.info(f"  ✗ {feat}")
    else:
        logger.info("  ✓ No hay pares con correlación alta")
        features_to_drop = set()

    return {
        'high_corr_pairs': high_corr_pairs,
        'features_to_drop': features_to_drop
    }


def calculate_feature_importance_rf(df, feature_cols, target='pm25', n_estimators=100):
    """
    Calcula importancia de features usando Random Forest.

    Args:
        df: DataFrame
        feature_cols: Lista de features
        target: Nombre de la columna target
        n_estimators: Número de árboles

    Returns:
        DataFrame con importancias ordenadas
    """
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE - RANDOM FOREST")
    logger.info("="*60)

    X = df[feature_cols].copy()
    y = df[target]

    # Random Forest
    logger.info(f"\nEntrenando Random Forest ({n_estimators} árboles)...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # Obtener importancias
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 15 features más importantes:")
    for i, row in importances.head(15).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    logger.info(f"\nBottom 10 features menos importantes:")
    for i, row in importances.tail(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return importances


def calculate_mutual_information(df, feature_cols, target='pm25'):
    """
    Calcula Mutual Information Score para cada feature.

    Args:
        df: DataFrame
        feature_cols: Lista de features
        target: Columna target

    Returns:
        DataFrame con MI scores
    """
    logger.info("\n" + "="*60)
    logger.info("MUTUAL INFORMATION SCORE")
    logger.info("="*60)

    X = df[feature_cols].copy()
    y = df[target]

    logger.info("\nCalculando MI scores...")
    mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    logger.info(f"\nTop 15 features por MI score:")
    for i, row in mi_df.head(15).iterrows():
        logger.info(f"  {row['feature']:40s} {row['mi_score']:.4f}")

    return mi_df


def calculate_vif(df, feature_cols, threshold=10):
    """
    Calcula Variance Inflation Factor para detectar multicolinealidad.

    Args:
        df: DataFrame
        feature_cols: Lista de features
        threshold: VIF threshold (default 10)

    Returns:
        DataFrame con VIF scores (o None si statsmodels no está disponible)
    """
    if not HAS_STATSMODELS:
        logger.info("\n" + "="*60)
        logger.info("VARIANCE INFLATION FACTOR (VIF)")
        logger.info("="*60)
        logger.warning("\n⚠ statsmodels no disponible - VIF omitido")
        logger.info("  Para instalarlo: pip install statsmodels")
        return None

    logger.info("\n" + "="*60)
    logger.info("VARIANCE INFLATION FACTOR (VIF)")
    logger.info("="*60)

    X = df[feature_cols].copy()

    # Standardize para VIF
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols
    )

    logger.info(f"\nCalculando VIF (puede tardar unos minutos)...")

    vif_data = []
    for i, col in enumerate(feature_cols):
        if (i+1) % 10 == 0:
            logger.info(f"  Procesando {i+1}/{len(feature_cols)}...")

        try:
            vif = variance_inflation_factor(X_scaled.values, i)
            vif_data.append({'feature': col, 'vif': vif})
        except Exception as e:
            logger.warning(f"  Error calculando VIF para {col}: {e}")
            vif_data.append({'feature': col, 'vif': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)

    high_vif = vif_df[vif_df['vif'] > threshold]

    logger.info(f"\nFeatures con VIF > {threshold} (multicolinealidad alta):")
    if len(high_vif) > 0:
        for i, row in high_vif.iterrows():
            logger.info(f"  • {row['feature']:40s} VIF={row['vif']:.2f}")
    else:
        logger.info("  ✓ No hay features con VIF alto")

    return vif_df


def select_features_combined(importances_rf, mi_df, vif_df, corr_to_drop,
                              importance_threshold=0.001,
                              mi_threshold=0.01,
                              vif_threshold=15):
    """
    Selección final de features combinando todos los métodos.

    Criterios de eliminación:
    1. Correlación alta (ya identificadas)
    2. Importancia RF muy baja
    3. MI score muy bajo
    4. VIF muy alto (multicolinealidad extrema)

    Args:
        importances_rf: DataFrame con importancias RF
        mi_df: DataFrame con MI scores
        vif_df: DataFrame con VIF
        corr_to_drop: Set de features a eliminar por correlación
        importance_threshold: Umbral mínimo de importancia RF
        mi_threshold: Umbral mínimo de MI score
        vif_threshold: Umbral máximo de VIF

    Returns:
        Dict con features seleccionadas y eliminadas
    """
    logger.info("\n" + "="*60)
    logger.info("SELECCIÓN FINAL DE FEATURES")
    logger.info("="*60)

    all_features = set(importances_rf['feature'].tolist())

    # 1. Eliminar por correlación
    drop_corr = corr_to_drop

    # 2. Eliminar por importancia baja
    drop_importance = set(
        importances_rf[importances_rf['importance'] < importance_threshold]['feature'].tolist()
    )

    # 3. Eliminar por MI score bajo
    drop_mi = set(
        mi_df[mi_df['mi_score'] < mi_threshold]['feature'].tolist()
    )

    # 4. Eliminar por VIF extremadamente alto (si está disponible)
    if vif_df is not None:
        drop_vif = set(
            vif_df[vif_df['vif'] > vif_threshold]['feature'].tolist()
        )
    else:
        drop_vif = set()

    # Combinar todas las eliminaciones
    features_to_drop = drop_corr | drop_importance | drop_mi | drop_vif

    # Features seleccionadas
    selected_features = sorted(all_features - features_to_drop)

    logger.info(f"\nResumen de eliminación:")
    logger.info(f"  • Total features iniciales: {len(all_features)}")
    logger.info(f"  • Eliminadas por correlación alta: {len(drop_corr)}")
    logger.info(f"  • Eliminadas por importancia baja: {len(drop_importance)}")
    logger.info(f"  • Eliminadas por MI score bajo: {len(drop_mi)}")
    logger.info(f"  • Eliminadas por VIF alto: {len(drop_vif)}")
    logger.info(f"  • Total eliminadas (únicas): {len(features_to_drop)}")
    logger.info(f"  • Total seleccionadas: {len(selected_features)}")

    logger.info(f"\nFeatures ELIMINADAS ({len(features_to_drop)}):")
    for feat in sorted(features_to_drop):
        reasons = []
        if feat in drop_corr:
            reasons.append("correlación")
        if feat in drop_importance:
            reasons.append(f"import<{importance_threshold}")
        if feat in drop_mi:
            reasons.append(f"MI<{mi_threshold}")
        if feat in drop_vif:
            reasons.append(f"VIF>{vif_threshold}")

        logger.info(f"  ✗ {feat:40s} [{', '.join(reasons)}]")

    logger.info(f"\nFeatures SELECCIONADAS ({len(selected_features)}):")
    for feat in selected_features:
        logger.info(f"  ✓ {feat}")

    return {
        'selected': selected_features,
        'dropped': features_to_drop,
        'drop_reasons': {
            'correlation': drop_corr,
            'importance': drop_importance,
            'mi_score': drop_mi,
            'vif': drop_vif
        }
    }


def create_visualizations(df, feature_cols, importances_rf, output_dir):
    """
    Crea visualizaciones de feature importance y correlaciones.

    Args:
        df: DataFrame
        feature_cols: Lista de features
        importances_rf: DataFrame con importancias
        output_dir: Directorio para guardar figuras
    """
    logger.info("\n" + "="*60)
    logger.info("CREANDO VISUALIZACIONES")
    logger.info("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Importance (top 20)
    plt.figure(figsize=(12, 8))
    top_features = importances_rf.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Features - Random Forest Importance')
    plt.tight_layout()

    output_file = output_dir / 'feature_importance_top20.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 2. Correlation Heatmap (top 20 features)
    top_20_cols = importances_rf.head(20)['feature'].tolist()
    corr_matrix = df[top_20_cols].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix - Top 20 Features')
    plt.tight_layout()

    output_file = output_dir / 'correlation_heatmap_top20.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()

    # 3. Correlation with target
    target_corr = df[feature_cols + ['pm25']].corr()['pm25'].drop('pm25').abs().sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(target_corr.head(20))), target_corr.head(20).values)
    plt.yticks(range(len(target_corr.head(20))), target_corr.head(20).index)
    plt.xlabel('Absolute Correlation with PM2.5')
    plt.title('Top 20 Features - Correlation with PM2.5')
    plt.tight_layout()

    output_file = output_dir / 'correlation_with_target.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Guardado: {output_file}")
    plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION - SINCA + SATÉLITE")
    logger.info("="*70)

    # Cargar dataset
    input_file = Path('data/processed/sinca_features_engineered.csv')
    logger.info(f"\nCargando dataset: {input_file}")

    df = pd.read_csv(input_file, parse_dates=['datetime', 'date'])
    logger.info(f"  Registros: {len(df):,}")
    logger.info(f"  Columnas: {len(df.columns)}")

    # Identificar features
    feature_cols = get_feature_columns(df)
    logger.info(f"\nFeatures totales: {len(feature_cols)}")

    # 1. Análisis de correlación
    corr_results = analyze_correlations(df, feature_cols, threshold=0.90)

    # 2. Feature importance con RF
    importances_rf = calculate_feature_importance_rf(df, feature_cols, n_estimators=100)

    # 3. Mutual Information
    mi_df = calculate_mutual_information(df, feature_cols)

    # 4. VIF (puede tardar)
    vif_df = calculate_vif(df, feature_cols, threshold=10)

    # 5. Selección combinada
    selection_results = select_features_combined(
        importances_rf=importances_rf,
        mi_df=mi_df,
        vif_df=vif_df,
        corr_to_drop=corr_results['features_to_drop'],
        importance_threshold=0.001,
        mi_threshold=0.01,
        vif_threshold=15
    )

    # 6. Crear visualizaciones
    create_visualizations(
        df=df,
        feature_cols=feature_cols,
        importances_rf=importances_rf,
        output_dir=Path('reports/figures')
    )

    # 7. Guardar dataset con features seleccionadas
    selected_features = selection_results['selected']

    # Columnas a mantener: metadatos + target + features seleccionadas
    keep_cols = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'lat', 'lon', 'elevation',
        'pm25', 'validado',
        'archivo'
    ] + selected_features

    df_selected = df[keep_cols].copy()

    output_file = Path('data/processed/sinca_features_selected.csv')
    logger.info("\n" + "="*60)
    logger.info("GUARDANDO DATASET CON FEATURES SELECCIONADAS")
    logger.info("="*60)
    logger.info(f"Archivo: {output_file}")

    df_selected.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Guardado exitosamente")
    logger.info(f"  Tamaño: {size_mb:.2f} MB")
    logger.info(f"  Registros: {len(df_selected):,}")
    logger.info(f"  Features: {len(selected_features)}")
    logger.info(f"  Columnas totales: {len(df_selected.columns)}")

    # Guardar también metadatos de selección
    selection_summary = {
        'total_features': len(feature_cols),
        'selected_features': len(selected_features),
        'dropped_features': len(selection_results['dropped']),
        'selected_list': selected_features,
        'dropped_list': list(selection_results['dropped'])
    }

    summary_file = Path('data/processed/feature_selection_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("FEATURE SELECTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total features iniciales: {selection_summary['total_features']}\n")
        f.write(f"Features seleccionadas: {selection_summary['selected_features']}\n")
        f.write(f"Features eliminadas: {selection_summary['dropped_features']}\n\n")

        f.write("FEATURES SELECCIONADAS:\n")
        f.write("-"*70 + "\n")
        for feat in selected_features:
            f.write(f"  ✓ {feat}\n")

        f.write("\n\nFEATURES ELIMINADAS:\n")
        f.write("-"*70 + "\n")
        for feat in sorted(selection_results['dropped']):
            f.write(f"  ✗ {feat}\n")

    logger.info(f"\n✓ Resumen guardado: {summary_file}")

    # Guardar rankings
    rankings_file = Path('data/processed/feature_rankings.csv')
    rankings = importances_rf.merge(mi_df, on='feature')
    if vif_df is not None:
        rankings = rankings.merge(vif_df, on='feature')
    rankings.to_csv(rankings_file, index=False)
    logger.info(f"✓ Rankings guardados: {rankings_file}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ FEATURE SELECTION COMPLETADO ✓✓✓")
    logger.info("="*70)

    return df_selected, selection_results


if __name__ == "__main__":
    df_selected, results = main()
