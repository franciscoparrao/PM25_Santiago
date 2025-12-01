"""
Análisis de Detección de Episodios Críticos de PM2.5

Este script evalúa la capacidad del modelo para detectar episodios críticos
de contaminación (PM2.5 > 80 μg/m³), utilizando las predicciones del modelo
de forecasting con walk-forward validation.

Métricas calculadas:
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC
- Análisis por estación del año
- Análisis por horizonte de predicción (1d, 3d, 7d)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging
from pathlib import Path

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


def assign_season(month):
    """Asignar estación del año basado en el mes (hemisferio sur)."""
    if month in [12, 1, 2]:
        return 'Verano'
    elif month in [3, 4, 5]:
        return 'Otoño'
    elif month in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'


def load_forecasting_results():
    """Cargar resultados del forecasting walk-forward validation."""
    logging.info("\n" + "="*70)
    logging.info("CARGANDO RESULTADOS DE FORECASTING")
    logging.info("="*70)

    results = {}
    for horizon in [1, 3, 7]:
        file_path = DATA_DIR / f'forecast_{horizon}d_predictions.csv'
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=['date'])
            df['season'] = df['date'].dt.month.map(assign_season)
            results[horizon] = df
            logging.info(f"\n✓ Horizonte {horizon}d: {len(df):,} predicciones")
            logging.info(f"  Fecha rango: {df['date'].min()} → {df['date'].max()}")
        else:
            logging.warning(f"⚠ Archivo no encontrado: {file_path}")

    return results


def calculate_classification_metrics(y_true, y_pred, threshold=80):
    """
    Calcular métricas de clasificación binaria para episodios críticos.

    Args:
        y_true: Valores reales de PM2.5
        y_pred: Valores predichos de PM2.5
        threshold: Umbral para episodio crítico (default: 80 μg/m³)

    Returns:
        Dict con métricas de clasificación
    """
    # Convertir a clasificación binaria
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    # Métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # False Alarm Rate
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'False_Alarm_Rate': false_alarm_rate,
        'Total_Episodes': int((y_true >= threshold).sum()),
        'Detected_Episodes': int((y_pred >= threshold).sum())
    }

    return metrics


def analyze_by_horizon(results_dict, threshold=80):
    """Analizar detección de episodios críticos por horizonte de predicción."""
    logging.info("\n" + "="*70)
    logging.info("ANÁLISIS POR HORIZONTE DE PREDICCIÓN")
    logging.info("="*70)

    metrics_by_horizon = []

    for horizon, df in results_dict.items():
        y_true = df['pm25_real'].values
        y_pred = df['pm25_pred'].values

        metrics = calculate_classification_metrics(y_true, y_pred, threshold)
        metrics['Horizon'] = f'{horizon}d'
        metrics_by_horizon.append(metrics)

        logging.info(f"\n{'-'*70}")
        logging.info(f"HORIZONTE: {horizon} días")
        logging.info(f"{'-'*70}")
        logging.info(f"  Episodios reales (PM2.5 ≥ {threshold}):     {metrics['Total_Episodes']:,}")
        logging.info(f"  Episodios detectados (pred ≥ {threshold}):  {metrics['Detected_Episodes']:,}")
        logging.info(f"\nConfusion Matrix:")
        logging.info(f"  TP (Correcto Positivo): {metrics['TP']:>6,}  |  FN (Falso Negativo): {metrics['FN']:>6,}")
        logging.info(f"  FP (Falso Positivo):    {metrics['FP']:>6,}  |  TN (Correcto Negativo): {metrics['TN']:>6,}")
        logging.info(f"\nMétricas de Clasificación:")
        logging.info(f"  Precision:          {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
        logging.info(f"  Recall (Sensibilidad): {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
        logging.info(f"  F1-Score:           {metrics['F1-Score']:.4f}")
        logging.info(f"  Specificity:        {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)")
        logging.info(f"  Accuracy:           {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        logging.info(f"  False Alarm Rate:   {metrics['False_Alarm_Rate']:.4f} ({metrics['False_Alarm_Rate']*100:.2f}%)")

    return pd.DataFrame(metrics_by_horizon)


def analyze_by_season(results_dict, threshold=80):
    """Analizar detección de episodios críticos por estación del año."""
    logging.info("\n" + "="*70)
    logging.info("ANÁLISIS POR ESTACIÓN DEL AÑO")
    logging.info("="*70)

    metrics_by_season = []

    # Usar horizonte 1d para análisis estacional
    df = results_dict[1]

    for season in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
        season_data = df[df['season'] == season]

        if len(season_data) == 0:
            continue

        y_true = season_data['pm25_real'].values
        y_pred = season_data['pm25_pred'].values

        metrics = calculate_classification_metrics(y_true, y_pred, threshold)
        metrics['Season'] = season
        metrics_by_season.append(metrics)

        logging.info(f"\n{'-'*70}")
        logging.info(f"ESTACIÓN: {season}")
        logging.info(f"{'-'*70}")
        logging.info(f"  Total observaciones: {len(season_data):,}")
        logging.info(f"  Episodios reales:    {metrics['Total_Episodes']:,} ({metrics['Total_Episodes']/len(season_data)*100:.2f}%)")
        logging.info(f"  Episodios detectados: {metrics['Detected_Episodes']:,}")
        logging.info(f"\nMétricas:")
        logging.info(f"  Precision:  {metrics['Precision']:.4f}")
        logging.info(f"  Recall:     {metrics['Recall']:.4f}")
        logging.info(f"  F1-Score:   {metrics['F1-Score']:.4f}")

    return pd.DataFrame(metrics_by_season)


def analyze_error_cases(results_dict, threshold=80):
    """Analizar casos de falsos positivos y falsos negativos."""
    logging.info("\n" + "="*70)
    logging.info("ANÁLISIS DE ERRORES (Falsos Positivos y Negativos)")
    logging.info("="*70)

    # Usar horizonte 1d
    df = results_dict[1].copy()

    # Clasificación binaria
    df['actual_critical'] = (df['pm25_real'] >= threshold).astype(int)
    df['pred_critical'] = (df['pm25_pred'] >= threshold).astype(int)

    # Identificar tipos de predicción
    df['prediction_type'] = 'TN'
    df.loc[(df['actual_critical'] == 1) & (df['pred_critical'] == 1), 'prediction_type'] = 'TP'
    df.loc[(df['actual_critical'] == 0) & (df['pred_critical'] == 1), 'prediction_type'] = 'FP'
    df.loc[(df['actual_critical'] == 1) & (df['pred_critical'] == 0), 'prediction_type'] = 'FN'

    # Analizar Falsos Negativos (episodios críticos NO detectados)
    fn_cases = df[df['prediction_type'] == 'FN'].copy()
    logging.info(f"\n{'─'*70}")
    logging.info(f"FALSOS NEGATIVOS (Episodios críticos NO detectados)")
    logging.info(f"{'─'*70}")
    logging.info(f"  Total: {len(fn_cases):,} casos")

    if len(fn_cases) > 0:
        fn_cases['underprediction'] = fn_cases['pm25_real'] - fn_cases['pm25_pred']
        logging.info(f"  PM2.5 real promedio:    {fn_cases['pm25_real'].mean():.2f} μg/m³")
        logging.info(f"  PM2.5 predicho promedio: {fn_cases['pm25_pred'].mean():.2f} μg/m³")
        logging.info(f"  Subestimación promedio: {fn_cases['underprediction'].mean():.2f} μg/m³")
        logging.info(f"\n  Por estación:")
        for season in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
            season_fn = fn_cases[fn_cases['season'] == season]
            if len(season_fn) > 0:
                logging.info(f"    {season:12}: {len(season_fn):>4} casos ({len(season_fn)/len(fn_cases)*100:>5.2f}%)")

    # Analizar Falsos Positivos (alertas falsas)
    fp_cases = df[df['prediction_type'] == 'FP'].copy()
    logging.info(f"\n{'─'*70}")
    logging.info(f"FALSOS POSITIVOS (Alertas falsas)")
    logging.info(f"{'─'*70}")
    logging.info(f"  Total: {len(fp_cases):,} casos")

    if len(fp_cases) > 0:
        fp_cases['overprediction'] = fp_cases['pm25_pred'] - fp_cases['pm25_real']
        logging.info(f"  PM2.5 real promedio:    {fp_cases['pm25_real'].mean():.2f} μg/m³")
        logging.info(f"  PM2.5 predicho promedio: {fp_cases['pm25_pred'].mean():.2f} μg/m³")
        logging.info(f"  Sobrestimación promedio: {fp_cases['overprediction'].mean():.2f} μg/m³")
        logging.info(f"\n  Por estación:")
        for season in ['Verano', 'Otoño', 'Invierno', 'Primavera']:
            season_fp = fp_cases[fp_cases['season'] == season]
            if len(season_fp) > 0:
                logging.info(f"    {season:12}: {len(season_fp):>4} casos ({len(season_fp)/len(fp_cases)*100:>5.2f}%)")

    return df, fn_cases, fp_cases


def plot_critical_episodes_analysis(metrics_horizon, metrics_season, df_classified):
    """Crear visualizaciones del análisis de episodios críticos."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Métricas por horizonte
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(metrics_horizon))
    width = 0.25

    ax1.bar(x - width, metrics_horizon['Precision'], width, label='Precision', alpha=0.8)
    ax1.bar(x, metrics_horizon['Recall'], width, label='Recall', alpha=0.8)
    ax1.bar(x + width, metrics_horizon['F1-Score'], width, label='F1-Score', alpha=0.8)

    ax1.set_xlabel('Horizonte de Predicción', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Métricas de Detección por Horizonte (PM2.5 ≥ 80 μg/m³)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_horizon['Horizon'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Añadir valores en las barras
    for i, row in metrics_horizon.iterrows():
        ax1.text(i - width, row['Precision'] + 0.02, f"{row['Precision']:.3f}",
                ha='center', va='bottom', fontsize=9)
        ax1.text(i, row['Recall'] + 0.02, f"{row['Recall']:.3f}",
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, row['F1-Score'] + 0.02, f"{row['F1-Score']:.3f}",
                ha='center', va='bottom', fontsize=9)

    # 2. Confusion Matrix (1 día)
    ax2 = fig.add_subplot(gs[0, 2])
    cm = np.array([[metrics_horizon.iloc[0]['TN'], metrics_horizon.iloc[0]['FP']],
                   [metrics_horizon.iloc[0]['FN'], metrics_horizon.iloc[0]['TP']]])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=['No Crítico', 'Crítico'],
                yticklabels=['No Crítico', 'Crítico'])
    ax2.set_xlabel('Predicho', fontsize=11)
    ax2.set_ylabel('Real', fontsize=11)
    ax2.set_title('Confusion Matrix\n(1-día ahead)', fontsize=11, fontweight='bold')

    # 3. Métricas por estación
    ax3 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(metrics_season))

    ax3.bar(x - width, metrics_season['Precision'], width, label='Precision', alpha=0.8)
    ax3.bar(x, metrics_season['Recall'], width, label='Recall', alpha=0.8)
    ax3.bar(x + width, metrics_season['F1-Score'], width, label='F1-Score', alpha=0.8)

    ax3.set_xlabel('Estación del Año', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Métricas de Detección por Estación (1-día ahead)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_season['Season'])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Episodios por estación
    ax4 = fig.add_subplot(gs[1, 2])
    colors_season = ['#FFD700', '#FF8C00', '#4169E1', '#32CD32']
    bars = ax4.bar(metrics_season['Season'], metrics_season['Total_Episodes'],
                   color=colors_season, alpha=0.7, edgecolor='black')

    ax4.set_ylabel('Número de Episodios', fontsize=11)
    ax4.set_title('Episodios Críticos\npor Estación', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for i, (bar, row) in enumerate(zip(bars, metrics_season.itertuples())):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5. Distribución de tipos de predicción
    ax5 = fig.add_subplot(gs[2, 0])
    pred_counts = df_classified['prediction_type'].value_counts()
    colors_pred = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#f39c12'}

    wedges, texts, autotexts = ax5.pie(pred_counts.values, labels=pred_counts.index,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=[colors_pred[k] for k in pred_counts.index])

    ax5.set_title('Distribución de\nPredicciones (1d)', fontsize=11, fontweight='bold')

    # Leyenda
    ax5.legend(['TP: Verdadero Positivo', 'TN: Verdadero Negativo',
                'FP: Falso Positivo', 'FN: Falso Negativo'],
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

    # 6. False Alarm Rate por horizonte
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.bar(metrics_horizon['Horizon'], metrics_horizon['False_Alarm_Rate'],
            color='#e74c3c', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Horizonte', fontsize=11)
    ax6.set_ylabel('False Alarm Rate', fontsize=11)
    ax6.set_title('Tasa de Falsas Alarmas', fontsize=11, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim([0, max(metrics_horizon['False_Alarm_Rate']) * 1.2])

    for i, row in metrics_horizon.iterrows():
        ax6.text(i, row['False_Alarm_Rate'] + 0.001,
                f"{row['False_Alarm_Rate']:.4f}",
                ha='center', va='bottom', fontsize=9)

    # 7. Accuracy por estación
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.bar(metrics_season['Season'], metrics_season['Accuracy'],
            color=colors_season, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Estación', fontsize=11)
    ax7.set_ylabel('Accuracy', fontsize=11)
    ax7.set_title('Accuracy por Estación', fontsize=11, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim([0.9, 1.0])

    for i, (bar, row) in enumerate(zip(ax7.patches, metrics_season.itertuples())):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{row.Accuracy:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Análisis de Detección de Episodios Críticos (PM2.5 ≥ 80 μg/m³)',
                fontsize=14, fontweight='bold', y=0.995)

    return fig


def main():
    """Ejecutar análisis completo de detección de episodios críticos."""

    # 1. Cargar resultados de forecasting
    results = load_forecasting_results()

    if not results:
        logging.error("No se encontraron archivos de predicción.")
        return

    # 2. Análisis por horizonte
    metrics_horizon = analyze_by_horizon(results, threshold=80)

    # 3. Análisis por estación
    metrics_season = analyze_by_season(results, threshold=80)

    # 4. Análisis de errores
    df_classified, fn_cases, fp_cases = analyze_error_cases(results, threshold=80)

    # 5. Guardar resultados
    logging.info("\n" + "="*70)
    logging.info("GUARDANDO RESULTADOS")
    logging.info("="*70)

    metrics_horizon.to_csv(DATA_DIR / 'critical_episodes_metrics_by_horizon.csv', index=False)
    logging.info(f"\n✓ Métricas por horizonte: {DATA_DIR / 'critical_episodes_metrics_by_horizon.csv'}")

    metrics_season.to_csv(DATA_DIR / 'critical_episodes_metrics_by_season.csv', index=False)
    logging.info(f"✓ Métricas por estación: {DATA_DIR / 'critical_episodes_metrics_by_season.csv'}")

    # 6. Generar visualizaciones
    fig = plot_critical_episodes_analysis(metrics_horizon, metrics_season, df_classified)
    fig_path = REPORTS_DIR / 'critical_episodes_detection.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Gráfico guardado: {fig_path}")
    plt.close()

    # 7. Resumen ejecutivo
    logging.info("\n" + "="*70)
    logging.info("✓✓✓ ANÁLISIS DE EPISODIOS CRÍTICOS COMPLETADO ✓✓✓")
    logging.info("="*70)

    logging.info("\n" + "="*70)
    logging.info("RESUMEN EJECUTIVO")
    logging.info("="*70)

    logging.info("\nDetección 1-día ahead:")
    m1 = metrics_horizon[metrics_horizon['Horizon'] == '1d'].iloc[0]
    logging.info(f"  Precision:  {m1['Precision']:.4f} ({m1['Precision']*100:.2f}%)")
    logging.info(f"  Recall:     {m1['Recall']:.4f} ({m1['Recall']*100:.2f}%)")
    logging.info(f"  F1-Score:   {m1['F1-Score']:.4f}")
    logging.info(f"  Episodios detectados: {m1['Detected_Episodes']}/{m1['Total_Episodes']}")

    logging.info("\nMejor estación (F1-Score):")
    best_season = metrics_season.loc[metrics_season['F1-Score'].idxmax()]
    logging.info(f"  {best_season['Season']}: F1 = {best_season['F1-Score']:.4f}")

    logging.info("\nPeor estación (F1-Score):")
    worst_season = metrics_season.loc[metrics_season['F1-Score'].idxmin()]
    logging.info(f"  {worst_season['Season']}: F1 = {worst_season['F1-Score']:.4f}")

    logging.info("\n" + "="*70)


if __name__ == '__main__':
    main()
