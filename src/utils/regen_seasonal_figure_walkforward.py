#!/usr/bin/env python3
"""
APR 2.3 — Regenera seasonal_pm25_and_performance.png con:
  Panel A: distribucion REAL de PM2.5 por estacion (no mock).
  Panel B: R2 por estacion del WALK-FORWARD (mismos puntos que el headline 0.76),
           con linea de R2 global (0.764). Reemplaza los valores del split 80/20.
Estilo matplotlib consistente con las demas figuras del paper.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE/'data'/'processed'
OUTS = [BASE/'reports'/'figures', BASE/'elsarticle']

ORDER = ['Summer','Autumn','Winter','Spring']
COLORS = ['#f39c12', '#e67e22', '#3498db', '#2ecc71']
MONTHS = {'Summer':[12,1,2],'Autumn':[3,4,5],'Winter':[6,7,8],'Spring':[9,10,11]}

def season_of(m):
    for s,mm in MONTHS.items():
        if m in mm: return s
    return 'NA'

def main():
    # --- datos reales de PM2.5 por estacion ---
    feats = pd.read_csv(DATA/'sinca_features_spatial.csv', parse_dates=['date'])
    feats['season'] = feats['date'].dt.month.map(season_of)
    bp_data = [feats.loc[feats.season==s,'pm25'].dropna().values for s in ORDER]

    # --- R2 walk-forward por estacion ---
    x = pd.read_csv(DATA/'forecast_1d_predictions.csv', parse_dates=['date'])
    x['season'] = x['date'].dt.month.map(season_of)
    r2 = {s: r2_score(g.pm25_real, g.pm25_pred) for s,g in x.groupby('season')}
    r2_vals = [r2[s] for s in ORDER]
    r2_global = r2_score(x.pm25_real, x.pm25_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Panel A
    bp = axes[0].boxplot(bp_data, labels=ORDER, patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp['medians']: med.set_color('black')
    axes[0].set_ylabel('PM$_{2.5}$ (µg/m³)', fontweight='bold')
    axes[0].set_title('(A) PM$_{2.5}$ Distribution by Season', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Panel B
    bars = axes[1].bar(ORDER, r2_vals, color=COLORS, alpha=0.7, edgecolor='black')
    axes[1].axhline(r2_global, color='black', linestyle='--', linewidth=1.3,
                    label=f'Overall walk-forward R$^2$ = {r2_global:.2f}')
    axes[1].set_ylabel('R$^2$ Score', fontweight='bold')
    axes[1].set_title('(B) XGBoost 1-Day-Ahead Performance by Season (walk-forward)', fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(loc='lower left', fontsize=9)
    for bar, val in zip(bars, r2_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+0.012, f'{val:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    for od in OUTS:
        od.mkdir(parents=True, exist_ok=True)
        plt.savefig(od/'seasonal_pm25_and_performance.png', dpi=150, bbox_inches='tight')
        print('guardado', od/'seasonal_pm25_and_performance.png')
    plt.close()
    print('R2 walk-forward por estacion:', {s: round(r2[s],3) for s in ORDER}, '| global', round(r2_global,4))

if __name__ == '__main__':
    main()
