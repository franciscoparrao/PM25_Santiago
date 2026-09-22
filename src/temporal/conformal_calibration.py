#!/usr/bin/env python3
"""
APR review Issue #6 — Calibracion de incertidumbre via conformal prediction.

El manuscrito reporta 61% de cobertura empirica (quantile regression) vs 90% nominal.
Aqui IMPLEMENTO split-conformal sobre las predicciones walk-forward 1-dia y reporto
la cobertura corregida + ancho medio, tanto global como estratificada (estacion del
anio y episodios PM2.5>=80), para exponer honestamente la heteroscedasticidad.

Dos variantes:
  (A) Split-conformal estandar (intervalo simetrico y_hat +/- q).
  (B) Conformal normalizado por nivel (y_hat +/- q*scale), ancho adaptativo -> mejora
      la cobertura condicional en episodios altos.

Split cronologico (calibracion = pasado, test = futuro) = escenario de despliegue real.
"""
import numpy as np, pandas as pd

X = 'data/processed/forecast_1d_predictions.csv'
ALPHA = 0.10  # 90% nominal

def conformal_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n+1)*(1-alpha)))
    k = min(k, n)  # clip
    return np.sort(scores)[k-1]

def season_of(m):
    return ('Summer' if m in (12,1,2) else 'Autumn' if m in (3,4,5)
            else 'Winter' if m in (6,7,8) else 'Spring')

def cov_width(y, lo, hi):
    cov = np.mean((y>=lo)&(y<=hi))
    return cov, np.mean(hi-lo)

def main():
    df = pd.read_csv(X, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df['resid'] = df['pm25_real'] - df['pm25_pred']
    df['season'] = df['date'].dt.month.map(season_of)
    # split cronologico: calibracion primeros 60%, test ultimos 40%
    cut = df['date'].quantile(0.6)
    cal = df[df['date'] <= cut]; te = df[df['date'] > cut].copy()
    print(f"Calibracion: {len(cal)} ({cal.date.min().date()}..{cal.date.max().date()}) | "
          f"Test: {len(te)} ({te.date.min().date()}..{te.date.max().date()})")

    # --- (A) split-conformal estandar ---
    qA = conformal_q(np.abs(cal['resid'].values), ALPHA)
    te['loA'] = te['pm25_pred'] - qA; te['hiA'] = te['pm25_pred'] + qA
    covA, wA = cov_width(te['pm25_real'].values, te['loA'].values, te['hiA'].values)

    # --- (B) conformal normalizado por nivel (scale = max(y_hat, 10)) ---
    floor = 10.0
    scale_cal = np.maximum(cal['pm25_pred'].values, floor)
    qB = conformal_q(np.abs(cal['resid'].values)/scale_cal, ALPHA)
    scale_te = np.maximum(te['pm25_pred'].values, floor)
    te['loB'] = te['pm25_pred'] - qB*scale_te; te['hiB'] = te['pm25_pred'] + qB*scale_te
    covB, wB = cov_width(te['pm25_real'].values, te['loB'].values, te['hiB'].values)

    print(f"\n(A) Split-conformal estandar:   coverage={covA*100:.1f}%  ancho medio={wA:.1f} ug/m3  (q={qA:.1f})")
    print(f"(B) Conformal normalizado nivel: coverage={covB*100:.1f}%  ancho medio={wB:.1f} ug/m3  (q={qB:.3f})")

    # cobertura condicional (por estacion y por episodio) para la variante A y B
    def strata(col, lo, hi, tag):
        print(f"\nCobertura condicional {tag}:")
        for s in ['Summer','Autumn','Winter','Spring']:
            d = te[te.season==s]
            c,_ = cov_width(d['pm25_real'].values, d[lo].values, d[hi].values)
            print(f"  {s:7s}: {c*100:5.1f}% (n={len(d)})")
        epi = te[te['pm25_real']>=80]
        if len(epi):
            c,_ = cov_width(epi['pm25_real'].values, epi[lo].values, epi[hi].values)
            print(f"  Episodios PM2.5>=80: {c*100:5.1f}% (n={len(epi)})")
    strata('season','loA','hiA','(A) estandar')
    strata('season','loB','hiB','(B) normalizado')

    out = pd.DataFrame([
        dict(method='raw_quantile_reg(manuscript)', coverage_pct=61.0, mean_width=np.nan),
        dict(method='split_conformal', coverage_pct=round(covA*100,1), mean_width=round(wA,1)),
        dict(method='normalized_conformal', coverage_pct=round(covB*100,1), mean_width=round(wB,1)),
    ])
    out.to_csv('data/processed/apr_6_conformal_coverage.csv', index=False)
    print('\nGuardado: data/processed/apr_6_conformal_coverage.csv')

if __name__ == '__main__':
    main()
