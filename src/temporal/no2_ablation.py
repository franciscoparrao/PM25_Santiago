#!/usr/bin/env python3
"""
APR 2.1 — Ablacion del NO2 satelital en el forecasting 1-dia.

Compara tres condiciones sobre EXACTAMENTE los mismos origenes (walk-forward 1-dia,
ventana 3 anios rodante), en un modelo XGBoost temporal representativo:
  (a) sin NO2
  (b) NO2 mensual  (s5p_no2, el composite actual del manuscrito)
  (c) NO2 diario   (s5p_no2_daily, OFFL re-extraido; gaps -> NaN, XGBoost los maneja)

Reporta R2/RMSE/MAE por condicion y la importancia relativa del NO2 (b vs c).
Objetivo: responder al reviewer si re-extraer NO2 a resolucion diaria recupera
senal, o si el satelite es una capa evaluada-pero-no-decisiva para el pronostico.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

SPATIAL = 'data/processed/sinca_features_spatial.csv'
NO2DAILY = 'data/processed/no2_daily_offl.csv'
TRAIN = 365*3
PER_STATION = int(sys.argv[1]) if len(sys.argv) > 1 else 80

METEO = ['era5_u_component_of_wind_10m', 'era5_total_precipitation_hourly',
         'precipitation_sum7', 'wind_direction_rad']

def build(df):
    """features temporales por estacion (lags/rolling/diff/calendario + meteo)."""
    out = []
    for st, g in df.groupby('estacion'):
        g = g.sort_values('date').copy()
        s = g['pm25']
        for L in [1,2,3,7]:
            g[f'pm25_lag{L}'] = s.shift(L)
        g['pm25_ma3'] = s.shift(1).rolling(3).mean()
        g['pm25_ma7'] = s.shift(1).rolling(7).mean()
        g['pm25_ma14'] = s.shift(1).rolling(14).mean()
        g['pm25_std3'] = s.shift(1).rolling(3).std()
        g['pm25_std7'] = s.shift(1).rolling(7).std()
        g['pm25_diff1'] = s.shift(1) - s.shift(2)
        d = pd.to_datetime(g['date'])
        g['doy_sin'] = np.sin(2*np.pi*d.dt.dayofyear/365)
        g['doy_cos'] = np.cos(2*np.pi*d.dt.dayofyear/365)
        g['month'] = d.dt.month; g['dow'] = d.dt.dayofweek
        g['target'] = s.shift(-1)   # PM2.5 de manana (horizonte 1 dia)
        out.append(g)
    return pd.concat(out).reset_index(drop=True)

BASE = (['pm25_lag1','pm25_lag2','pm25_lag3','pm25_lag7','pm25_ma3','pm25_ma7','pm25_ma14',
         'pm25_std3','pm25_std7','pm25_diff1','doy_sin','doy_cos','month','dow','elevation']
        + METEO)

def xgbm():
    return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            n_jobs=-1, verbosity=0)

def sample_origins(df):
    """indices de origenes con >=900 dias de historia y target no nulo, ~PER_STATION/estacion."""
    chosen = []
    for st, g in df.groupby('estacion'):
        g = g.sort_values('date')
        elig = g[g['pm25_lag7'].notna() & g['target'].notna()].copy()
        # exigir historia suficiente: posicion en la serie
        elig = elig.iloc[TRAIN:] if len(elig) > TRAIN else elig.iloc[0:0]
        if len(elig) == 0: continue
        idx = np.linspace(0, len(elig)-1, min(PER_STATION, len(elig))).round().astype(int)
        chosen.append(elig.iloc[idx])
    return pd.concat(chosen)

def run_condition(df, feats, origins, no2col):
    cols = feats + ([no2col] if no2col else [])
    preds, acts, imp_no2 = [], [], []
    for _, r in origins.iterrows():
        st = r['estacion']; d = r['date']
        g = df[(df['estacion']==st) & (df['date'] < d)].tail(TRAIN)
        tr = g.dropna(subset=['target'])
        Xtr = tr[cols]; ytr = tr['target']
        # filas con demasiados NaN en features base se caen; XGBoost maneja NaN restantes
        m = xgbm(); m.fit(Xtr.values, ytr.values)
        Xte = r[cols].values.reshape(1,-1)
        preds.append(m.predict(Xte)[0]); acts.append(r['target'])
        if no2col:
            fi = m.feature_importances_[cols.index(no2col)]
            imp_no2.append(fi)
    a = np.array(acts); p = np.array(preds)
    res = dict(r2=r2_score(a,p), rmse=np.sqrt(mean_squared_error(a,p)), mae=mean_absolute_error(a,p))
    if no2col: res['no2_importance'] = float(np.mean(imp_no2))
    return res

def main():
    sp = pd.read_csv(SPATIAL, parse_dates=['date'])
    nd = pd.read_csv(NO2DAILY, parse_dates=['date'])
    df = sp.merge(nd, on=['estacion','date'], how='left')
    print(f"NO2 diario cobertura: {df['s5p_no2_daily'].notna().mean()*100:.1f}% de filas; "
          f"NO2 mensual: {df['s5p_no2'].notna().mean()*100:.1f}%")
    df = build(df)
    origins = sample_origins(df)
    print(f"Origenes comunes: {len(origins)} ({origins['estacion'].nunique()} estaciones, ~{PER_STATION}/est)")

    t0 = time.time()
    rows = []
    for name, no2col in [('(a) sin NO2', None), ('(b) NO2 mensual', 's5p_no2'),
                         ('(c) NO2 diario', 's5p_no2_daily')]:
        r = run_condition(df, BASE, origins, no2col)
        r['condition'] = name; rows.append(r)
        extra = f", NO2_imp={r.get('no2_importance', float('nan')):.4f}" if no2col else ""
        print(f"{name}: R2={r['r2']:.4f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}{extra}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)[['condition','r2','rmse','mae','no2_importance']]
    out.to_csv('data/processed/apr_2_1_no2_ablation.csv', index=False)
    print("\n"+out.to_string(index=False))
    base_r2 = out.loc[out['condition']=='(a) sin NO2','r2'].values[0]
    for _, r in out.iterrows():
        if r['condition'] != '(a) sin NO2':
            print(f"  DeltaR2 {r['condition']} vs sin NO2: {r['r2']-base_r2:+.4f}")
    print("\nGuardado: data/processed/apr_2_1_no2_ablation.csv")

if __name__ == '__main__':
    main()
