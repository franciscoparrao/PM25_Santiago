#!/usr/bin/env python3
"""
APR review Issue #3 — Configuracion operacional integrada (un solo artefacto).

Integra los dos upgrades que las ablaciones justificaron en un UNICO modelo desplegado
y lo evalua end-to-end:
  (1) NO2 diario OFFL como predictor (vs el composite mensual del modelo base de analisis)
  (2) intervalos calibrados por conformal NORMALIZADO por nivel (episode-aware)

Walk-forward 1-dia, refit por origen, ventana 3 anios. Reporta skill puntual del sistema
desplegado + cobertura calibrada (global y en episodios >=80), en una sola evaluacion.
Salida: data/processed/apr_3_deployed_system.csv (+ per-origin predictions).
"""
import sys, time
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

SPATIAL = 'data/processed/sinca_features_spatial.csv'
NO2DAILY = 'data/processed/no2_daily_offl.csv'
TRAIN = 365*3
PER_STATION = int(sys.argv[1]) if len(sys.argv) > 1 else 150
ALPHA = 0.10
METEO = ['era5_u_component_of_wind_10m','era5_total_precipitation_hourly',
         'precipitation_sum7','wind_direction_rad']
BASE = (['pm25_lag1','pm25_lag2','pm25_lag3','pm25_lag7','pm25_ma3','pm25_ma7','pm25_ma14',
         'pm25_std3','pm25_std7','pm25_diff1','doy_sin','doy_cos','month','dow','elevation'] + METEO)
DEPLOYED = BASE + ['s5p_no2_daily']   # sistema desplegado = base + NO2 diario

def build(df):
    out=[]
    for st,g in df.groupby('estacion'):
        g=g.sort_values('date').copy(); s=g['pm25']
        for L in [1,2,3,7]: g[f'pm25_lag{L}']=s.shift(L)
        g['pm25_ma3']=s.shift(1).rolling(3).mean(); g['pm25_ma7']=s.shift(1).rolling(7).mean()
        g['pm25_ma14']=s.shift(1).rolling(14).mean(); g['pm25_std3']=s.shift(1).rolling(3).std()
        g['pm25_std7']=s.shift(1).rolling(7).std(); g['pm25_diff1']=s.shift(1)-s.shift(2)
        d=pd.to_datetime(g['date'])
        g['doy_sin']=np.sin(2*np.pi*d.dt.dayofyear/365); g['doy_cos']=np.cos(2*np.pi*d.dt.dayofyear/365)
        g['month']=d.dt.month; g['dow']=d.dt.dayofweek; g['target']=s.shift(-1)
        out.append(g)
    return pd.concat(out).reset_index(drop=True)

def xgbm():
    return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)

def conformal_q(scores, alpha):
    n=len(scores); k=min(int(np.ceil((n+1)*(1-alpha))), n); return np.sort(scores)[k-1]

def season_of(m):
    return ('Summer' if m in (12,1,2) else 'Autumn' if m in (3,4,5)
            else 'Winter' if m in (6,7,8) else 'Spring')

def main():
    sp=pd.read_csv(SPATIAL, parse_dates=['date'])
    nd=pd.read_csv(NO2DAILY, parse_dates=['date'])
    df=build(sp.merge(nd,on=['estacion','date'],how='left'))

    # origenes: densos y espaciados en el tiempo, >=900 dias de historia
    origins=[]
    for st,g in df.groupby('estacion'):
        g=g.sort_values('date'); elig=g[g['pm25_lag7'].notna() & g['target'].notna()]
        elig=elig.iloc[TRAIN:] if len(elig)>TRAIN else elig.iloc[0:0]
        if len(elig)==0: continue
        idx=np.linspace(0,len(elig)-1,min(PER_STATION,len(elig))).round().astype(int)
        origins.append(elig.iloc[idx])
    origins=pd.concat(origins).sort_values('date').reset_index(drop=True)
    print(f"Origenes: {len(origins)} ({origins.estacion.nunique()} estaciones)")

    # walk-forward refit por origen (sistema desplegado)
    preds=np.full(len(origins),np.nan); t0=time.time()
    for k,(_,r) in enumerate(origins.iterrows()):
        g=df[(df.estacion==r['estacion'])&(df.date<r['date'])].tail(TRAIN).dropna(subset=['target'])
        m=xgbm(); m.fit(g[DEPLOYED].values, g['target'].values)
        preds[k]=m.predict(r[DEPLOYED].values.reshape(1,-1))[0]
        if (k+1)%200==0: print(f"  {k+1}/{len(origins)} ({time.time()-t0:.0f}s)")
    origins=origins.assign(pred=preds, actual=origins['target'].values)
    origins['season']=pd.to_datetime(origins['date']).dt.month.map(season_of)

    a=origins['actual'].values; p=origins['pred'].values
    r2=r2_score(a,p); rmse=np.sqrt(mean_squared_error(a,p)); mae=mean_absolute_error(a,p)
    print(f"\nSistema desplegado (base+NO2 diario): R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f}")

    # conformal normalizado por nivel sobre las predicciones del sistema desplegado
    origins=origins.sort_values('date').reset_index(drop=True)
    cut=origins['date'].quantile(0.6); cal=origins[origins.date<=cut]; te=origins[origins.date>cut].copy()
    floor=10.0; resid_cal=np.abs(cal['actual'].values-cal['pred'].values)
    q=conformal_q(resid_cal/np.maximum(cal['pred'].values,floor), ALPHA)
    sc=np.maximum(te['pred'].values,floor); lo=te['pred'].values-q*sc; hi=te['pred'].values+q*sc
    cov=np.mean((te['actual'].values>=lo)&(te['actual'].values<=hi)); width=np.mean(hi-lo)
    epi=te[te['actual']>=80]
    if len(epi):
        sce=np.maximum(epi['pred'].values,floor)
        loe=epi['pred'].values-q*sce; hie=epi['pred'].values+q*sce
        cov_epi=np.mean((epi['actual'].values>=loe)&(epi['actual'].values<=hie))
    else: cov_epi=np.nan
    print(f"Intervalos conformal-normalizado (90% nominal): cobertura={cov*100:.1f}% "
          f"(episodios>=80: {cov_epi*100:.1f}%, n={len(epi)}), ancho medio={width:.1f} ug/m3")

    pd.DataFrame([dict(config='deployed (base+daily NO2)', n=len(origins), r2=round(r2,3),
                       rmse=round(rmse,2), mae=round(mae,2),
                       conformal_coverage_pct=round(cov*100,1),
                       conformal_episode_coverage_pct=round(cov_epi*100,1),
                       conformal_mean_width=round(width,1))]
                 ).to_csv('data/processed/apr_3_deployed_system.csv', index=False)
    origins[['date','estacion','actual','pred','season']].to_csv(
        'data/processed/apr_3_deployed_predictions.csv', index=False)
    print("Guardado: apr_3_deployed_system.csv, apr_3_deployed_predictions.csv")

if __name__=='__main__':
    main()
