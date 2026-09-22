#!/usr/bin/env python3
"""
APR review Issue #2 — Replicacion del hallazgo de dias de transicion en una 2da cuenca.

Salt Lake City (Utah), cuenca profunda con inversiones invernales severas (analogo de
Santiago). Prueba si el hallazgo central transfiere: el valor del modelo aprendido sobre
persistence se concentra en los dias de transicion de episodio y crece con el horizonte.

Walk-forward 1-dia, XGBoost (lags/rolling PM2.5 + meteo ERA5 + calendario) vs persistence,
mismos puntos. Descomposicion: R2 en dias quasi-estacionarios vs dias de transicion
(|delta PM2.5| en el ~12% superior, para comparabilidad con Santiago).
Salida: data/processed/apr_2_slc_transition.csv
"""
import sys, time
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

PM25='data/external/slc_pm25_daily.csv'
MET='data/external/slc_meteo_daily.csv'
TRAIN=365*3
PER_SITE=int(sys.argv[1]) if len(sys.argv)>1 else 120
SITES=['Hawthorne','Copper View','Herriman #3','Near Road','ROSE PARK']

def build(df):
    out=[]
    for st,g in df.groupby('site'):
        g=g.sort_values('date').copy(); s=g['pm25']
        for L in [1,2,3,7]: g[f'lag{L}']=s.shift(L)
        g['ma3']=s.shift(1).rolling(3).mean(); g['ma7']=s.shift(1).rolling(7).mean()
        g['ma14']=s.shift(1).rolling(14).mean(); g['std3']=s.shift(1).rolling(3).std()
        g['std7']=s.shift(1).rolling(7).std(); g['diff1']=s.shift(1)-s.shift(2)
        g['wspd']=np.hypot(g['u10'],g['v10']); g['wdir']=np.arctan2(g['v10'],g['u10'])
        g['precip7']=g['precip'].shift(1).rolling(7).sum()
        d=pd.to_datetime(g['date'])
        g['doy_sin']=np.sin(2*np.pi*d.dt.dayofyear/365); g['doy_cos']=np.cos(2*np.pi*d.dt.dayofyear/365)
        g['month']=d.dt.month; g['dow']=d.dt.dayofweek
        g['target']=s.shift(-1); g['persist']=s   # persistence = hoy
        out.append(g)
    return pd.concat(out).reset_index(drop=True)

FEATS=['lag1','lag2','lag3','lag7','ma3','ma7','ma14','std3','std7','diff1',
       'u10','v10','wspd','wdir','precip','precip7','t2m','doy_sin','doy_cos','month','dow']

def xgbm():
    return xgb.XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=4,subsample=0.8,
                            colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=0)

def r2_safe(a,p):
    return r2_score(a,p) if len(a)>=5 else np.nan

def main():
    pm=pd.read_csv(PM25,parse_dates=['date']); met=pd.read_csv(MET,parse_dates=['date'])
    df=pm.merge(met,on=['site','date'],how='inner')
    df=df[df.site.isin(SITES)]
    print(f"SLC merged: {len(df)} filas, sitios {sorted(df.site.unique())}")
    print(f"PM2.5 media {df.pm25.mean():.1f}, p95 {df.pm25.quantile(.95):.1f}, max {df.pm25.max():.1f}")
    df=build(df)

    origins=[]
    for st,g in df.groupby('site'):
        g=g.sort_values('date'); elig=g[g['lag7'].notna()&g['target'].notna()&g['t2m'].notna()]
        elig=elig.iloc[TRAIN:] if len(elig)>TRAIN else elig.iloc[0:0]
        if len(elig)==0: continue
        idx=np.linspace(0,len(elig)-1,min(PER_SITE,len(elig))).round().astype(int)
        origins.append(elig.iloc[idx])
    origins=pd.concat(origins).reset_index(drop=True)
    print(f"Origenes: {len(origins)} ({origins.site.nunique()} sitios)")

    preds=np.full(len(origins),np.nan); t0=time.time()
    for k,(_,r) in enumerate(origins.iterrows()):
        g=df[(df.site==r['site'])&(df.date<r['date'])].tail(TRAIN).dropna(subset=['target']+FEATS)
        if len(g)<200: preds[k]=r['persist']; continue
        m=xgbm(); m.fit(g[FEATS].values,g['target'].values)
        preds[k]=m.predict(r[FEATS].values.reshape(1,-1))[0]
        if (k+1)%100==0: print(f"  {k+1}/{len(origins)} ({time.time()-t0:.0f}s)")
    o=origins.assign(xgb=preds, persist=origins['persist'].values, actual=origins['target'].values)
    o['delta']=np.abs(o['actual']-o['persist'])

    # overall
    a=o['actual'].values
    R2x=r2_safe(a,o['xgb'].values); R2p=r2_safe(a,o['persist'].values)
    print(f"\nOVERALL 1-dia: XGBoost R2={R2x:.3f} | Persistence R2={R2p:.3f} "
          f"| RMSE_xgb={np.sqrt(mean_squared_error(a,o['xgb'])):.2f}")

    # descomposicion transicion (~12% superior de |delta|, como Santiago)
    thr=o['delta'].quantile(0.88)
    trans=o[o['delta']>thr]; stat=o[o['delta']<=thr]
    print(f"\nUmbral transicion |delta|>{thr:.1f} ug/m3 ({len(trans)}/{len(o)} = {100*len(trans)/len(o):.0f}% dias)")
    print(f"  QUASI-ESTACIONARIOS: Persistence R2={r2_safe(stat['actual'],stat['persist']):.3f}  "
          f"XGBoost R2={r2_safe(stat['actual'],stat['xgb']):.3f}")
    print(f"  TRANSICION:          Persistence R2={r2_safe(trans['actual'],trans['persist']):.3f}  "
          f"XGBoost R2={r2_safe(trans['actual'],trans['xgb']):.3f}")

    res=pd.DataFrame([
      dict(regime='overall', n=len(o), persistence_r2=round(R2p,3), xgboost_r2=round(R2x,3)),
      dict(regime='quasi-stationary', n=len(stat),
           persistence_r2=round(r2_safe(stat['actual'],stat['persist']),3),
           xgboost_r2=round(r2_safe(stat['actual'],stat['xgb']),3)),
      dict(regime='transition', n=len(trans),
           persistence_r2=round(r2_safe(trans['actual'],trans['persist']),3),
           xgboost_r2=round(r2_safe(trans['actual'],trans['xgb']),3)),
    ])
    res['transition_threshold']=round(thr,1)
    res.to_csv('data/processed/apr_2_slc_transition.csv',index=False)
    o[['date','site','actual','persist','xgb','delta']].to_csv(
        'data/processed/apr_2_slc_predictions.csv',index=False)
    print("\nGuardado: apr_2_slc_transition.csv, apr_2_slc_predictions.csv")

if __name__=='__main__':
    main()
