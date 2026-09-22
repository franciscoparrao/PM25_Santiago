#!/usr/bin/env python3
"""
APR 1.3 — Comparacion like-for-like de modelos temporales a 1 dia.

Problema (review APR): en el manuscrito XGBoost se evaluo sobre 15.128 puntos
estacion-dia mientras ARIMA/Prophet sobre ~47 origenes de la serie PROMEDIO-CIUDAD
-> R2 no comparable por tamano de muestra Y por granularidad del target (la media de
8 estaciones es mas suave y facil de predecir).

Solucion (este script): evaluar persistence, ARIMA y Prophet sobre EXACTAMENTE los
mismos origenes (date, estacion) que XGBoost ya uso (forecast_1d_predictions.csv),
con una muestra comun >=500 estratificada por estacion y tiempo, misma ventana de
entrenamiento (3 anios rodantes) y mismo horizonte (1 dia). XGBoost se toma de su
archivo (preserva el headline R2=0.76). Metricas sobre pares identicos + Diebold-Mariano
XGBoost vs cada baseline.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

DATA = 'data/processed/sinca_features_spatial.csv'
XGB = 'data/processed/forecast_1d_predictions.csv'
TRAIN = 365*3
PER_STATION = int(sys.argv[1]) if len(sys.argv) > 1 else 75   # origenes por estacion
SEED = 42

def load():
    feats = pd.read_csv(DATA, parse_dates=['date'])
    series = {st: g.set_index('date')['pm25'].sort_index().asfreq('D')
              for st, g in feats.groupby('estacion')}
    xgb = pd.read_csv(XGB, parse_dates=['date'])
    return series, xgb

def sample_origins(xgb, series):
    chosen = []
    for st, g in xgb.groupby('estacion'):
        g = g.sort_values('date').reset_index(drop=True)
        s = series.get(st)
        if s is None: continue
        valid = s.dropna().index.values.astype('datetime64[ns]')   # fechas con obs
        d = g['date'].values.astype('datetime64[ns]')
        lo = d - np.timedelta64(TRAIN, 'D'); hi = d - np.timedelta64(1, 'D')
        # obs disponibles en [d-TRAIN, d-1]
        cnt = np.searchsorted(valid, hi, side='right') - np.searchsorted(valid, lo, side='left')
        ok = g[cnt >= 900].reset_index(drop=True)
        if len(ok) == 0: continue
        take = min(PER_STATION, len(ok))
        idx = np.linspace(0, len(ok)-1, take).round().astype(int)   # espaciado uniforme en el tiempo
        chosen.append(ok.iloc[idx])
    return pd.concat(chosen).reset_index(drop=True)

def main():
    series, xgb = load()
    origins = sample_origins(xgb, series)
    print(f"Origenes comunes: {len(origins)} (objetivo {PER_STATION}/estacion, {origins['estacion'].nunique()} estaciones)")
    print(origins['estacion'].value_counts().to_string())

    actual = origins['pm25_real'].values
    pred_xgb = origins['pm25_pred'].values
    pred_persist = np.full(len(origins), np.nan)
    pred_arima = np.full(len(origins), np.nan)
    pred_prophet = np.full(len(origins), np.nan)

    # --- persistence: ultimo valor observado antes del target ---
    for k, (_, r) in enumerate(origins.iterrows()):
        s = series[r['estacion']]
        hist = s.loc[:r['date'] - pd.Timedelta(days=1)].dropna()
        pred_persist[k] = hist.iloc[-1] if len(hist) else np.nan

    # --- ARIMA: orden por estacion (auto_arima una vez), SARIMAX refit por origen ---
    from pmdarima import auto_arima
    import statsmodels.api as sm
    orders = {}
    t0 = time.time()
    for st in origins['estacion'].unique():
        s = series[st].dropna()
        warm = s.iloc[-TRAIN:].values if len(s) >= TRAIN else s.values
        try:
            aa = auto_arima(warm, seasonal=True, m=7, max_p=3,max_q=3,max_P=1,max_Q=1,
                            max_d=2,max_D=1, stepwise=True, suppress_warnings=True,
                            error_action='ignore')
            orders[st] = (aa.order, aa.seasonal_order)
        except Exception:
            orders[st] = ((1,1,1),(0,0,0,0))
    print(f"ARIMA ordenes por estacion ({time.time()-t0:.0f}s): " +
          "; ".join(f"{st}:{o[0]}x{o[1]}" for st,o in orders.items()))
    t0 = time.time()
    for k, (_, r) in enumerate(origins.iterrows()):
        s = series[r['estacion']]
        hist = s.loc[r['date'] - pd.Timedelta(days=TRAIN): r['date'] - pd.Timedelta(days=1)]
        hist = hist.interpolate(limit=3).dropna()
        order, sorder = orders[r['estacion']]
        try:
            m = sm.tsa.statespace.SARIMAX(hist.values, order=order, seasonal_order=sorder,
                                          enforce_stationarity=False, enforce_invertibility=False)
            res = m.fit(disp=False, maxiter=50)
            pred_arima[k] = res.forecast(steps=1)[-1]
        except Exception:
            pred_arima[k] = hist.values[-1]
        if (k+1) % 100 == 0: print(f"  ARIMA {k+1}/{len(origins)} ({time.time()-t0:.0f}s)")
    print(f"ARIMA total {time.time()-t0:.0f}s")

    # --- Prophet: refit por origen (patch bug 1.2.1 + pandas>=2) ---
    from prophet import Prophet
    import logging as _lg
    _lg.getLogger('prophet').setLevel(_lg.ERROR); _lg.getLogger('cmdstanpy').setLevel(_lg.ERROR)
    def _patched(self, components, name, group):
        nc = components[components['component'].isin(set(group))].copy()
        gc = nc['col'].unique()
        if len(gc) > 0:
            nc = pd.DataFrame({'col': gc, 'component': name})
            components = pd.concat([components, nc], ignore_index=True)
        return components
    Prophet.add_group_component = _patched
    pfail = 0; t0 = time.time()
    for k, (_, r) in enumerate(origins.iterrows()):
        s = series[r['estacion']]
        hist = s.loc[r['date'] - pd.Timedelta(days=TRAIN): r['date'] - pd.Timedelta(days=1)]
        hist = hist.interpolate(limit=3).dropna()
        tr = pd.DataFrame({'ds': pd.to_datetime(list(hist.index)), 'y': hist.values.astype(float)})
        try:
            mp = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                         changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
            mp.fit(tr)
            pred_prophet[k] = mp.predict(pd.DataFrame({'ds':[r['date']]}))['yhat'].values[0]
        except Exception:
            pred_prophet[k] = hist.values[-1]; pfail += 1
        if (k+1) % 100 == 0: print(f"  Prophet {k+1}/{len(origins)} ({time.time()-t0:.0f}s)")
    print(f"Prophet total {time.time()-t0:.0f}s (fallbacks: {pfail}/{len(origins)})")

    def metrics(name, pred):
        m = ~np.isnan(pred) & ~np.isnan(actual)
        a, p = actual[m], pred[m]
        mape = np.mean(np.abs((a-p)/np.where(a==0,np.nan,a)))*100
        return dict(model=name, n=int(m.sum()), r2=r2_score(a,p),
                    rmse=np.sqrt(mean_squared_error(a,p)), mae=mean_absolute_error(a,p), mape=mape)

    res = pd.DataFrame([metrics('Persistence',pred_persist), metrics('ARIMA',pred_arima),
                        metrics('Prophet',pred_prophet), metrics('XGBoost',pred_xgb)])
    print("\n"+"="*72+"\nLIKE-FOR-LIKE (mismos origenes date-estacion que XGBoost, 1 dia)\n"+"="*72)
    print(res.to_string(index=False))
    res.to_csv('data/processed/apr_1_3_like_for_like.csv', index=False)

    # Diebold-Mariano XGBoost vs cada uno (perdida = error absoluto)
    def dm(p1, p2):
        m = ~np.isnan(p1) & ~np.isnan(p2) & ~np.isnan(actual)
        d = np.abs(actual[m]-p1[m]) - np.abs(actual[m]-p2[m])
        dbar = d.mean(); n = len(d)
        # varianza con correccion Newey-West (lag 1)
        g0 = np.var(d, ddof=0); g1 = np.mean((d[1:]-dbar)*(d[:-1]-dbar))
        var = (g0 + 2*g1)/n
        stat = dbar/np.sqrt(var) if var>0 else np.nan
        from scipy import stats as st
        pval = 2*(1-st.norm.cdf(abs(stat))) if not np.isnan(stat) else np.nan
        return stat, pval
    print("\nDiebold-Mariano (XGBoost vs baseline, perdida=|error|; stat<0 => XGBoost mejor):")
    for name, p in [('Persistence',pred_persist),('ARIMA',pred_arima),('Prophet',pred_prophet)]:
        st_, pv = dm(pred_xgb, p)
        print(f"  XGBoost vs {name}: DM={st_:.2f}, p={pv:.4g}")

    out = origins[['date','estacion','pm25_real']].copy()
    out['persistence']=pred_persist; out['arima']=pred_arima
    out['prophet']=pred_prophet; out['xgboost']=pred_xgb
    out.to_csv('data/processed/apr_1_3_predictions_1d.csv', index=False)
    print("\nGuardado: apr_1_3_like_for_like.csv, apr_1_3_predictions_1d.csv")

if __name__ == '__main__':
    main()
