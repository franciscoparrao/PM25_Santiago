#!/usr/bin/env python3
"""
APR 1.2 — Re-evaluación espacial bajo LOSO-CV.

Objetivo: reportar el mejor R² espacial alcanzable bajo features gruesas + 8 estaciones,
para reformular el claim "spatial interpolation fails (R²=-1.09)" de forma honesta.

Modelos: Lasso (reproduce el baseline del manuscrito), XGBoost-espacial, Regression Kriging.
Feature sets: 'base' (13 features, como el manuscrito) y 'enhanced' (+5 features OSM locales).

RK implementado correctamente para interpolación espacial:
  tendencia XGBoost global (fit sobre estaciones de entrenamiento)
  + kriging de residuales POR DÍA sobre las estaciones de entrenamiento
  -> predice el residual en la estación excluida.
(Evita el kriging degenerado sobre miles de coordenadas coincidentes.)
"""
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from pykrige.ok import OrdinaryKriging
warnings.filterwarnings('ignore')

DATA = 'data/processed/sinca_features_spatial_enhanced.csv'
EXCLUDE = ['datetime','date','year','month','day','estacion','archivo','validado','pm25','lat.1']
OSM = ['dist_to_highway_km','dist_to_primary_km','road_density_500m','road_density_1km','highway_count_1km']

def xgb_model():
    return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)

def eval_plain_model(df, feats, make_model):
    """LOSO-CV para un modelo sklearn-like con scaling."""
    stations = sorted(df['estacion'].unique())
    rows = []
    for st in stations:
        tr = df[df['estacion'] != st]; te = df[df['estacion'] == st]
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].values); Xte = sc.transform(te[feats].values)
        m = make_model(); m.fit(Xtr, tr['pm25'].values)
        pred = m.predict(Xte)
        yte = te['pm25'].values
        rows.append(dict(station=st, n_test=len(yte),
                         r2=r2_score(yte,pred), rmse=np.sqrt(mean_squared_error(yte,pred)),
                         mae=mean_absolute_error(yte,pred)))
    return pd.DataFrame(rows)

def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _fit_spherical(h, g):
    """Ajuste simple de variograma esférico por grid-search (robusto, 21 puntos)."""
    from itertools import product
    h = np.asarray(h); g = np.asarray(g)
    nugget0 = max(g.min(), 0.0)
    best = None
    for rng in np.linspace(h[h>0].min() if (h>0).any() else 1.0, h.max()*1.5, 30):
        for sill in np.linspace(g.max()*0.3, g.max()*1.5, 25):
            for nug in np.linspace(0, g.max()*0.6, 10):
                hh = np.minimum(h/rng, 1.0)
                model = nug + sill*(1.5*hh - 0.5*hh**3)
                err = np.sum((model - g)**2)
                if best is None or err < best[0]:
                    best = (err, nug, sill, rng)
    _, nug, sill, rng = best
    return dict(nugget=nug, sill=sill, rng=rng)

def _gamma(h, vp):
    hh = np.minimum(h/vp['rng'], 1.0)
    return vp['nugget'] + vp['sill']*(1.5*hh - 0.5*hh**3)

def eval_regression_kriging(df, feats):
    """RK: tendencia XGBoost global + kriging de residuales.
    Pesos OK calculados una vez por fold (geometria fija) y aplicados como suma ponderada por dia."""
    stations = sorted(df['estacion'].unique())
    rows = []
    for st in stations:
        tr = df[df['estacion'] != st].copy(); te = df[df['estacion'] == st].copy()
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].values); Xte = sc.transform(te[feats].values)
        trend = xgb_model(); trend.fit(Xtr, tr['pm25'].values)
        tr['resid'] = tr['pm25'].values - trend.predict(Xtr)
        te_trend = trend.predict(Xte)

        # localizaciones de las estaciones de entrenamiento
        locs = tr.groupby('estacion')[['lon','lat']].first()
        train_sts = list(locs.index)
        lon_s = locs['lon'].values; lat_s = locs['lat'].values
        tlon = te['lon'].iloc[0]; tlat = te['lat'].iloc[0]
        n = len(train_sts)

        # variograma empirico de residuales entre pares de estaciones
        piv = tr.pivot_table(index='date', columns='estacion', values='resid')[train_sts]
        hs, gs = [], []
        for i in range(n):
            for j in range(i+1, n):
                d = _haversine_km(lon_s[i],lat_s[i],lon_s[j],lat_s[j])
                diff = (piv.iloc[:,i] - piv.iloc[:,j]).dropna()
                if len(diff) > 10:
                    hs.append(d); gs.append(0.5*np.mean(diff.values**2))
        vp = _fit_spherical(hs, gs)

        # sistema OK: pesos de las n estaciones -> punto objetivo (geometria fija)
        G = np.zeros((n+1, n+1))
        for i in range(n):
            for j in range(n):
                dij = _haversine_km(lon_s[i],lat_s[i],lon_s[j],lat_s[j])
                G[i,j] = _gamma(dij, vp)
        G[:n, n] = 1.0; G[n, :n] = 1.0
        g0 = np.ones(n+1)
        for i in range(n):
            g0[i] = _gamma(_haversine_km(lon_s[i],lat_s[i],tlon,tlat), vp)
        try:
            sol = np.linalg.lstsq(G, g0, rcond=None)[0]
            lam = sol[:n]
        except Exception:
            lam = np.ones(n)/n

        # aplicar pesos por dia (renormalizar si faltan estaciones)
        R = piv.reindex(te['date'].values)  # dates x stations
        Rv = R.values  # (n_test, n)
        krig_resid = np.zeros(len(te))
        for k in range(len(te)):
            rv = Rv[k]; mask = ~np.isnan(rv)
            if mask.sum() == 0:
                krig_resid[k] = 0.0
            else:
                w = lam[mask]; sw = w.sum()
                w = w/sw if abs(sw) > 1e-9 else np.ones(mask.sum())/mask.sum()
                krig_resid[k] = np.dot(w, rv[mask])
        pred = te_trend + krig_resid
        yte = te['pm25'].values
        rows.append(dict(station=st, n_test=len(yte),
                         r2=r2_score(yte,pred), rmse=np.sqrt(mean_squared_error(yte,pred)),
                         mae=mean_absolute_error(yte,pred)))
    return pd.DataFrame(rows)

def summarize(name, res):
    v = res.dropna(subset=['r2'])
    w = (v['r2']*v['n_test']).sum()/v['n_test'].sum()
    return dict(model=name, r2_mean=v['r2'].mean(), r2_weighted=w, r2_median=v['r2'].median(),
                rmse_mean=v['rmse'].mean(), mae_mean=v['mae'].mean(),
                pct_pos=100*(v['r2']>0).mean())

def main():
    df = pd.read_csv(DATA, parse_dates=['date'])
    base = [c for c in df.columns if c not in EXCLUDE and c not in OSM]
    enhanced = base + OSM
    print(f"Filas: {len(df)} | estaciones: {df['estacion'].nunique()}")
    print(f"Base features ({len(base)}): {base}")
    print(f"Enhanced (+OSM): {OSM}\n")

    all_summ = []
    per_station = {}
    for fsname, feats in [('base', base), ('enhanced', enhanced)]:
        print(f"\n{'='*60}\nFEATURE SET: {fsname} ({len(feats)} feats)\n{'='*60}")
        for mname, res in [
            (f'Lasso[{fsname}]', eval_plain_model(df, feats, lambda: Lasso(alpha=1.0, random_state=42))),
            (f'XGBoost[{fsname}]', eval_plain_model(df, feats, xgb_model)),
            (f'RegKriging[{fsname}]', eval_regression_kriging(df, feats)),
        ]:
            s = summarize(mname, res); all_summ.append(s); per_station[mname] = res
            print(f"\n{mname}: R2_mean={s['r2_mean']:.3f}  R2_wt={s['r2_weighted']:.3f}  "
                  f"R2_med={s['r2_median']:.3f}  RMSE={s['rmse_mean']:.1f}  %pos={s['pct_pos']:.0f}")
            print(res.to_string(index=False))

    summ = pd.DataFrame(all_summ).sort_values('r2_mean', ascending=False)
    print(f"\n\n{'='*70}\nRESUMEN — ordenado por R² mean (cada estacion = 1 voto)\n{'='*70}")
    print(summ.to_string(index=False))
    summ.to_csv('data/processed/apr_1_2_spatial_summary.csv', index=False)
    best = summ.iloc[0]
    print(f"\nMEJOR R² espacial alcanzable: {best['model']} "
          f"-> R2_mean={best['r2_mean']:.3f}, R2_median={best['r2_median']:.3f}, "
          f"%estaciones R2>0 = {best['pct_pos']:.0f}%")
    # guardar per-station del mejor
    per_station[best['model']].to_csv('data/processed/apr_1_2_best_per_station.csv', index=False)

if __name__ == '__main__':
    main()
