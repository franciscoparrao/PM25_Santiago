#!/usr/bin/env python3
"""
APR review b3 Issue #1 — Inferencia robusta a autocorrelacion.

Las series diarias de PM2.5 y meteorologia estan fuertemente autocorrelacionadas, de modo que
los p-values Pearson con n nominal sobreestiman la significancia. Aqui recomputamos:

(A) Correlaciones de ventilacion (log VC vs PM2.5 nivel; d logVC vs dPM2.5 en transicion) con
    - tamano de muestra EFECTIVO (ajuste por autocorrelacion lag-1 de ambas series)
    - moving-block bootstrap (preserva la dependencia serial) -> IC 95% del r
(B) Diebold-Mariano XGBoost vs baselines con varianza HAC (Newey-West) de lag largo
    (bandwidth automatico), no lag-1, sobre el diferencial de perdida ordenado en el tiempo.
Salida: data/processed/apr_b3_robust_inference.csv
"""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(42)

def lag1(x):
    x = np.asarray(x, float); x = x - x.mean()
    return np.sum(x[1:]*x[:-1])/np.sum(x*x)

def n_eff(x, y):
    """tamano efectivo para correlacion entre dos series autocorrelacionadas (AR1 approx)."""
    rx, ry = lag1(x), lag1(y); n = len(x)
    factor = (1 - rx*ry)/(1 + rx*ry)
    return max(n*factor, 3.0)

def p_from_r(r, n):
    t = r*np.sqrt((n-2)/max(1-r*r,1e-12))
    return 2*stats.t.sf(abs(t), df=n-2)

def block_bootstrap_corr(x, y, L, B=3000):
    x = np.asarray(x,float); y = np.asarray(y,float); n = len(x)
    nb = int(np.ceil(n/L)); rs = np.empty(B)
    starts_max = n - L
    for b in range(B):
        idx = []
        for _ in range(nb):
            s = rng.integers(0, starts_max+1); idx.extend(range(s, s+L))
        idx = np.array(idx[:n])
        rs[b] = np.corrcoef(x[idx], y[idx])[0,1]
    return np.percentile(rs,2.5), np.percentile(rs,97.5), rs

def dm_hac(d):
    """Diebold-Mariano con varianza HAC Newey-West, bandwidth automatico."""
    d = np.asarray(d,float); n = len(d); dbar = d.mean()
    m = int(np.floor(4*(n/100)**(2/9)))  # regla estandar de bandwidth
    g0 = np.mean((d-dbar)**2); var = g0
    for k in range(1, m+1):
        gk = np.mean((d[k:]-dbar)*(d[:-k]-dbar))
        w = 1 - k/(m+1)
        var += 2*w*gk
    se = np.sqrt(var/n); stat = dbar/se
    return stat, 2*stats.norm.sf(abs(stat)), m

def main():
    out=[]
    # ---------- (A) ventilacion ----------
    v = pd.read_csv('data/processed/santiago_ventilation_raw.csv', parse_dates=['date'])
    v['wspd']=np.hypot(v['u'],v['v']); v['VC']=v['blh']*v['wspd']
    v['logVC']=np.log(v['VC'].clip(lower=1))
    pm=pd.read_csv('data/processed/sinca_features_spatial.csv', parse_dates=['date'])
    city=pm.groupby('date').agg(pm25=('pm25','mean'),
                                precip=('era5_total_precipitation_hourly','mean')).reset_index()
    m=city.merge(v[['date','logVC']],on='date',how='inner').sort_values('date').reset_index(drop=True)

    # nivel
    d1=m.dropna(subset=['logVC','pm25'])
    r=np.corrcoef(d1['logVC'],d1['pm25'])[0,1]; ne=n_eff(d1['logVC'].values,d1['pm25'].values)
    lo,hi,_=block_bootstrap_corr(d1['logVC'].values,d1['pm25'].values,L=30)
    print(f"(A1) log VC vs PM2.5: r={r:.3f} | n={len(d1)} n_eff={ne:.0f} | p_eff={p_from_r(r,ne):.2e} "
          f"| bootstrap95%CI=[{lo:.3f},{hi:.3f}]")
    out.append(dict(test='corr_logVC_pm25', r=round(r,3), n=len(d1), n_eff=round(ne),
                    p_eff=p_from_r(r,ne), ci_lo=round(lo,3), ci_hi=round(hi,3)))

    # transicion (dif)
    m['dPM']=m['pm25'].diff(); m['dlogVC']=m['logVC'].diff()
    mm=m.dropna(subset=['dPM','dlogVC']); thr=mm['dPM'].abs().quantile(0.88)
    tr=mm[mm['dPM'].abs()>thr]
    r=np.corrcoef(tr['dlogVC'],tr['dPM'])[0,1]; ne=n_eff(tr['dlogVC'].values,tr['dPM'].values)
    lo,hi,_=block_bootstrap_corr(tr['dlogVC'].values,tr['dPM'].values,L=10)
    print(f"(A2) d logVC vs dPM (transicion): r={r:.3f} | n={len(tr)} n_eff={ne:.0f} | "
          f"p_eff={p_from_r(r,ne):.2e} | bootstrap95%CI=[{lo:.3f},{hi:.3f}]")
    out.append(dict(test='corr_dlogVC_dPM_transition', r=round(r,3), n=len(tr), n_eff=round(ne),
                    p_eff=p_from_r(r,ne), ci_lo=round(lo,3), ci_hi=round(hi,3)))

    # ---------- (B) Diebold-Mariano HAC ----------
    p=pd.read_csv('data/processed/apr_1_3_predictions_1d.csv', parse_dates=['date']).sort_values(['date'])
    a=p['pm25_real'].values
    for base in ['persistence','arima','prophet']:
        mask=~np.isnan(p[base].values)
        d=np.abs(a[mask]-p['xgboost'].values[mask])-np.abs(a[mask]-p[base].values[mask])
        stat,pv,m_=dm_hac(d)
        print(f"(B) DM XGBoost vs {base}: stat={stat:.2f}, p_HAC={pv:.2e} (lag={m_}, n={mask.sum()})")
        out.append(dict(test=f'DM_xgb_vs_{base}', r=round(stat,2), n=int(mask.sum()),
                        n_eff=m_, p_eff=pv, ci_lo=None, ci_hi=None))

    pd.DataFrame(out).to_csv('data/processed/apr_b3_robust_inference.csv', index=False)
    print("\nGuardado: data/processed/apr_b3_robust_inference.csv")

if __name__=='__main__':
    main()
