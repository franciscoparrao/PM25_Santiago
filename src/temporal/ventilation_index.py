#!/usr/bin/env python3
"""
APR review Issue #5 — Cuantificar el vinculo ventilacion <-> dinamica de dias de transicion.

La ventilacion de cuenca (altura de capa limite x viento de transporte) es una cantidad de
escala de cuenca; usamos un punto central de Santiago. Extrae de ERA5 (GEE) BLH y viento 10m
a las 18 UTC (~14 local, maximo de mezcla vespertina) por MES (evita el limite de memoria de
GEE sobre la coleccion horaria), calcula el coeficiente de ventilacion VC = BLH * wind_speed
(metrica estandar de dispersion) y lo relaciona con el PM2.5 promedio-ciudad:
  (1) corr( log VC , nivel PM2.5 )  -> esperado negativo (dias estancados acumulan)
  (2) corr( d(logVC) , dPM2.5 ) en dias de transicion (top ~12% |dPM|)
  (3) fraccion de varianza de dPM en transicion explicada por ventilacion (+precip)
Salida: data/processed/apr_5_ventilation.csv
"""
import os, time
import numpy as np, pandas as pd
from scipy import stats

CENTER=(-70.66,-33.42)   # Santiago centro (Independencia)
YEARS=list(range(2019,2026))
RAW='data/processed/santiago_ventilation_raw.csv'

def extract():
    import ee; ee.Initialize()
    BANDS=['boundary_layer_height','u_component_of_wind_10m','v_component_of_wind_10m']
    pt=ee.Geometry.Point(list(CENTER)); rows=[]; t0=time.time()
    for yr in YEARS:
        for mo in range(1,13):
            m2=mo%12+1; y2=yr+(1 if mo==12 else 0)
            col=(ee.ImageCollection('ECMWF/ERA5/HOURLY').select(BANDS).filterBounds(pt)
                 .filterDate(f'{yr}-{mo:02d}-01', f'{y2}-{m2:02d}-01')
                 .filter(ee.Filter.calendarRange(18,18,'hour')))
            def f(img):
                v=img.reduceRegion(ee.Reducer.first(), pt, 27830)
                return ee.Feature(None, v.set('d', img.date().format('YYYY-MM-dd')))
            try:
                data=col.map(f).getInfo()
            except Exception as e:
                print(f'  {yr}-{mo:02d} ERR {str(e)[:50]}'); continue
            for ft in data['features']:
                p=ft['properties']
                rows.append(dict(date=p.get('d'), blh=p.get('boundary_layer_height'),
                                 u=p.get('u_component_of_wind_10m'), v=p.get('v_component_of_wind_10m')))
        # guardado incremental por anio (robustez ante interrupciones)
        pd.DataFrame(rows).to_csv(RAW, index=False)
        print(f'  {yr}: acumulado {len(rows)} dias ({time.time()-t0:.0f}s)')
    df=pd.DataFrame(rows).dropna(subset=['date']); df['date']=pd.to_datetime(df['date'])
    df=df.drop_duplicates('date').sort_values('date')
    df.to_csv(RAW, index=False)
    return df

def main():
    v = pd.read_csv(RAW, parse_dates=['date']) if os.path.exists(RAW) else extract()
    v['wspd']=np.hypot(v['u'],v['v']); v['VC']=v['blh']*v['wspd']
    v['logVC']=np.log(v['VC'].clip(lower=1))

    pm=pd.read_csv('data/processed/sinca_features_spatial.csv', parse_dates=['date'])
    city=pm.groupby('date').agg(pm25=('pm25','mean'),
                                precip=('era5_total_precipitation_hourly','mean')).reset_index()
    m=city.merge(v[['date','blh','wspd','VC','logVC']], on='date', how='inner').sort_values('date')
    print(f"Merged: {len(m)} dias | BLH medio {m.blh.mean():.0f} m | VC medio {m.VC.mean():.0f} m2/s")
    print(f"BLH invierno(JJA) {m[m.date.dt.month.isin([6,7,8])].blh.mean():.0f} vs verano(DJF) "
          f"{m[m.date.dt.month.isin([12,1,2])].blh.mean():.0f} m")

    d1=m.dropna(subset=['logVC','pm25'])
    r_lvl,p_lvl=stats.pearsonr(d1['logVC'], d1['pm25'])
    print(f"\n(1) corr(log VC, PM2.5 nivel) = {r_lvl:.3f} (p={p_lvl:.1e}, n={len(d1)})")

    m['dPM']=m['pm25'].diff(); m['dlogVC']=m['logVC'].diff()
    mm=m.dropna(subset=['dPM','dlogVC'])
    thr=mm['dPM'].abs().quantile(0.88); tr=mm[mm['dPM'].abs()>thr]
    r_all,_=stats.pearsonr(mm['dlogVC'],mm['dPM']); r_tr,p_tr=stats.pearsonr(tr['dlogVC'],tr['dPM'])
    print(f"\n(2) corr(d logVC, dPM2.5): todos r={r_all:.3f} (n={len(mm)}) | "
          f"transicion(|dPM|>{thr:.1f}) r={r_tr:.3f} (p={p_tr:.1e}, n={len(tr)})")

    tr2=tr.dropna(subset=['dlogVC','precip'])
    X=np.column_stack([tr2['dlogVC'].values, tr2['precip'].values, np.ones(len(tr2))]); y=tr2['dPM'].values
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None); yhat=X@beta
    R2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    print(f"(3) dPM ~ d(logVC)+precip en transicion: R2={R2:.3f} "
          f"(coef d logVC={beta[0]:.1f} ug/m3, n={len(tr2)})")

    pd.DataFrame([dict(metric='corr_logVC_pm25_level',value=round(r_lvl,3),n=len(d1)),
                  dict(metric='corr_dlogVC_dPM_all',value=round(r_all,3),n=len(mm)),
                  dict(metric='corr_dlogVC_dPM_transition',value=round(r_tr,3),n=len(tr)),
                  dict(metric='R2_dPM_vent+precip_transition',value=round(R2,3),n=len(tr2)),
                  dict(metric='blh_winter_m',value=round(m[m.date.dt.month.isin([6,7,8])].blh.mean()),n=0),
                 ]).to_csv('data/processed/apr_5_ventilation.csv', index=False)
    print('\nGuardado: data/processed/apr_5_ventilation.csv')

if __name__=='__main__':
    main()
