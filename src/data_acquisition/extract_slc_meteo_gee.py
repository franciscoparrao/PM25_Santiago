#!/usr/bin/env python3
"""
APR review Issue #2 — Meteo ERA5-Land diaria para los sitios de Salt Lake City (GEE).
Extrae u/v viento 10m, precipitacion, temperatura 2m diarios por sitio.
Salida: data/external/slc_meteo_daily.csv
"""
import ee, time
import pandas as pd

SITES = {  # sitios con buena cobertura
    'Hawthorne':     (-111.872222, 40.736389),
    'Copper View':   (-111.894167, 40.598056),
    'Herriman #3':   (-112.036298, 40.496392),
    'Near Road':     (-111.901851, 40.662961),
    'ROSE PARK':     (-111.931000, 40.784220),
}
YEARS = list(range(2019, 2024))
BANDS = ['u_component_of_wind_10m','v_component_of_wind_10m',
         'total_precipitation_sum','temperature_2m']

def main():
    ee.Initialize()
    rows=[]; t0=time.time()
    for site,(lon,lat) in SITES.items():
        pt=ee.Geometry.Point([lon,lat])
        for yr in YEARS:
            col=(ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                 .select(BANDS).filterDate(f'{yr}-01-01', f'{yr+1}-01-01'))
            def f(img):
                v=img.reduceRegion(ee.Reducer.first(), pt, 11132)
                return ee.Feature(None, v.set('d', img.date().format('YYYY-MM-dd')))
            data=col.map(f).getInfo()
            for ft in data['features']:
                p=ft['properties']
                rows.append(dict(site=site, date=p.get('d'),
                                 u10=p.get('u_component_of_wind_10m'),
                                 v10=p.get('v_component_of_wind_10m'),
                                 precip=p.get('total_precipitation_sum'),
                                 t2m=p.get('temperature_2m')))
            print(f'  {site} {yr}: {len(data["features"])} dias ({time.time()-t0:.0f}s)')
    df=pd.DataFrame(rows).dropna(subset=['date'])
    df['date']=pd.to_datetime(df['date'])
    df.to_csv('data/external/slc_meteo_daily.csv', index=False)
    print(f'\nGuardado data/external/slc_meteo_daily.csv: {len(df)} filas')

if __name__=='__main__':
    main()
