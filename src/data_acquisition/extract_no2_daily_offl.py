#!/usr/bin/env python3
"""
APR 2.1 — Re-extraccion de NO2 diario (Sentinel-5P OFFL) para las 8 estaciones SINCA.

- Producto: COPERNICUS/S5P/OFFL/L3_NO2 (reprocesado; reemplaza el NRTI y el composite MENSUAL).
- Banda: tropospheric_NO2_column_number_density (mol/m2), qa via el producto OFFL.
- Salida: data/processed/no2_daily_offl.csv  (columnas: date, estacion, s5p_no2_daily)
Extrae por estacion en chunks anuales (getInfo manejable). Multiples orbitas/dia -> media diaria.
"""
import ee, time
import pandas as pd

STATIONS = {
    'Cerrillos II':     (-70.71, -33.50),
    'Cerro Navia':      (-70.74, -33.42),
    'El Bosque':        (-70.69, -33.56),
    'Independencia':    (-70.66, -33.41),
    'Las Condes':       (-70.58, -33.40),
    "Parque O'Higgins": (-70.65, -33.46),
    'Pudahuel':         (-70.77, -33.44),
    'Talagante':        (-70.93, -33.66),
}
YEARS = list(range(2019, 2026))
BAND = 'tropospheric_NO2_column_number_density'
SCALE = 7000

def main():
    ee.Initialize()
    rows = []
    t0 = time.time()
    for st, (lon, lat) in STATIONS.items():
        pt = ee.Geometry.Point([lon, lat])
        for yr in YEARS:
            col = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
                   .select(BAND).filterBounds(pt)
                   .filterDate(f'{yr}-01-01', f'{yr+1}-01-01'))
            def f(img):
                v = img.reduceRegion(ee.Reducer.mean(), pt, SCALE).get(BAND)
                return ee.Feature(None, {'d': img.date().format('YYYY-MM-dd'), 'no2': v})
            try:
                data = col.map(f).filter(ee.Filter.notNull(['no2'])).getInfo()
            except Exception as e:
                print(f'  {st} {yr}: ERROR {str(e)[:60]}'); continue
            for ft in data['features']:
                rows.append({'date': ft['properties']['d'], 'estacion': st,
                             's5p_no2_daily': ft['properties']['no2']})
            print(f'  {st} {yr}: {len(data["features"])} obs  ({time.time()-t0:.0f}s)')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    # multiples orbitas mismo dia -> media diaria
    df = df.groupby(['estacion','date'], as_index=False)['s5p_no2_daily'].mean()
    df = df.sort_values(['estacion','date'])
    out = 'data/processed/no2_daily_offl.csv'
    df.to_csv(out, index=False)
    print(f'\nGuardado {out}: {len(df)} filas, {df["estacion"].nunique()} estaciones')
    # cobertura por estacion
    cov = df.groupby('estacion')['date'].agg(['count','min','max'])
    print(cov.to_string())

if __name__ == '__main__':
    main()
