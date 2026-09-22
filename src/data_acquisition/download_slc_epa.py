#!/usr/bin/env python3
"""
APR review Issue #2 — Segunda cuenca: Salt Lake City (Utah), analogo de inversion.
Descarga PM2.5 diario (parametro 88101) de EPA AQS airdata (publico, sin API key),
filtra a Salt Lake County (state 49, county 035) y construye serie diaria por sitio.
Salida: data/external/slc_pm25_daily.csv (date, site, lat, lon, pm25)
"""
import io, zipfile, urllib.request
import numpy as np, pandas as pd
from pathlib import Path

YEARS = range(2019, 2024)  # 2019-2023 (5 anios, ~ como Santiago)
STATE, COUNTY = 49, 35     # Salt Lake County, Utah
OUT = Path('data/external'); OUT.mkdir(parents=True, exist_ok=True)

def fetch_year(yr):
    url = f'https://aqs.epa.gov/aqsweb/airdata/daily_88101_{yr}.zip'
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = z.namelist()[0]
    df = pd.read_csv(z.open(name), low_memory=False)
    df = df[(df['State Code'].astype(str).str.zfill(2)=='49') &
            (df['County Code'].astype(int)==COUNTY)]
    return df

def main():
    frames = []
    for yr in YEARS:
        d = fetch_year(yr)
        print(f'{yr}: {len(d)} filas Salt Lake County | duraciones: {sorted(d["Sample Duration"].unique())}')
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    print('\nSitios:', sorted(raw['Local Site Name'].dropna().unique())[:15])
    print('Sample Duration counts:'); print(raw['Sample Duration'].value_counts())

    # preferir duracion horaria promediada a diario ('1 HOUR'); fallback 24 HOUR
    dur = '1 HOUR' if (raw['Sample Duration']=='1 HOUR').sum() > 1000 else '24 HOUR'
    d = raw[raw['Sample Duration']==dur].copy()
    d['date'] = pd.to_datetime(d['Date Local'])
    d['site'] = d['Local Site Name'].fillna(d['Site Num'].astype(str))
    # media diaria por sitio (colapsa POCs)
    daily = (d.groupby(['site','date'])
               .agg(pm25=('Arithmetic Mean','mean'),
                    lat=('Latitude','first'), lon=('Longitude','first'))
               .reset_index())
    daily = daily[daily['pm25'] > -5]  # quitar invalidos
    daily = daily.sort_values(['site','date'])
    print(f'\nDuracion usada: {dur} | filas diarias: {len(daily)} | sitios: {daily.site.nunique()}')
    print(daily.groupby('site').agg(n=('date','count'), lat=('lat','first'),
                                    lon=('lon','first'), pm25_mean=('pm25','mean')).to_string())
    daily.to_csv(OUT/'slc_pm25_daily.csv', index=False)
    print(f'\nGuardado: {OUT}/slc_pm25_daily.csv')

if __name__ == '__main__':
    main()
