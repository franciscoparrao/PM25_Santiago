#!/usr/bin/env python3
"""
Script para limpiar y consolidar datos PM2.5 de SINCA.

Procesa todos los archivos CSV descargados desde SINCA y los consolida
en un dataset maestro limpio con formato estándar.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def identify_station_from_filename(filepath):
    """
    Identifica la estación basándose en el rango de fechas del archivo.

    Los archivos tienen nombres como: datos_YYMMDD_YYMMDD.csv
    Usaremos un mapeo basado en las fechas de inicio conocidas.
    """
    filename = Path(filepath).name

    # Extraer fechas del nombre del archivo
    match = re.search(r'datos_(\d{6})_(\d{6})', filename)
    if not match:
        return None

    start_date = match.group(1)

    # Mapeo de fechas de inicio a estaciones (basado en los archivos descargados)
    station_mapping = {
        '970402': 'Parque O\'Higgins',
        '970403': 'Parque O\'Higgins',  # Duplicado
        '000101': 'Independencia',  # Hay 3 archivos con esta fecha
        '080509': 'El Bosque',
        '080510': 'Cerro Navia',
        '080519': 'Puente Alto',  # o Quilicura I (hay 2 archivos)
        '080521': 'Las Condes',
        '080526': 'Talagante',
        '160422': 'Pudahuel',
        '220401': 'Cerrillos II',  # o Cerrillos I (hay 2 archivos)
    }

    return station_mapping.get(start_date, f'Unknown_{start_date}')


def load_csv_file(filepath):
    """
    Carga un archivo CSV de SINCA con el formato correcto.

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con los datos limpios
    """
    logger.info(f"Cargando: {Path(filepath).name}")

    try:
        # Leer CSV con formato europeo (separador ; y decimal ,)
        df = pd.read_csv(
            filepath,
            sep=';',
            decimal=',',
            encoding='latin-1'  # Encoding para caracteres españoles
        )

        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()

        # Renombrar columnas a nombres más cortos
        column_mapping = {
            'FECHA (YYMMDD)': 'fecha',
            'HORA (HHMM)': 'hora',
            'Registros validados': 'pm25_validado',
            'Registros preliminares': 'pm25_preliminar',
            'Registros no validados': 'pm25_no_validado'
        }

        df = df.rename(columns=column_mapping)

        # Identificar estación
        station = identify_station_from_filename(filepath)
        df['estacion'] = station
        df['archivo'] = Path(filepath).name

        logger.info(f"  Estación identificada: {station}")
        logger.info(f"  Registros: {len(df)}")

        return df

    except Exception as e:
        logger.error(f"Error cargando {filepath}: {e}")
        return None


def parse_date(fecha_str, hora_str='0000'):
    """
    Convierte fecha YYMMDD y hora HHMM a datetime.

    Args:
        fecha_str: String con formato YYMMDD
        hora_str: String con formato HHMM (default 0000)

    Returns:
        datetime object o None si hay error
    """
    try:
        # Convertir a string si es necesario
        fecha_str = str(int(fecha_str)).zfill(6)
        hora_str = str(int(hora_str)).zfill(4) if pd.notna(hora_str) else '0000'

        yy = int(fecha_str[:2])
        mm = int(fecha_str[2:4])
        dd = int(fecha_str[4:6])

        hh = int(hora_str[:2])
        mi = int(hora_str[2:4])

        # Determinar siglo (00-50 = 2000s, 51-99 = 1900s)
        if yy <= 50:
            year = 2000 + yy
        else:
            year = 1900 + yy

        return datetime(year, mm, dd, hh, mi)
    except Exception as e:
        logger.debug(f"Error parseando fecha {fecha_str} {hora_str}: {e}")
        return None


def clean_dataframe(df):
    """
    Limpia un DataFrame de SINCA.

    Args:
        df: DataFrame con datos crudos

    Returns:
        DataFrame limpio
    """
    # Copiar para no modificar original
    df_clean = df.copy()

    # Parsear fechas
    df_clean['datetime'] = df_clean.apply(
        lambda row: parse_date(row['fecha'], row.get('hora', '0000')),
        axis=1
    )

    # Eliminar filas sin fecha válida
    invalid_dates = df_clean['datetime'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"  Eliminando {invalid_dates} registros con fechas inválidas")
        df_clean = df_clean[df_clean['datetime'].notna()]

    # Extraer componentes de fecha
    df_clean['year'] = df_clean['datetime'].dt.year
    df_clean['month'] = df_clean['datetime'].dt.month
    df_clean['day'] = df_clean['datetime'].dt.day
    df_clean['date'] = df_clean['datetime'].dt.date

    # Consolidar columnas PM2.5 (priorizar validado, luego preliminar)
    df_clean['pm25'] = df_clean['pm25_validado'].fillna(df_clean['pm25_preliminar'])

    # Flag para indicar si el dato es validado o preliminar
    df_clean['validado'] = df_clean['pm25_validado'].notna()

    # Ordenar por fecha
    df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

    return df_clean


def identify_duplicates(all_files):
    """
    Identifica archivos duplicados basándose en contenido.

    Args:
        all_files: Lista de paths a archivos CSV

    Returns:
        Diccionario con archivos a mantener y archivos duplicados
    """
    logger.info("\n" + "="*60)
    logger.info("IDENTIFICANDO DUPLICADOS")
    logger.info("="*60)

    file_data = {}

    for filepath in all_files:
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin-1', nrows=10)

        # Usar primeras filas como "firma" del archivo
        first_row = df.iloc[0].to_dict() if len(df) > 0 else {}
        num_rows = len(pd.read_csv(filepath, sep=';', decimal=',', encoding='latin-1'))

        file_data[filepath] = {
            'first_row': str(first_row),
            'num_rows': num_rows,
            'filename': Path(filepath).name
        }

    # Agrupar archivos similares
    groups = {}
    for filepath, data in file_data.items():
        key = (data['first_row'], data['num_rows'])
        if key not in groups:
            groups[key] = []
        groups[key].append(filepath)

    # Identificar duplicados
    duplicates = []
    keep = []

    for group in groups.values():
        if len(group) > 1:
            # Mantener el archivo sin (N) en el nombre
            primary = sorted(group, key=lambda x: '(' not in Path(x).name)[0]
            keep.append(primary)

            for dup in group:
                if dup != primary:
                    duplicates.append(dup)
                    logger.info(f"  Duplicado: {Path(dup).name} -> mantener {Path(primary).name}")
        else:
            keep.append(group[0])

    logger.info(f"\nArchivos únicos: {len(keep)}")
    logger.info(f"Duplicados encontrados: {len(duplicates)}")

    return {'keep': keep, 'duplicates': duplicates}


def main():
    """Función principal de procesamiento."""

    logger.info("\n" + "="*70)
    logger.info("LIMPIEZA Y CONSOLIDACIÓN DE DATOS SINCA PM2.5")
    logger.info("="*70)

    # Directorios
    data_dir = Path('data/external')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar todos los archivos CSV
    all_files = sorted(data_dir.glob('datos_*.csv'))
    logger.info(f"\nArchivos encontrados: {len(all_files)}")

    # Identificar y remover duplicados
    dup_result = identify_duplicates(all_files)
    files_to_process = dup_result['keep']

    logger.info(f"\nArchivos a procesar: {len(files_to_process)}")

    # Cargar y limpiar cada archivo
    logger.info("\n" + "="*60)
    logger.info("CARGANDO Y LIMPIANDO ARCHIVOS")
    logger.info("="*60)

    all_dataframes = []

    for filepath in files_to_process:
        df = load_csv_file(filepath)

        if df is not None:
            df_clean = clean_dataframe(df)
            all_dataframes.append(df_clean)

            # Mostrar resumen
            valid_count = df_clean['validado'].sum()
            prelim_count = (~df_clean['validado']).sum()
            pm25_count = df_clean['pm25'].notna().sum()

            logger.info(f"  Validados: {valid_count}, Preliminares: {prelim_count}, Total con PM2.5: {pm25_count}")

    # Consolidar todos los DataFrames
    logger.info("\n" + "="*60)
    logger.info("CONSOLIDANDO DATOS")
    logger.info("="*60)

    if not all_dataframes:
        logger.error("No se pudieron cargar datos!")
        return

    df_master = pd.concat(all_dataframes, ignore_index=True)

    logger.info(f"Total de registros: {len(df_master):,}")
    logger.info(f"Rango de fechas: {df_master['datetime'].min()} a {df_master['datetime'].max()}")
    logger.info(f"Estaciones únicas: {df_master['estacion'].nunique()}")

    # Estadísticas por estación
    logger.info("\n" + "="*60)
    logger.info("ESTADÍSTICAS POR ESTACIÓN")
    logger.info("="*60)

    station_stats = df_master.groupby('estacion').agg({
        'pm25': ['count', 'mean', 'std', 'min', 'max'],
        'datetime': ['min', 'max'],
        'validado': 'sum'
    }).round(2)

    print("\n", station_stats)

    # Guardar dataset consolidado
    output_file = output_dir / 'sinca_pm25_consolidated.csv'
    logger.info(f"\nGuardando dataset consolidado: {output_file}")

    # Seleccionar columnas finales
    final_columns = [
        'datetime', 'date', 'year', 'month', 'day',
        'estacion', 'pm25', 'validado',
        'pm25_validado', 'pm25_preliminar',
        'archivo'
    ]

    df_master[final_columns].to_csv(output_file, index=False, encoding='utf-8')

    logger.info(f"✓ Dataset guardado: {output_file}")
    logger.info(f"  Tamaño: {output_file.stat().st_size / 1024:.1f} KB")

    # Guardar resumen de estaciones
    summary_file = output_dir / 'sinca_stations_summary.csv'
    station_summary = df_master.groupby('estacion').agg({
        'datetime': ['min', 'max', 'count'],
        'pm25': ['count', 'mean', 'std'],
        'validado': 'sum'
    }).round(2)

    station_summary.to_csv(summary_file)
    logger.info(f"✓ Resumen de estaciones: {summary_file}")

    # Reporte de calidad
    logger.info("\n" + "="*60)
    logger.info("REPORTE DE CALIDAD")
    logger.info("="*60)

    total_records = len(df_master)
    records_with_pm25 = df_master['pm25'].notna().sum()
    validated_records = df_master['validado'].sum()

    logger.info(f"Total de registros: {total_records:,}")
    logger.info(f"Registros con PM2.5: {records_with_pm25:,} ({records_with_pm25/total_records*100:.1f}%)")
    logger.info(f"Registros validados: {validated_records:,} ({validated_records/records_with_pm25*100:.1f}% de los que tienen PM2.5)")
    logger.info(f"Registros preliminares: {records_with_pm25 - validated_records:,}")

    # Valores extremos
    if records_with_pm25 > 0:
        logger.info(f"\nEstadísticas PM2.5 (μg/m³):")
        logger.info(f"  Media: {df_master['pm25'].mean():.2f}")
        logger.info(f"  Mediana: {df_master['pm25'].median():.2f}")
        logger.info(f"  Std: {df_master['pm25'].std():.2f}")
        logger.info(f"  Min: {df_master['pm25'].min():.2f}")
        logger.info(f"  Max: {df_master['pm25'].max():.2f}")

    logger.info("\n" + "="*70)
    logger.info("✓✓✓ PROCESAMIENTO COMPLETADO EXITOSAMENTE ✓✓✓")
    logger.info("="*70)

    return df_master


if __name__ == "__main__":
    df = main()
