"""
Data Exploration Script 1: Load and Summarize Downloaded Data
Loads all monthly CSV files and generates summary statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_sentinel5p_data():
    """Load all Sentinel-5P monthly files."""
    print("\n" + "="*80)
    print("LOADING SENTINEL-5P DATA")
    print("="*80)

    sentinel_dir = DATA_RAW / "sentinel5p"
    files = sorted(sentinel_dir.glob("sentinel5p_*.csv"))

    print(f"Found {len(files)} Sentinel-5P files")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined):,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Variables: {combined['variable'].unique()}")
    print(f"Unique locations: {combined[['lon', 'lat']].drop_duplicates().shape[0]}")

    return combined

def load_modis_data():
    """Load all MODIS monthly files."""
    print("\n" + "="*80)
    print("LOADING MODIS AOD DATA")
    print("="*80)

    modis_dir = DATA_RAW / "modis"
    files = sorted(modis_dir.glob("modis_aod_*.csv"))

    print(f"Found {len(files)} MODIS files")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined):,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Unique locations: {combined[['lon', 'lat']].drop_duplicates().shape[0]}")

    return combined

def load_era5_data():
    """Load all ERA5 monthly files."""
    print("\n" + "="*80)
    print("LOADING ERA5-LAND DATA")
    print("="*80)

    era5_dir = DATA_RAW / "era5"
    files = sorted(era5_dir.glob("era5_*.csv"))

    print(f"Found {len(files)} ERA5 files")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined):,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Variables: {combined.columns.tolist()}")
    print(f"Unique locations: {combined[['lon', 'lat']].drop_duplicates().shape[0]}")

    return combined

def generate_summary_stats(df, name, group_by='variable'):
    """Generate summary statistics for a dataset."""
    print(f"\n{name} - SUMMARY STATISTICS")
    print("-" * 80)

    if group_by in df.columns:
        # For Sentinel-5P (has 'variable' column)
        summary = df.groupby(group_by)['value'].describe()
        print(summary)

        # Count by variable
        counts = df.groupby(group_by).size()
        print(f"\nRecord counts by {group_by}:")
        print(counts)

        return summary
    else:
        # For MODIS and ERA5
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['lon', 'lat']]

        summary = df[numeric_cols].describe()
        print(summary)

        return summary

def check_missing_data(df, name):
    """Check for missing data."""
    print(f"\n{name} - MISSING DATA CHECK")
    print("-" * 80)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })

    print(missing_df[missing_df['Missing_Count'] > 0])

    if missing_df['Missing_Count'].sum() == 0:
        print("No missing data found!")

    return missing_df

def check_data_quality(df, name):
    """Check data quality issues."""
    print(f"\n{name} - DATA QUALITY CHECK")
    print("-" * 80)

    issues = {}

    # Check for duplicates
    duplicates = df.duplicated().sum()
    issues['duplicates'] = duplicates
    print(f"Duplicate rows: {duplicates}")

    # Check date format
    try:
        df['date'] = pd.to_datetime(df['date'])
        print(f"Date format: OK")
        issues['date_format'] = 'OK'
    except:
        print(f"Date format: ERROR")
        issues['date_format'] = 'ERROR'

    # Check coordinate ranges
    if 'lon' in df.columns and 'lat' in df.columns:
        lon_range = (df['lon'].min(), df['lon'].max())
        lat_range = (df['lat'].min(), df['lat'].max())
        print(f"Longitude range: {lon_range}")
        print(f"Latitude range: {lat_range}")

        # Santiago bbox: -71.0 to -70.4 (lon), -33.8 to -33.2 (lat)
        if lon_range[0] >= -71.1 and lon_range[1] <= -70.3:
            print("Longitude: Within expected range ✓")
            issues['lon_range'] = 'OK'
        else:
            print("Longitude: Outside expected range!")
            issues['lon_range'] = 'WARNING'

        if lat_range[0] >= -33.9 and lat_range[1] <= -33.1:
            print("Latitude: Within expected range ✓")
            issues['lat_range'] = 'OK'
        else:
            print("Latitude: Outside expected range!")
            issues['lat_range'] = 'WARNING'

    return issues

def save_summary_report(sentinel_summary, modis_summary, era5_summary,
                        sentinel_missing, modis_missing, era5_missing,
                        sentinel_quality, modis_quality, era5_quality):
    """Save comprehensive summary report."""

    # Convert to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    report = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {
            'sentinel5p': {
                'summary_stats': convert_to_native(sentinel_summary.to_dict()) if isinstance(sentinel_summary, pd.DataFrame) else {},
                'missing_data': convert_to_native(sentinel_missing.to_dict()),
                'quality_checks': convert_to_native(sentinel_quality)
            },
            'modis': {
                'summary_stats': convert_to_native(modis_summary.to_dict()) if isinstance(modis_summary, pd.DataFrame) else {},
                'missing_data': convert_to_native(modis_missing.to_dict()),
                'quality_checks': convert_to_native(modis_quality)
            },
            'era5': {
                'summary_stats': convert_to_native(era5_summary.to_dict()) if isinstance(era5_summary, pd.DataFrame) else {},
                'missing_data': convert_to_native(era5_missing.to_dict()),
                'quality_checks': convert_to_native(era5_quality)
            }
        }
    }

    # Save as JSON
    output_file = RESULTS_DIR / "data_summary_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Summary report saved to: {output_file}")
    print(f"{'='*80}")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PM2.5 SANTIAGO - DATA EXPLORATION")
    print("Script 1: Load and Summarize Downloaded Data")
    print("="*80)
    print(f"Execution time: {datetime.now()}")

    # Load data
    sentinel_df = load_sentinel5p_data()
    modis_df = load_modis_data()
    era5_df = load_era5_data()

    # Summary statistics
    sentinel_summary = generate_summary_stats(sentinel_df, "SENTINEL-5P", group_by='variable')
    modis_summary = generate_summary_stats(modis_df, "MODIS AOD", group_by=None)
    era5_summary = generate_summary_stats(era5_df, "ERA5-LAND", group_by=None)

    # Missing data check
    sentinel_missing = check_missing_data(sentinel_df, "SENTINEL-5P")
    modis_missing = check_missing_data(modis_df, "MODIS AOD")
    era5_missing = check_missing_data(era5_df, "ERA5-LAND")

    # Data quality check
    sentinel_quality = check_data_quality(sentinel_df, "SENTINEL-5P")
    modis_quality = check_data_quality(modis_df, "MODIS AOD")
    era5_quality = check_data_quality(era5_df, "ERA5-LAND")

    # Save report
    save_summary_report(sentinel_summary, modis_summary, era5_summary,
                       sentinel_missing, modis_missing, era5_missing,
                       sentinel_quality, modis_quality, era5_quality)

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)

    # Quick dataset info
    print(f"\nDataset Sizes:")
    print(f"  Sentinel-5P: {len(sentinel_df):,} records")
    print(f"  MODIS AOD:   {len(modis_df):,} records")
    print(f"  ERA5-Land:   {len(era5_df):,} records")
    print(f"  TOTAL:       {len(sentinel_df) + len(modis_df) + len(era5_df):,} records")

if __name__ == "__main__":
    main()
