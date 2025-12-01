"""
Data Exploration Script 2: Visualize Satellite Data
Creates visualizations of spatial and temporal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
FIGURES_DIR = BASE_DIR / "results" / "figures" / "exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all datasets quickly."""
    print("Loading data...")

    # Load Sentinel-5P
    sentinel_files = sorted((DATA_RAW / "sentinel5p").glob("sentinel5p_*.csv"))
    sentinel_df = pd.concat([pd.read_csv(f) for f in sentinel_files], ignore_index=True)
    sentinel_df['date'] = pd.to_datetime(sentinel_df['date'])

    # Load MODIS
    modis_files = sorted((DATA_RAW / "modis").glob("modis_aod_*.csv"))
    modis_df = pd.concat([pd.read_csv(f) for f in modis_files], ignore_index=True)
    modis_df['date'] = pd.to_datetime(modis_df['date'])

    # Load ERA5
    era5_files = sorted((DATA_RAW / "era5").glob("era5_*.csv"))
    era5_df = pd.concat([pd.read_csv(f) for f in era5_files], ignore_index=True)
    era5_df['date'] = pd.to_datetime(era5_df['date'])

    print(f"Loaded {len(sentinel_df):,} Sentinel-5P records")
    print(f"Loaded {len(modis_df):,} MODIS records")
    print(f"Loaded {len(era5_df):,} ERA5 records")

    return sentinel_df, modis_df, era5_df

def plot_temporal_coverage(sentinel_df, modis_df, era5_df):
    """Plot temporal coverage of all datasets."""
    print("\nCreating temporal coverage plot...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Count records per month
    sentinel_monthly = sentinel_df.groupby(pd.Grouper(key='date', freq='M')).size()
    modis_monthly = modis_df.groupby(pd.Grouper(key='date', freq='M')).size()
    era5_monthly = era5_df.groupby(pd.Grouper(key='date', freq='M')).size()

    ax.plot(sentinel_monthly.index, sentinel_monthly.values, label='Sentinel-5P', marker='o', markersize=3)
    ax.plot(modis_monthly.index, modis_monthly.values, label='MODIS AOD', marker='s', markersize=3)
    ax.plot(era5_monthly.index, era5_monthly.values, label='ERA5-Land', marker='^', markersize=3)

    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Records per Month')
    ax.set_title('Temporal Coverage of Satellite Datasets (2019-2025)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_temporal_coverage.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 01_temporal_coverage.png")

def plot_sentinel5p_variables(sentinel_df):
    """Plot Sentinel-5P variables over time."""
    print("\nCreating Sentinel-5P time series plots...")

    # Pivot data: average by date and variable
    pivot = sentinel_df.pivot_table(index='date', columns='variable', values='value', aggfunc='mean')

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    variables = ['no2', 'so2', 'co', 'o3', 'aod']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (var, color) in enumerate(zip(variables, colors)):
        if var in pivot.columns:
            ax = axes[idx]
            ax.plot(pivot.index, pivot[var], color=color, linewidth=1.5, alpha=0.8)
            ax.set_title(var.upper(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Sentinel-5P Variables - Temporal Evolution (2019-2025)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_sentinel5p_timeseries.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 02_sentinel5p_timeseries.png")

def plot_spatial_distribution(sentinel_df, modis_df, era5_df):
    """Plot spatial distribution of measurement points."""
    print("\nCreating spatial distribution plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sentinel-5P
    unique_points = sentinel_df[['lon', 'lat']].drop_duplicates()
    sample = unique_points.sample(min(1000, len(unique_points)))
    axes[0].scatter(sample['lon'], sample['lat'], alpha=0.5, s=10, c='#e74c3c')
    axes[0].set_title('Sentinel-5P\nSampling Points', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, alpha=0.3)

    # MODIS
    unique_points = modis_df[['lon', 'lat']].drop_duplicates()
    sample = unique_points.sample(min(1000, len(unique_points)))
    axes[1].scatter(sample['lon'], sample['lat'], alpha=0.5, s=10, c='#3498db')
    axes[1].set_title('MODIS AOD\nSampling Points', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(True, alpha=0.3)

    # ERA5
    sample = era5_df[['lon', 'lat']].drop_duplicates()
    axes[2].scatter(sample['lon'], sample['lat'], alpha=0.7, s=20, c='#2ecc71')
    axes[2].set_title('ERA5-Land\nSampling Points', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Spatial Distribution of Satellite Data over Santiago',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_spatial_distribution.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 03_spatial_distribution.png")

def plot_era5_meteorology(era5_df):
    """Plot ERA5 meteorological variables."""
    print("\nCreating ERA5 meteorology plots...")

    # Select key variables
    met_vars = ['temperature_2m', 'surface_pressure', 'total_precipitation_hourly']
    era5_monthly = era5_df.groupby(pd.Grouper(key='date', freq='M'))[met_vars].mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Temperature
    axes[0].plot(era5_monthly.index, era5_monthly['temperature_2m'] - 273.15,
                 color='#e74c3c', linewidth=2)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('2m Temperature', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Pressure
    axes[1].plot(era5_monthly.index, era5_monthly['surface_pressure'] / 100,
                 color='#3498db', linewidth=2)
    axes[1].set_ylabel('Pressure (hPa)')
    axes[1].set_title('Surface Pressure', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Precipitation
    axes[2].plot(era5_monthly.index, era5_monthly['total_precipitation_hourly'] * 1000,
                 color='#2ecc71', linewidth=2)
    axes[2].set_ylabel('Precipitation (mm/h)')
    axes[2].set_title('Total Precipitation (Hourly)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('ERA5-Land Meteorological Variables (2019-2025)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_era5_meteorology.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 04_era5_meteorology.png")

def plot_seasonal_patterns(sentinel_df):
    """Plot seasonal patterns in pollutants."""
    print("\nCreating seasonal pattern plots...")

    # Add month column
    sentinel_df['month'] = sentinel_df['date'].dt.month

    # Calculate monthly averages by variable
    monthly_avg = sentinel_df.groupby(['month', 'variable'])['value'].mean().reset_index()

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    variables = ['no2', 'so2', 'co', 'o3', 'aod']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[idx]
        data = monthly_avg[monthly_avg['variable'] == var]
        ax.plot(data['month'], data['value'], marker='o', color=color, linewidth=2, markersize=6)
        ax.set_title(var.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Value')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Seasonal Patterns in Air Pollutants (Monthly Averages)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_seasonal_patterns.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 05_seasonal_patterns.png")

def plot_data_availability_heatmap(sentinel_df, modis_df, era5_df):
    """Create heatmap showing data availability over time."""
    print("\nCreating data availability heatmap...")

    # Create monthly counts
    sentinel_monthly = sentinel_df.groupby(pd.Grouper(key='date', freq='M')).size()
    modis_monthly = modis_df.groupby(pd.Grouper(key='date', freq='M')).size()
    era5_monthly = era5_df.groupby(pd.Grouper(key='date', freq='M')).size()

    # Combine into DataFrame
    availability = pd.DataFrame({
        'Sentinel-5P': sentinel_monthly,
        'MODIS AOD': modis_monthly,
        'ERA5-Land': era5_monthly
    }).fillna(0)

    # Normalize by max for each dataset
    availability_norm = availability / availability.max()

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(availability_norm.T, cmap='YlGnBu', cbar_kws={'label': 'Normalized Availability'},
                ax=ax, linewidths=0.5, linecolor='white')
    ax.set_xlabel('Month')
    ax.set_ylabel('Dataset')
    ax.set_title('Data Availability Heatmap (2019-2025)', fontsize=14, fontweight='bold')

    # Improve x-axis labels
    tick_positions = range(0, len(availability), 6)
    tick_labels = [availability.index[i].strftime('%Y-%m') for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_data_availability.png', bbox_inches='tight')
    plt.close()

    print(f"  Saved: 06_data_availability.png")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PM2.5 SANTIAGO - DATA VISUALIZATION")
    print("Script 2: Visualize Spatial and Temporal Patterns")
    print("="*80)
    print(f"Execution time: {datetime.now()}")

    # Load data
    sentinel_df, modis_df, era5_df = load_data()

    # Create visualizations
    plot_temporal_coverage(sentinel_df, modis_df, era5_df)
    plot_sentinel5p_variables(sentinel_df)
    plot_spatial_distribution(sentinel_df, modis_df, era5_df)
    plot_era5_meteorology(era5_df)
    plot_seasonal_patterns(sentinel_df)
    plot_data_availability_heatmap(sentinel_df, modis_df, era5_df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
