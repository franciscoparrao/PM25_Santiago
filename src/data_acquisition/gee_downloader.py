"""
Google Earth Engine Data Downloader
Downloads satellite data for PM2.5 prediction in Santiago, Chile
"""

import ee
import geemap
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GEEDataDownloader:
    """
    Download satellite data from Google Earth Engine for Santiago study area.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize GEE downloader.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._initialize_gee()
        self.study_area = self._define_study_area()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_gee(self):
        """Initialize Google Earth Engine."""
        try:
            ee.Initialize()
            logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GEE: {e}")
            logger.info("Please run: earthengine authenticate")
            raise

    def _define_study_area(self) -> ee.Geometry:
        """
        Define study area (Santiago Metropolitan Region) as ee.Geometry.

        Returns:
            ee.Geometry.Rectangle of study area
        """
        bbox = self.config['study_area']['bbox']
        geometry = ee.Geometry.Rectangle([
            bbox['west'], bbox['south'],
            bbox['east'], bbox['north']
        ])
        logger.info(f"Study area defined: {self.config['study_area']['name']}")
        return geometry

    def download_sentinel5p(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw/sentinel5p"
    ) -> pd.DataFrame:
        """
        Download Sentinel-5P TROPOMI data (NO2, SO2, CO, O3, AOD).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory

        Returns:
            DataFrame with extracted values
        """
        logger.info(f"Downloading Sentinel-5P data: {start_date} to {end_date}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        datasets = self.config['gee_datasets']['sentinel5p']

        for variable, info in datasets.items():
            logger.info(f"Processing {variable}...")

            try:
                # Load image collection
                collection = (ee.ImageCollection(info['collection'])
                            .filterDate(start_date, end_date)
                            .filterBounds(self.study_area))

                # Get image count
                count = collection.size().getInfo()
                logger.info(f"Found {count} images for {variable}")

                if count == 0:
                    logger.warning(f"No images found for {variable}")
                    continue

                # Calculate mean over time period
                mean_image = collection.select(info['band']).mean()

                # Sample the image at study area centroid
                sample = mean_image.sample(
                    region=self.study_area,
                    scale=info['scale'],
                    geometries=True
                )

                # Convert to list
                sample_list = sample.getInfo()['features']

                # Extract values
                for feature in sample_list:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']

                    results.append({
                        'variable': variable,
                        'value': props.get(info['band']),
                        'lon': coords[0],
                        'lat': coords[1],
                        'date': end_date  # Using end date as representative
                    })

                logger.info(f"Extracted {len(sample_list)} samples for {variable}")

            except Exception as e:
                logger.error(f"Error processing {variable}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        output_file = Path(output_dir) / f"sentinel5p_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved Sentinel-5P data to {output_file}")

        return df

    def download_modis_aod(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw/modis"
    ) -> pd.DataFrame:
        """
        Download MODIS AOD data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory

        Returns:
            DataFrame with extracted values
        """
        logger.info(f"Downloading MODIS AOD data: {start_date} to {end_date}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        dataset_info = self.config['gee_datasets']['modis']['aod']

        # Load collection
        collection = (ee.ImageCollection(dataset_info['collection'])
                    .filterDate(start_date, end_date)
                    .filterBounds(self.study_area))

        count = collection.size().getInfo()
        logger.info(f"Found {count} MODIS images")

        if count == 0:
            logger.warning("No MODIS images found")
            return pd.DataFrame()

        # Calculate mean
        mean_image = collection.select(dataset_info['band']).mean()

        # Sample
        sample = mean_image.sample(
            region=self.study_area,
            scale=dataset_info['scale'],
            geometries=True
        )

        # Extract to DataFrame
        sample_list = sample.getInfo()['features']

        results = []
        for feature in sample_list:
            props = feature['properties']
            coords = feature['geometry']['coordinates']

            results.append({
                'variable': 'modis_aod',
                'value': props.get(dataset_info['band']),
                'lon': coords[0],
                'lat': coords[1],
                'date': end_date
            })

        df = pd.DataFrame(results)

        # Save
        output_file = Path(output_dir) / f"modis_aod_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved MODIS AOD data to {output_file}")

        return df

    def download_era5(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw/era5"
    ) -> pd.DataFrame:
        """
        Download ERA5 meteorological data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory

        Returns:
            DataFrame with extracted values
        """
        logger.info(f"Downloading ERA5 data: {start_date} to {end_date}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        dataset_info = self.config['gee_datasets']['era5']

        # Load collection (ERA5-Land Hourly)
        collection = (ee.ImageCollection(dataset_info['collection'])
                    .filterDate(start_date, end_date)
                    .filterBounds(self.study_area))

        count = collection.size().getInfo()
        logger.info(f"Found {count} ERA5 images")

        if count == 0:
            logger.warning("No ERA5 data found")
            return pd.DataFrame()

        # Select bands and aggregate to daily mean
        # ERA5-Land is hourly, so we take the mean for the period
        bands = dataset_info['bands']
        mean_image = collection.select(bands).mean()

        # Sample
        sample = mean_image.sample(
            region=self.study_area,
            scale=dataset_info['scale'],
            geometries=True
        )

        # Extract to DataFrame
        sample_list = sample.getInfo()['features']

        results = []
        for feature in sample_list:
            props = feature['properties']
            coords = feature['geometry']['coordinates']

            row = {
                'lon': coords[0],
                'lat': coords[1],
                'date': end_date
            }

            # Add all meteorological variables
            for band in bands:
                row[band] = props.get(band)

            results.append(row)

        df = pd.DataFrame(results)

        # Save
        output_file = Path(output_dir) / f"era5_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved ERA5 data to {output_file}")

        return df

    def download_all_data(
        self,
        start_date: str,
        end_date: str,
        monthly: bool = True
    ):
        """
        Download all satellite data for the study period.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            monthly: If True, download data month by month to avoid memory issues
        """
        logger.info(f"Starting full data download: {start_date} to {end_date}")

        if not monthly:
            # Download all at once
            self.download_sentinel5p(start_date, end_date)
            self.download_modis_aod(start_date, end_date)
            self.download_era5(start_date, end_date)
        else:
            # Download month by month
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            current = start
            while current < end:
                # Calculate end of month
                next_month = current.replace(day=28) + timedelta(days=4)
                month_end = next_month - timedelta(days=next_month.day)

                if month_end > end:
                    month_end = end

                month_start_str = current.strftime("%Y-%m-%d")
                month_end_str = month_end.strftime("%Y-%m-%d")

                logger.info(f"\n{'='*60}")
                logger.info(f"Processing period: {month_start_str} to {month_end_str}")
                logger.info(f"{'='*60}\n")

                try:
                    self.download_sentinel5p(month_start_str, month_end_str)
                    time.sleep(5)  # Avoid rate limits

                    self.download_modis_aod(month_start_str, month_end_str)
                    time.sleep(5)

                    self.download_era5(month_start_str, month_end_str)
                    time.sleep(5)

                except Exception as e:
                    logger.error(f"Error processing {month_start_str}: {e}")
                    continue

                # Move to next month
                current = month_end + timedelta(days=1)

        logger.info("Data download completed!")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download GEE data for PM2.5 Santiago")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--start-date", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-11-10", help="End date (YYYY-MM-DD)")
    parser.add_argument("--monthly", action="store_true", help="Download month by month")

    args = parser.parse_args()

    # Initialize downloader
    downloader = GEEDataDownloader(config_path=args.config)

    # Download data
    downloader.download_all_data(
        start_date=args.start_date,
        end_date=args.end_date,
        monthly=args.monthly
    )


if __name__ == "__main__":
    main()
