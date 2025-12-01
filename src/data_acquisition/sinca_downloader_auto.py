"""
Automatic SINCA Data Downloader
Downloads PM2.5 data from Chilean air quality monitoring network using web scraping.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import List, Dict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sinca_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SINCAAutoDownloader:
    """
    Automatic downloader for SINCA data using their JSON API.
    """

    def __init__(self):
        self.base_url = "https://sinca.mma.gob.cl"
        self.api_endpoint = f"{self.base_url}/index.php/json/listadomapa2k19"

        # Santiago stations (approximate IDs - need to verify)
        self.stations = {
            'Cerrillos': 'CE',
            'Cerro Navia': 'CN',
            'El Bosque': 'EB',
            'Independencia': 'IN',
            'La Florida': 'LF',
            'Las Condes': 'LC',
            'Pudahuel': 'PU',
            'Puente Alto': 'PA',
            'Quilicura': 'QU',
            'Santiago Centro': 'SC',
            'Talagante': 'TA',
            'Parque O\'Higgins': 'PO'
        }

    def fetch_recent_data(self) -> pd.DataFrame:
        """
        Fetch recent data (last 48 hours) from SINCA JSON API.

        Returns:
            DataFrame with recent measurements
        """
        logger.info("Fetching recent data from SINCA API...")

        try:
            # Try to get data from JSON endpoint
            response = requests.get(
                self.api_endpoint,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            logger.info(f"API response received: {len(data)} stations")

            # Parse JSON structure
            records = []
            for station_data in data:
                if 'realtime' in station_data:
                    for measurement in station_data['realtime']:
                        if measurement.get('parameter') == 'MP2.5':
                            records.append({
                                'station': station_data.get('nombre', 'Unknown'),
                                'datetime': measurement.get('datetime'),
                                'value': measurement.get('valor'),
                                'unit': measurement.get('unidad')
                            })

            df = pd.DataFrame(records)
            logger.info(f"Extracted {len(df)} PM2.5 measurements")
            return df

        except Exception as e:
            logger.error(f"Error fetching from API: {e}")
            return pd.DataFrame()

    def download_historical_csv_method(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/external"
    ) -> pd.DataFrame:
        """
        Attempt to download historical data using form submission.

        This method tries to simulate the web form submission.
        May require Selenium for JavaScript-rendered content.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory

        Returns:
            DataFrame with historical data
        """
        logger.info(f"Attempting to download historical data: {start_date} to {end_date}")

        try:
            # SINCA download form endpoint (need to verify actual URL)
            download_url = f"{self.base_url}/index.php/datos/descarga"

            # Prepare form data
            form_data = {
                'region': '13',  # Región Metropolitana
                'contaminante': 'MP2.5',
                'fecha_inicio': start_date,
                'fecha_fin': end_date,
                'formato': 'csv',
                'todas_estaciones': '1'
            }

            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es-CL,es;q=0.9,en;q=0.8'
            })

            # First GET to establish session
            logger.info("Establishing session...")
            session.get(download_url, timeout=30)
            time.sleep(2)

            # POST to request data
            logger.info("Submitting download request...")
            response = session.post(
                download_url,
                data=form_data,
                timeout=120,
                allow_redirects=True
            )

            if response.status_code == 200:
                # Check if we got CSV data
                if 'text/csv' in response.headers.get('Content-Type', ''):
                    # Save raw CSV
                    output_file = Path(output_dir) / f"sinca_raw_{start_date}_{end_date}.csv"
                    output_file.write_bytes(response.content)
                    logger.info(f"Downloaded CSV saved to {output_file}")

                    # Parse CSV
                    df = pd.read_csv(output_file)
                    return df
                else:
                    logger.warning("Response is not CSV format. May need Selenium for JavaScript rendering.")
                    logger.debug(f"Response content type: {response.headers.get('Content-Type')}")
                    return pd.DataFrame()
            else:
                logger.error(f"HTTP error: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in CSV download method: {e}")
            return pd.DataFrame()

    def download_by_year(
        self,
        start_year: int,
        end_year: int,
        output_dir: str = "data/external"
    ):
        """
        Download data year by year to avoid timeouts.

        Args:
            start_year: Starting year
            end_year: Ending year
            output_dir: Output directory
        """
        logger.info(f"Downloading data from {start_year} to {end_year} (year by year)")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_files = []

        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"

            # Handle last year (current year)
            if year == end_year and year == datetime.now().year:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = f"{year}-12-31"

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {year}: {start_date} to {end_date}")
            logger.info(f"{'='*60}")

            try:
                df = self.download_historical_csv_method(start_date, end_date, output_dir)

                if not df.empty:
                    output_file = Path(output_dir) / f"sinca_{year}.csv"
                    df.to_csv(output_file, index=False)
                    all_files.append(output_file)
                    logger.info(f"✓ Year {year} complete: {len(df)} records saved")
                else:
                    logger.warning(f"✗ No data for year {year}")

                # Rate limiting
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error processing year {year}: {e}")
                continue

        # Combine all yearly files
        if all_files:
            logger.info(f"\nCombining {len(all_files)} yearly files...")
            dfs = [pd.read_csv(f) for f in all_files]
            combined = pd.concat(dfs, ignore_index=True)

            output_file = Path(output_dir) / "sinca_pm25_raw.csv"
            combined.to_csv(output_file, index=False)
            logger.info(f"✓ Combined file saved: {output_file}")
            logger.info(f"  Total records: {len(combined):,}")

            return combined
        else:
            logger.error("No data downloaded!")
            return pd.DataFrame()

    def download_with_selenium(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/external"
    ):
        """
        Download using Selenium for JavaScript-rendered content.

        Requires: pip install selenium

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium")
            logger.info("Also need Chrome/Firefox driver: https://chromedriver.chromium.org/")
            return pd.DataFrame()

        logger.info("Using Selenium method (for JavaScript-rendered content)...")

        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            # Initialize driver
            driver = webdriver.Chrome(options=chrome_options)

            # Navigate to download page
            download_url = f"{self.base_url}/index.php/datos/descarga"
            logger.info(f"Opening {download_url}")
            driver.get(download_url)

            # Wait for page to load
            wait = WebDriverWait(driver, 10)

            # Fill form
            # (Need to inspect actual form elements)
            logger.info("Filling download form...")

            # Region selector
            region_select = wait.until(
                EC.presence_of_element_located((By.ID, "region"))
            )
            region_select.send_keys("Metropolitana")

            # Pollutant selector
            pollutant_select = driver.find_element(By.ID, "contaminante")
            pollutant_select.send_keys("MP2.5")

            # Date fields
            start_field = driver.find_element(By.ID, "fecha_inicio")
            start_field.clear()
            start_field.send_keys(start_date)

            end_field = driver.find_element(By.ID, "fecha_fin")
            end_field.clear()
            end_field.send_keys(end_date)

            # Select all stations
            all_stations_checkbox = driver.find_element(By.ID, "todas_estaciones")
            if not all_stations_checkbox.is_selected():
                all_stations_checkbox.click()

            # Submit form
            submit_button = driver.find_element(By.ID, "descargar")
            submit_button.click()

            logger.info("Form submitted, waiting for download...")
            time.sleep(10)  # Wait for download to complete

            # Close browser
            driver.quit()

            logger.info("Selenium download complete. Check download folder.")
            logger.warning("Note: File may be in system Downloads folder, move to data/external/")

            return pd.DataFrame()  # Placeholder

        except Exception as e:
            logger.error(f"Selenium error: {e}")
            try:
                driver.quit()
            except:
                pass
            return pd.DataFrame()


def create_station_metadata():
    """
    Create metadata file with station coordinates.
    """
    logger.info("Creating station metadata file...")

    # Known Santiago station coordinates
    stations = {
        'Cerrillos': {'lat': -33.50, 'lon': -70.71, 'elevation': 522},
        'Cerro Navia': {'lat': -33.42, 'lon': -70.74, 'elevation': 485},
        'El Bosque': {'lat': -33.56, 'lon': -70.69, 'elevation': 580},
        'Independencia': {'lat': -33.41, 'lon': -70.66, 'elevation': 540},
        'La Florida': {'lat': -33.52, 'lon': -70.58, 'elevation': 600},
        'Las Condes': {'lat': -33.40, 'lon': -70.58, 'elevation': 700},
        'Pudahuel': {'lat': -33.44, 'lon': -70.77, 'elevation': 475},
        'Puente Alto': {'lat': -33.61, 'lon': -70.58, 'elevation': 640},
        'Quilicura': {'lat': -33.36, 'lon': -70.72, 'elevation': 500},
        'Santiago Centro': {'lat': -33.45, 'lon': -70.67, 'elevation': 520},
        'Talagante': {'lat': -33.66, 'lon': -70.93, 'elevation': 355},
        'Parque O\'Higgins': {'lat': -33.46, 'lon': -70.65, 'elevation': 525}
    }

    df = pd.DataFrame.from_dict(stations, orient='index')
    df.index.name = 'station'
    df.reset_index(inplace=True)

    output_file = Path("data/external/sinca_stations_metadata.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Station metadata saved to {output_file}")

    return df


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatic SINCA Data Downloader")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year")
    parser.add_argument("--end-year", type=int, default=2025, help="End year")
    parser.add_argument("--method", choices=['api', 'csv', 'selenium', 'year-by-year'],
                       default='year-by-year', help="Download method")
    parser.add_argument("--output", default="data/external", help="Output directory")

    args = parser.parse_args()

    downloader = SINCAAutoDownloader()

    print("\n" + "="*80)
    print("SINCA AUTOMATIC DOWNLOADER")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Period: {args.start_year} - {args.end_year}")
    print("="*80 + "\n")

    if args.method == 'api':
        # API only gets recent data (last 48h)
        logger.warning("API method only provides recent data (last 48 hours)")
        df = downloader.fetch_recent_data()
        if not df.empty:
            output_file = Path(args.output) / "sinca_recent.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Recent data saved to {output_file}")

    elif args.method == 'csv':
        # Try direct CSV download
        start_date = f"{args.start_year}-01-01"
        end_date = f"{args.end_year}-12-31"
        df = downloader.download_historical_csv_method(start_date, end_date, args.output)

    elif args.method == 'selenium':
        # Use Selenium
        start_date = f"{args.start_year}-01-01"
        end_date = f"{args.end_year}-12-31"
        df = downloader.download_with_selenium(start_date, end_date, args.output)

    elif args.method == 'year-by-year':
        # Download year by year (most reliable)
        df = downloader.download_by_year(args.start_year, args.end_year, args.output)

    # Create station metadata
    create_station_metadata()

    print("\n" + "="*80)
    print("DOWNLOAD ATTEMPT COMPLETE")
    print("="*80)
    print("\nIMPORTANT NOTES:")
    print("1. If automatic download failed, use manual method:")
    print("   https://sinca.mma.gob.cl/index.php/datos/descarga")
    print("2. See docs/SINCA_DOWNLOAD_GUIDE.md for detailed instructions")
    print("3. Manual download is often more reliable for large date ranges")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
