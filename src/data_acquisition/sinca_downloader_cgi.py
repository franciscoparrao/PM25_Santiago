"""
SINCA Data Downloader using CGI endpoint
Downloads PM2.5 historical data from SINCA using their CGI interface.

Based on: https://sinca.mma.gob.cl/index.php/region/index/id/M
CGI endpoint: /cgi-bin/APUB-MMA/apub.tsindico2.cgi
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sinca_cgi_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SINCACGIDownloader:
    """
    Download SINCA data using CGI interface.
    """

    def __init__(self):
        self.base_url = "https://sinca.mma.gob.cl"
        self.cgi_endpoint = f"{self.base_url}/cgi-bin/APUB-MMA/apub.tsindico2.cgi"

        # Stations in Metropolitan Region
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
            'Santiago': 'ST'
        }

    def download_station_data(
        self,
        station_code: str,
        station_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download data for a specific station using CGI.

        Args:
            station_code: Station code (e.g., 'CE', 'EB')
            station_name: Station name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with measurements
        """
        logger.info(f"Downloading {station_name} ({station_code}): {start_date} to {end_date}")

        try:
            # Convert dates to the format SINCA expects
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # CGI parameters (these may need adjustment based on actual SINCA API)
            params = {
                'estacion': station_code,
                'parametro': 'MP25',  # PM2.5
                'fecha_inicio': start_dt.strftime("%d/%m/%Y"),
                'fecha_fin': end_dt.strftime("%d/%m/%Y"),
                'formato': 'csv',
                'region': 'M'  # Metropolitan
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es-CL,es;q=0.9'
            }

            session = requests.Session()

            # Make request
            response = session.get(
                self.cgi_endpoint,
                params=params,
                headers=headers,
                timeout=60
            )

            if response.status_code == 200:
                logger.info(f"  Response received: {len(response.content)} bytes")
                logger.info(f"  Content-Type: {response.headers.get('Content-Type', 'unknown')}")

                # Try to parse as CSV
                try:
                    df = pd.read_csv(StringIO(response.text))
                    logger.info(f"  ✓ Parsed {len(df)} records")

                    # Add metadata
                    df['estacion'] = station_name
                    df['codigo_estacion'] = station_code

                    return df

                except Exception as e:
                    logger.warning(f"  Cannot parse as CSV: {e}")
                    logger.debug(f"  First 500 chars: {response.text[:500]}")
                    return pd.DataFrame()
            else:
                logger.error(f"  HTTP {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error downloading {station_name}: {e}")
            return pd.DataFrame()

    def download_all_stations(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/external"
    ) -> pd.DataFrame:
        """
        Download data for all stations.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory

        Returns:
            Combined DataFrame
        """
        logger.info(f"Downloading all stations: {start_date} to {end_date}")
        logger.info(f"Total stations: {len(self.stations)}\n")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_data = []

        for station_name, station_code in self.stations.items():
            try:
                df = self.download_station_data(
                    station_code,
                    station_name,
                    start_date,
                    end_date
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  ✓ {station_name}: {len(df)} records\n")
                else:
                    logger.warning(f"  ✗ {station_name}: No data\n")

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error with {station_name}: {e}\n")
                continue

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n{'='*60}")
            logger.info(f"Total records downloaded: {len(combined):,}")
            logger.info(f"{'='*60}\n")
            return combined
        else:
            logger.error("No data downloaded from any station!")
            return pd.DataFrame()

    def download_by_chunks(
        self,
        start_date: str,
        end_date: str,
        chunk_months: int = 6,
        output_dir: str = "data/external"
    ):
        """
        Download data in chunks to avoid timeouts.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_months: Size of each chunk in months
            output_dir: Output directory
        """
        logger.info(f"Downloading in {chunk_months}-month chunks")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_files = []

        current = start
        chunk_num = 1

        while current < end:
            # Calculate chunk end date
            chunk_end = current + timedelta(days=chunk_months*30)
            if chunk_end > end:
                chunk_end = end

            chunk_start_str = current.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"\n{'='*60}")
            logger.info(f"CHUNK {chunk_num}: {chunk_start_str} to {chunk_end_str}")
            logger.info(f"{'='*60}")

            try:
                df = self.download_all_stations(chunk_start_str, chunk_end_str, output_dir)

                if not df.empty:
                    output_file = Path(output_dir) / f"sinca_chunk_{chunk_num}_{chunk_start_str}_{chunk_end_str}.csv"
                    df.to_csv(output_file, index=False)
                    all_files.append(output_file)
                    logger.info(f"✓ Chunk {chunk_num} saved: {output_file}")
                else:
                    logger.warning(f"✗ Chunk {chunk_num}: No data")

                chunk_num += 1
                current = chunk_end + timedelta(days=1)

                # Sleep between chunks
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in chunk {chunk_num}: {e}")
                current = chunk_end + timedelta(days=1)
                continue

        # Combine all chunks
        if all_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Combining {len(all_files)} chunks...")
            logger.info(f"{'='*60}")

            dfs = [pd.read_csv(f) for f in all_files]
            combined = pd.concat(dfs, ignore_index=True)

            output_file = Path(output_dir) / "sinca_pm25_raw.csv"
            combined.to_csv(output_file, index=False)

            logger.info(f"\n✓ Final combined file: {output_file}")
            logger.info(f"  Total records: {len(combined):,}")
            logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

            return combined
        else:
            logger.error("No chunks downloaded successfully!")
            return pd.DataFrame()


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="SINCA CGI Downloader")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk-months", type=int, default=6, help="Chunk size in months")
    parser.add_argument("--output", default="data/external", help="Output directory")
    parser.add_argument("--single-chunk", action="store_true", help="Download as single chunk")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SINCA CGI DATA DOWNLOADER")
    print("="*80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    downloader = SINCACGIDownloader()

    if args.single_chunk:
        # Download entire period at once
        df = downloader.download_all_stations(args.start_date, args.end_date, args.output)
        if not df.empty:
            output_file = Path(args.output) / "sinca_pm25_raw.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"\nSaved to: {output_file}")
    else:
        # Download in chunks
        df = downloader.download_by_chunks(
            args.start_date,
            args.end_date,
            args.chunk_months,
            args.output
        )

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
