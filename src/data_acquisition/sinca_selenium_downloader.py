"""
SINCA Selenium Downloader - REAL Implementation
Uses Selenium to navigate SINCA website and download data like a human would.

Strategy:
1. Navigate to SINCA Metropolitan region page
2. Find and click on report/download links
3. Fill forms with date ranges
4. Download CSV files
5. Parse and combine data
"""

import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sinca_selenium.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SINCASeleniumDownloader:
    """
    Intelligent Selenium-based SINCA data downloader.
    """

    def __init__(self, headless=True, download_dir=None):
        """
        Initialize Selenium driver.

        Args:
            headless: Run browser in headless mode
            download_dir: Download directory
        """
        self.download_dir = download_dir or str(Path("data/external").absolute())
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Selenium WebDriver...")
        logger.info(f"Download directory: {self.download_dir}")

        # Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")

        # Set download preferences
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Initialize driver
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("✓ WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def explore_site(self, url):
        """
        Explore the SINCA site to understand its structure.

        Args:
            url: URL to explore
        """
        logger.info(f"Exploring: {url}")
        self.driver.get(url)
        time.sleep(3)

        # Log page title
        logger.info(f"Page title: {self.driver.title}")

        # Find all links
        links = self.driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"Found {len(links)} links")

        # Look for relevant links
        relevant_keywords = ['dato', 'reporte', 'descarga', 'histórico', 'histórica', 'tabla', 'consulta']
        relevant_links = []

        for link in links:
            try:
                text = link.text.strip().lower()
                href = link.get_attribute('href')

                if any(keyword in text for keyword in relevant_keywords):
                    relevant_links.append({
                        'text': link.text.strip(),
                        'href': href
                    })
            except:
                continue

        logger.info(f"\nRelevant links found:")
        for i, link_info in enumerate(relevant_links[:20], 1):
            logger.info(f"  {i}. {link_info['text'][:60]:60} -> {link_info['href'][:80]}")

        return relevant_links

    def navigate_to_main_page(self):
        """
        Navigate to SINCA Metropolitan page and wait for it to load.
        """
        logger.info("\n" + "="*60)
        logger.info("NAVIGATING TO SINCA METROPOLITAN PAGE")
        logger.info("="*60)

        base_url = "https://sinca.mma.gob.cl/index.php/region/index/id/M"
        logger.info(f"Loading: {base_url}")

        self.driver.get(base_url)

        # Wait for page to load completely
        wait = WebDriverWait(self.driver, 20)
        time.sleep(5)  # Let JavaScript fully render

        logger.info(f"Page title: {self.driver.title}")
        return True

    def inspect_page_structure(self):
        """
        Deep inspection of page structure to understand download mechanism.
        """
        logger.info("\n" + "="*60)
        logger.info("INSPECTING PAGE STRUCTURE")
        logger.info("="*60)

        # Find all tables
        tables = self.driver.find_elements(By.TAG_NAME, "table")
        logger.info(f"Found {len(tables)} table(s)")

        # Find all download-related links/buttons
        # Look for 'k' icons or download links
        download_links = []

        # Try different selectors
        selectors = [
            "a[href*='descarga']",
            "a[href*='download']",
            "a[href*='dato']",
            "img[src*='ico']",  # Icon images
            "a img",  # Links with images
        ]

        for selector in selectors:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                logger.info(f"  Found {len(elements)} elements matching '{selector}'")
                download_links.extend(elements)

        # Log unique hrefs
        unique_hrefs = set()
        for link in download_links:
            try:
                if link.tag_name == 'a':
                    href = link.get_attribute('href')
                    if href:
                        unique_hrefs.add(href)
            except:
                continue

        logger.info(f"\nUnique download URLs found: {len(unique_hrefs)}")
        for i, href in enumerate(list(unique_hrefs)[:10], 1):
            logger.info(f"  {i}. {href}")

        return list(unique_hrefs)

    def find_pm25_graph_icons_alt(self):
        """
        Alternative method: Find graph icons - they are <a class="iframe"> with <span class="icon-timeseries">
        """
        logger.info("\n" + "="*60)
        logger.info("SEARCHING FOR PM2.5 GRAPH ICONS (icon-timeseries)")
        logger.info("="*60)

        time.sleep(5)  # Let page fully load

        try:
            # Find all <a> links with class="iframe" that contain icon-timeseries
            # These are the graph icons!
            all_iframe_links = self.driver.find_elements(By.CSS_SELECTOR, "a.iframe")
            logger.info(f"Found {len(all_iframe_links)} iframe links")

            # Filter for MP 2.5 (PM25) links
            pm25_links = []
            for link in all_iframe_links:
                try:
                    href = link.get_attribute('href')
                    title = link.get_attribute('title')

                    # Check if this is a PM2.5 link
                    if 'PM25' in href or 'MP 2,5' in title or 'MP 2.5' in title:
                        # Get station name from title
                        station_name = title.split('|')[-1].strip() if '|' in title else "Unknown"

                        pm25_links.append({
                            'station': station_name,
                            'element': link,
                            'href': href,
                            'title': title
                        })

                        logger.info(f"  ✓ {station_name}")
                except Exception as e:
                    logger.debug(f"Error processing link: {e}")
                    continue

            logger.info(f"\n✓ Found {len(pm25_links)} PM2.5 graph icon links")
            return pm25_links

        except Exception as e:
            logger.error(f"Error finding graph icons: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def find_pm25_graph_icons(self):
        """
        Find all PM2.5 graph icons on the page.

        Strategy:
        1. Find table on the page
        2. Identify MP 2.5 column (second column based on user info)
        3. Find all graph icons in that column
        4. Return list of clickable elements with station names
        """
        logger.info("\n" + "="*60)
        logger.info("SEARCHING FOR PM2.5 GRAPH ICONS")
        logger.info("="*60)

        wait = WebDriverWait(self.driver, 20)
        time.sleep(3)  # Let page fully load

        try:
            # Find table headers to identify MP 2.5 column
            headers = self.driver.find_elements(By.TAG_NAME, "th")
            pm25_column_index = -1

            logger.info("\nTable columns found:")
            for i, header in enumerate(headers):
                text = header.text.strip()
                logger.info(f"  Column {i}: '{text}'")

                # Look for "MP 2,5" or "MP 2.5" (note: comma vs period!)
                # Exclude "discreto" version
                if ('MP 2,5' in text or 'MP 2.5' in text or 'PM2.5' in text or 'PM 2,5' in text) and 'discreto' not in text.lower():
                    pm25_column_index = i
                    logger.info(f"  ✓ Found MP 2.5 column at index {i}")
                    break

            if pm25_column_index == -1:
                logger.error("Could not find MP 2.5 column in table headers")
                return []

            # Find all table rows
            rows = self.driver.find_elements(By.TAG_NAME, "tr")
            logger.info(f"\nFound {len(rows)} table rows")

            pm25_graph_icons = []

            # Process each row (skip header)
            for row_idx, row in enumerate(rows[1:], 1):
                cells = row.find_elements(By.TAG_NAME, "td")

                if len(cells) > pm25_column_index:
                    # Get station name from first cell
                    station_name = cells[0].text.strip() if cells else f"Unknown_{row_idx}"

                    # Get MP 2.5 cell
                    pm25_cell = cells[pm25_column_index]

                    # Debug: log cell content
                    cell_html = pm25_cell.get_attribute('innerHTML')
                    logger.debug(f"  Station: {station_name}")
                    logger.debug(f"    Cell HTML: {cell_html[:200]}")

                    # Find clickable graph icon (link or image inside link)
                    # The icon opens a modal with the graph
                    graph_links = pm25_cell.find_elements(By.TAG_NAME, "a")

                    if graph_links:
                        # Found the graph icon link
                        icon_element = graph_links[0]
                        href = icon_element.get_attribute('href')
                        pm25_graph_icons.append({
                            'station': station_name,
                            'element': icon_element,
                            'row_index': row_idx
                        })
                        logger.info(f"  ✓ {station_name}: Found graph icon (href: {href[:50] if href else 'None'})")
                    else:
                        logger.warning(f"  ✗ {station_name}: No <a> tag found in cell")

            logger.info(f"\n✓ Found {len(pm25_graph_icons)} PM2.5 graph icons")
            return pm25_graph_icons

        except Exception as e:
            logger.error(f"Error finding PM2.5 graph icons: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def fill_download_form(self, start_date, end_date, station=None):
        """
        Fill the download form with date range and parameters.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            station: Station name/code (optional)
        """
        logger.info(f"\nFilling form for {start_date} to {end_date}")

        try:
            # Wait for form to be present
            wait = WebDriverWait(self.driver, 10)

            # Look for date inputs
            date_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='date'], input[type='text'][placeholder*='fecha'], input[name*='fecha']")

            if len(date_inputs) >= 2:
                logger.info(f"  Found {len(date_inputs)} date inputs")

                # Fill start date
                date_inputs[0].clear()
                date_inputs[0].send_keys(start_date)
                logger.info(f"  ✓ Start date: {start_date}")

                # Fill end date
                date_inputs[1].clear()
                date_inputs[1].send_keys(end_date)
                logger.info(f"  ✓ End date: {end_date}")

            else:
                logger.warning(f"  Could not find date inputs (found {len(date_inputs)})")

            # Look for contaminant selector
            try:
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in selects:
                    options_text = [opt.text for opt in Select(select_elem).options]

                    # Check if this is the contaminant selector
                    if any('MP2.5' in opt or 'PM2.5' in opt or 'PM25' in opt for opt in options_text):
                        logger.info(f"  Found contaminant selector")
                        select = Select(select_elem)

                        # Try different PM2.5 variations
                        for pm_variant in ['MP2.5', 'PM2.5', 'PM25', 'mp25']:
                            try:
                                select.select_by_visible_text(pm_variant)
                                logger.info(f"  ✓ Selected contaminant: {pm_variant}")
                                break
                            except:
                                continue
                        break

            except Exception as e:
                logger.debug(f"  Could not set contaminant: {e}")

            # Look for station selector
            if station:
                try:
                    station_select = self.driver.find_element(By.NAME, "estacion")
                    Select(station_select).select_by_visible_text(station)
                    logger.info(f"  ✓ Selected station: {station}")
                except:
                    logger.debug(f"  Could not set station")

            # Look for submit button
            submit_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button.btn")
            submit_buttons += self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Descargar') or contains(text(), 'Consultar') or contains(text(), 'Buscar')]")

            if submit_buttons:
                logger.info(f"  Found {len(submit_buttons)} submit button(s)")
                submit_buttons[0].click()
                logger.info(f"  ✓ Clicked submit button")
                time.sleep(5)  # Wait for download/processing
                return True
            else:
                logger.warning(f"  Could not find submit button")
                return False

        except Exception as e:
            logger.error(f"Error filling form: {e}")
            return False

    def click_graph_icon_and_download_csv(self, icon_info):
        """
        Click graph icon to open modal, then download CSV.

        Workflow:
        1. Click graph icon -> Modal opens with historical graph
        2. Wait for modal to appear
        3. Find CSV download button in modal
        4. Click CSV button -> File downloads
        5. Close modal
        6. Return to main table

        Args:
            icon_info: Dict with 'station', 'element', 'row_index'

        Returns:
            Downloaded file path or None
        """
        station = icon_info['station']
        logger.info(f"\n→ Processing {station}")

        try:
            # Get current files before download
            files_before = set(Path(self.download_dir).glob("*.csv"))
            files_before.update(Path(self.download_dir).glob("*.xls*"))
            files_before.update(Path(self.download_dir).glob("*.txt"))

            # Step 1: Click the graph icon to open modal
            logger.info(f"  Clicking graph icon...")

            # Scroll element into view and wait a bit
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", icon_info['element'])
            time.sleep(1)

            # Try click, with JavaScript fallback if normal click fails
            try:
                icon_info['element'].click()
            except Exception as click_error:
                logger.warning(f"  Normal click failed ({click_error}), trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", icon_info['element'])

            time.sleep(3)  # Wait for modal to open

            # Step 2: Wait for modal to appear
            wait = WebDriverWait(self.driver, 10)

            # Try different modal selectors
            modal_selectors = [
                "div.modal",
                "div[role='dialog']",
                "div.ui-dialog",
                "div#modal",
                "div.popup"
            ]

            modal_found = False
            for selector in modal_selectors:
                try:
                    modal = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    logger.info(f"  ✓ Modal opened (selector: {selector})")
                    modal_found = True
                    break
                except TimeoutException:
                    continue

            if not modal_found:
                logger.warning(f"  Could not detect modal, proceeding anyway...")

            time.sleep(2)  # Let modal content load

            # Step 3: Find CSV download link
            # The page shows "Descargar: PDF Texto Excel CSV" at bottom
            logger.info(f"  Looking for CSV download link...")

            csv_buttons = []

            # Strategy 1: Look for link/button with just "CSV" text (seen in screenshot)
            # The text appears as plain text links, not buttons
            csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[text()='CSV']"))
            csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[normalize-space(text())='CSV']"))

            # Strategy 2: Link that contains CSV and is near "Descargar:"
            csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[contains(@href, 'CSV') or contains(@onclick, 'CSV')]"))

            # Strategy 3: Any clickable element with CSV
            csv_buttons.extend(self.driver.find_elements(By.XPATH, "//*[text()='CSV' and (name()='a' or name()='button' or name()='span')]"))

            # Strategy 4: Search in iframe if modal is an iframe
            try:
                iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
                if iframes:
                    logger.info(f"  Found {len(iframes)} iframe(s), switching to first...")
                    self.driver.switch_to.frame(iframes[0])
                    time.sleep(2)

                    # The iframe contains a frameset with "left" and "right" frames
                    # Based on screenshot, CSV download buttons are in the "right" frame (where the graph is)
                    # But right frame loads dynamically, so we need to wait for it to load
                    time.sleep(5)  # Wait for right frame to load content

                    try:
                        right_frame = self.driver.find_element(By.NAME, "right")
                        logger.info(f"  Found 'right' frame inside iframe, switching to it...")
                        self.driver.switch_to.frame(right_frame)
                        time.sleep(3)  # Wait for content to fully render
                    except Exception as e:
                        logger.warning(f"  Could not switch to 'right' frame: {e}")
                        # Try left frame as fallback
                        try:
                            self.driver.switch_to.default_content()
                            self.driver.switch_to.frame(iframes[0])
                            left_frame = self.driver.find_element(By.NAME, "left")
                            logger.info(f"  Trying 'left' frame as fallback...")
                            self.driver.switch_to.frame(left_frame)
                            time.sleep(2)
                        except:
                            logger.debug(f"  No frame switching possible, staying in current context")
                            pass

                    # Use JavaScript to find CSV download options - could be links, buttons, or text elements
                    script = """
                    // Search all elements that contain 'CSV' text
                    var allElements = document.querySelectorAll('*');
                    var csvElements = [];

                    for (var i = 0; i < allElements.length; i++) {
                        var elem = allElements[i];
                        var text = elem.textContent.trim();
                        var tag = elem.tagName.toLowerCase();

                        // Check if element contains 'CSV' and is not too large (to avoid body/html)
                        if (text.includes('CSV') && text.length < 200) {
                            csvElements.push({
                                tag: tag,
                                text: text,
                                href: elem.getAttribute('href') || '',
                                onclick: elem.getAttribute('onclick') || '',
                                className: elem.className,
                                id: elem.id,
                                visible: elem.offsetParent !== null
                            });
                        }
                    }

                    return csvElements;
                    """
                    csv_elements_info = self.driver.execute_script(script)
                    logger.info(f"  JavaScript found {len(csv_elements_info)} element(s) containing 'CSV' in iframe")

                    # Log detailed information about each CSV element
                    for i, elem_info in enumerate(csv_elements_info):
                        logger.info(f"    Element {i}: tag='{elem_info['tag']}', text='{elem_info['text'][:80]}...', visible={elem_info['visible']}")
                        logger.info(f"              id='{elem_info['id']}', class='{elem_info['className']}', onclick='{elem_info['onclick'][:50] if elem_info['onclick'] else 'None'}...'")

                    # Based on debug findings, the CSV download is via "Excel CSV" link
                    # which calls javascript:Open('xcl')
                    logger.info(f"  Looking for 'Excel CSV' or CSV-related links...")

                    # Strategy 1: Link with text "Excel CSV" (most reliable)
                    csv_buttons.extend(self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Excel CSV"))
                    csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Excel') and contains(text(), 'CSV')]"))

                    # Strategy 2: Link with href containing Open('xcl')
                    csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[contains(@href, \"Open('xcl')\")]"))
                    csv_buttons.extend(self.driver.find_elements(By.XPATH, "//a[contains(@href, 'xcl')]"))

                    # Strategy 3: Any link with just "CSV" text
                    csv_buttons.extend(self.driver.find_elements(By.LINK_TEXT, "CSV"))
                    csv_buttons.extend(self.driver.find_elements(By.PARTIAL_LINK_TEXT, "CSV"))

                    logger.info(f"  Found {len(csv_buttons)} potential CSV links using Selenium")

                    # Log CSV elements found via JavaScript
                    if csv_elements_info:
                        logger.info(f"  JavaScript found {len(csv_elements_info)} elements with 'CSV':")
                        for i, elem_info in enumerate(csv_elements_info[:5]):  # Show first 5
                            logger.info(f"    {i+1}. tag={elem_info['tag']}, text='{elem_info['text'][:60]}', href='{elem_info['href'][:50] if elem_info['href'] else 'None'}'...")

            except Exception as e:
                logger.debug(f"  Could not switch to iframe: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            if csv_buttons:
                logger.info(f"  Found {len(csv_buttons)} potential CSV download button(s)")

                # Click the first visible CSV button
                clicked = False
                for btn in csv_buttons:
                    try:
                        if btn.is_displayed() and btn.is_enabled():
                            btn_text = btn.text
                            logger.info(f"  Clicking CSV button: '{btn_text}'...")

                            # Scroll into view first
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                            time.sleep(0.5)

                            btn.click()
                            logger.info(f"  ✓ Clicked CSV download button")
                            time.sleep(5)  # Wait for download to start
                            clicked = True
                            break
                    except Exception as e:
                        logger.debug(f"  Could not click button: {e}")
                        continue

                if not clicked:
                    logger.warning(f"  ✗ Found CSV buttons but couldn't click any")
                    self.take_screenshot(f"csv_button_not_clickable_{station.replace(' ', '_')}.png")
            else:
                logger.warning(f"  ✗ No CSV download button found")
                self.take_screenshot(f"no_csv_button_{station.replace(' ', '_')}.png")

            # Step 4: Switch back to main content if we switched to iframe
            try:
                self.driver.switch_to.default_content()
                logger.debug(f"  Switched back to default content")
            except:
                pass

            # Step 5: Close modal
            logger.info(f"  Closing modal...")

            close_methods = [
                # Click X button in top right
                (By.CSS_SELECTOR, "button.ui-dialog-titlebar-close"),
                (By.CSS_SELECTOR, "a.ui-dialog-titlebar-close"),
                (By.XPATH, "//button[contains(@class, 'ui-button')][@title='Cerrar']"),
                (By.XPATH, "//button[contains(@class, 'ui-button')][@title='Close']"),
                # Generic close buttons
                (By.CSS_SELECTOR, "button.close"),
                (By.CSS_SELECTOR, "a.close"),
                # Press ESC key
                ("ESC", None),
            ]

            modal_closed = False
            for method in close_methods:
                try:
                    if method[0] == "ESC":
                        # Press ESC key
                        from selenium.webdriver.common.keys import Keys
                        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                        time.sleep(1)
                        modal_closed = True
                        logger.info(f"  ✓ Modal closed (ESC key)")
                        break
                    else:
                        close_btn = self.driver.find_element(*method)
                        if close_btn.is_displayed():
                            close_btn.click()
                            time.sleep(1)
                            modal_closed = True
                            logger.info(f"  ✓ Modal closed (button)")
                            break
                except:
                    continue

            if not modal_closed:
                logger.debug(f"  Could not find close button, trying ESC anyway...")
                try:
                    from selenium.webdriver.common.keys import Keys
                    self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                    time.sleep(1)
                    logger.info(f"  ✓ Modal closed (ESC fallback)")
                except:
                    pass

            time.sleep(2)  # Wait for download to complete

            # Step 5: Check for new downloaded files
            files_after = set(Path(self.download_dir).glob("*.csv"))
            files_after.update(Path(self.download_dir).glob("*.xls*"))
            files_after.update(Path(self.download_dir).glob("*.txt"))

            new_files = files_after - files_before

            if new_files:
                new_file = list(new_files)[0]
                logger.info(f"  ✓ Downloaded: {new_file.name} ({new_file.stat().st_size/1024:.1f} KB)")
                return new_file
            else:
                logger.warning(f"  ✗ No file downloaded")
                return None

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Take screenshot for debugging
            self.take_screenshot(f"error_{station.replace(' ', '_')}.png")
            return None

    def download_all_pm25_data(self, start_date=None, end_date=None):
        """
        Download PM2.5 data from all stations.

        Note: start_date and end_date are not used because SINCA downloads
        complete historical data when clicking the CSV button.

        Args:
            start_date: Not used (kept for compatibility)
            end_date: Not used (kept for compatibility)

        Returns:
            List of downloaded files
        """
        logger.info("\n" + "="*60)
        logger.info(f"DOWNLOADING ALL PM2.5 DATA")
        logger.info(f"Note: Each CSV will contain complete historical data")
        logger.info("="*60)

        # Step 1: Navigate to main page
        self.navigate_to_main_page()

        # Step 2: Inspect structure (for debugging)
        self.inspect_page_structure()

        # Step 3: Find PM2.5 graph icons
        # Try alternative method first (finds all links with images)
        pm25_icons = self.find_pm25_graph_icons_alt()

        # If that fails, try the table-based method
        if not pm25_icons:
            logger.info("Alternative method found no icons, trying table-based method...")
            pm25_icons = self.find_pm25_graph_icons()

        if not pm25_icons:
            logger.error("No PM2.5 graph icons found!")
            self.take_screenshot("no_icons_found.png")
            return []

        # Step 4: Click each icon, open modal, download CSV
        downloaded_files = []

        for i, icon_info in enumerate(pm25_icons, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"[{i}/{len(pm25_icons)}] {icon_info['station']}")
            logger.info(f"{'='*50}")

            result = self.click_graph_icon_and_download_csv(icon_info)

            if result:
                downloaded_files.append(result)
            else:
                logger.warning(f"Failed to download data for {icon_info['station']}")

            # Rate limiting - be nice to the server
            time.sleep(3)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"DOWNLOAD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total stations found: {len(pm25_icons)}")
        logger.info(f"Files downloaded: {len(downloaded_files)}")
        logger.info(f"Success rate: {len(downloaded_files)}/{len(pm25_icons)} ({100*len(downloaded_files)/len(pm25_icons):.1f}%)")
        logger.info(f"{'='*60}")

        if downloaded_files:
            logger.info(f"\nDownloaded files:")
            for f in downloaded_files:
                logger.info(f"  - {f.name} ({f.stat().st_size/1024:.1f} KB)")

        return downloaded_files

    def download_data(self, start_date, end_date):
        """
        Main method to download data (compatibility wrapper).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Path to downloaded file or None
        """
        files = self.download_all_pm25_data(start_date, end_date)
        return files[0] if files else None

    def take_screenshot(self, filename="screenshot.png"):
        """Take screenshot for debugging."""
        path = Path("logs") / filename
        path.parent.mkdir(exist_ok=True)
        self.driver.save_screenshot(str(path))
        logger.info(f"Screenshot saved: {path}")

    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="SINCA Selenium Downloader")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--explore", action="store_true", help="Just explore the site")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SINCA SELENIUM DOWNLOADER - INTELLIGENT WEB SCRAPING")
    print("="*80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Mode: {'Headless' if args.headless else 'Visible'}")
    print("="*80 + "\n")

    downloader = None

    try:
        downloader = SINCASeleniumDownloader(headless=args.headless)

        if args.explore:
            # Just explore
            logger.info("EXPLORATION MODE")
            downloader.explore_site("https://sinca.mma.gob.cl/index.php/region/index/id/M")
        else:
            # Try to download
            result = downloader.download_data(args.start_date, args.end_date)

            if result:
                logger.info(f"\n{'='*60}")
                logger.info(f"SUCCESS! Downloaded: {result}")
                logger.info(f"{'='*60}")
            else:
                logger.error("\nDownload failed. Taking screenshot...")
                downloader.take_screenshot("failed_download.png")
                logger.info("\nRecommendation: Run with --explore to see site structure")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        if downloader:
            downloader.take_screenshot("error_screenshot.png")
    finally:
        if downloader:
            time.sleep(2)  # Give time to see result
            downloader.close()


if __name__ == "__main__":
    main()
