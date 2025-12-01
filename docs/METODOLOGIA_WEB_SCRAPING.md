# Metodología de Web Scraping con Selenium para Páginas con Iframes Anidados

## Resumen

Esta metodología documenta el proceso desarrollado para automatizar la descarga de datos históricos de PM2.5 desde el Sistema de Información Nacional de Calidad del Aire (SINCA) de Chile, que utiliza una estructura compleja de iframes anidados y framesets antiguos.

**Resultado**: 10 de 13 estaciones descargadas exitosamente (76.9% de éxito), ~1.2 MB de datos históricos PM2.5.

---

## Contexto del Problema

### Desafío Principal
La página de SINCA (https://sinca.mma.gob.cl) utiliza una arquitectura web antigua con:
- Íconos de gráficos que NO son elementos `<img>` sino font icons (`<span class="icon-timeseries">`)
- Modales que son iframes con estructura de frameset
- Frameset con dos frames: "left" (controles) y "right" (gráfico + botones de descarga)
- Botones de descarga que son links JavaScript, no botones HTML tradicionales

### Arquitectura de la Página
```
Página Principal
  └─ Tabla de estaciones
      └─ <a class="iframe"> con <span class="icon-timeseries">
          └─ Click abre iframe modal
              └─ Frameset con 2 frames
                  ├─ Frame "left" (controles de fecha/tipo)
                  └─ Frame "right" (gráfico Highcharts + botones descarga)
                      └─ Links: PDF, Texto, Excel CSV ← OBJETIVO
```

---

## Metodología Paso a Paso

### Fase 1: Exploración y Diagnóstico

#### 1.1 Identificar la estructura de la página

**Herramientas utilizadas:**
- Chrome DevTools (Inspector de elementos)
- Screenshots de Selenium
- `driver.page_source` para guardar HTML

**Técnicas clave:**
```python
# Guardar HTML completo para análisis
html = driver.page_source
with open('page_source.html', 'w', encoding='utf-8') as f:
    f.write(html)

# Tomar screenshots en diferentes etapas
driver.save_screenshot("step_1_main_page.png")
```

**Descubrimientos importantes:**
- Los íconos son `<a class="iframe">` con `<span>` internos (NO `<img>`)
- Necesitamos buscar por atributos de link, no por imágenes

#### 1.2 Crear scripts de exploración

Creamos `inspect_page.py` y `debug_iframe_content.py` para:
- Ejecutar JavaScript en el contexto del navegador
- Inspeccionar elementos dinámicos
- Probar diferentes selectores

**Script de exploración básico:**
```python
# Ejecutar JavaScript para explorar estructura
script = """
var table = document.querySelector('table');
if (table) {
    var headers = table.querySelectorAll('th');
    // Analizar headers y encontrar columna MP 2.5
    return headers.length;
}
return 0;
"""
result = driver.execute_script(script)
```

### Fase 2: Identificación de Elementos Objetivo

#### 2.1 Encontrar los íconos de gráfico

**Selector incorrecto inicial:**
```python
# ✗ NO FUNCIONA - busca imágenes
icons = driver.find_elements(By.XPATH, "//a[img]")
```

**Selector correcto:**
```python
# ✓ FUNCIONA - busca links con clase iframe
all_iframe_links = driver.find_elements(By.CSS_SELECTOR, "a.iframe")

# Filtrar por PM2.5
pm25_links = []
for link in all_iframe_links:
    href = link.get_attribute('href')
    title = link.get_attribute('title')

    if 'PM25' in href or 'MP 2,5' in title or 'MP 2.5' in title:
        pm25_links.append({
            'station': title.split('|')[-1].strip() if '|' in title else "Unknown",
            'element': link,
            'href': href,
            'title': title
        })
```

**Lección aprendida:** No asumir la estructura HTML. Usar inspector de elementos y confirmar los selectores reales.

#### 2.2 Hacer clic en elementos con interferencia

**Problema:** `ElementClickInterceptedException` - otro elemento bloquea el click.

**Solución con fallback a JavaScript:**
```python
# Scroll al elemento para asegurar visibilidad
driver.execute_script(
    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
    element
)
time.sleep(1)

# Intentar click normal, con fallback a JavaScript
try:
    element.click()
except Exception as click_error:
    logger.warning(f"Normal click failed, trying JavaScript click...")
    driver.execute_script("arguments[0].click();", element)
```

**Lección aprendida:** Siempre tener un fallback para clicks. JavaScript click es más confiable pero menos realista.

### Fase 3: Navegación en Iframes Anidados

#### 3.1 Detectar y cambiar a iframes

**Estructura descubierta:**
```html
<!-- Iframe principal -->
<iframe src="...apub.htmlindico2.cgi?page=pageFrame...">
    <!-- Frameset dentro del iframe -->
    <frameset cols="250,*" id="frMain">
        <frame id="left" name="left" src="...pageLeft...">
            <!-- Controles de fecha y tipo de gráfico -->
        </frame>
        <frame id="right" name="right" src="">
            <!-- Gráfico Highcharts + botones de descarga -->
            <!-- AQUÍ ESTÁN LOS BOTONES CSV -->
        </frame>
    </frameset>
</iframe>
```

**Código para navegar:**
```python
# Paso 1: Encontrar iframe principal
iframes = driver.find_elements(By.TAG_NAME, "iframe")
if iframes:
    # Paso 2: Cambiar al iframe principal
    driver.switch_to.frame(iframes[0])
    time.sleep(2)

    # Paso 3: Esperar a que frame "right" cargue contenido
    time.sleep(5)

    # Paso 4: Cambiar al frame "right"
    try:
        right_frame = driver.find_element(By.NAME, "right")
        driver.switch_to.frame(right_frame)
        time.sleep(3)  # Esperar renderizado completo
    except Exception as e:
        logger.error(f"Could not switch to right frame: {e}")
```

**Lección aprendida crítica:**
- Los framesets necesitan tiempo para cargar contenido dinámicamente
- Frame "right" inicialmente tiene `src=""` y carga contenido después
- Necesitamos esperas generosas (5-8 segundos) para contenido dinámico

#### 3.2 Volver al contexto principal

**Importante:** Siempre volver al contexto principal después de trabajar en iframes:

```python
# Después de trabajar en iframe/frame
driver.switch_to.default_content()
```

### Fase 4: Búsqueda de Elementos dentro de Frames

#### 4.1 Usar JavaScript para exploración

**Ventaja:** JavaScript puede buscar en todo el DOM sin limitaciones de Selenium.

```python
script = """
// Buscar TODOS los elementos que contienen 'CSV'
var allElements = document.querySelectorAll('*');
var csvElements = [];

for (var i = 0; i < allElements.length; i++) {
    var elem = allElements[i];
    var text = elem.textContent.trim();
    var tag = elem.tagName.toLowerCase();

    // Filtrar por tamaño para evitar body/html
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

csv_elements = driver.execute_script(script)

# Analizar resultados
for elem in csv_elements:
    print(f"Tag: {elem['tag']}, Text: {elem['text']}, Href: {elem['href']}")
```

**Resultado de la exploración:**
```
Tag: a, Text: 'Excel CSV', Href: 'javascript:Open('xcl')'
```

#### 4.2 Crear selectores específicos

Una vez identificado el elemento objetivo:

```python
# Estrategia 1: Buscar por texto parcial
csv_buttons = driver.find_elements(By.PARTIAL_LINK_TEXT, "Excel CSV")

# Estrategia 2: Buscar por contenido de href
csv_buttons.extend(driver.find_elements(
    By.XPATH,
    "//a[contains(@href, \"Open('xcl')\")]"
))

# Estrategia 3: Buscar por texto que contiene ambas palabras
csv_buttons.extend(driver.find_elements(
    By.XPATH,
    "//a[contains(text(), 'Excel') and contains(text(), 'CSV')]"
))
```

**Lección aprendida:** Usar múltiples estrategias de búsqueda aumenta robustez.

### Fase 5: Descarga de Archivos

#### 5.1 Configurar directorio de descarga

```python
from pathlib import Path

download_dir = str(Path("data/external").absolute())
Path(download_dir).mkdir(parents=True, exist_ok=True)

chrome_options = Options()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
}
chrome_options.add_experimental_option("prefs", prefs)
```

#### 5.2 Detectar archivos descargados

```python
# Antes de descargar
files_before = set(Path(download_dir).glob("*.csv"))

# Hacer click en botón de descarga
csv_button.click()
time.sleep(5)  # Esperar descarga

# Después de descargar
files_after = set(Path(download_dir).glob("*.csv"))

# Encontrar archivo nuevo
new_files = files_after - files_before
if new_files:
    downloaded_file = list(new_files)[0]
    logger.info(f"Downloaded: {downloaded_file.name}")
```

#### 5.3 Cerrar modal después de descarga

```python
# Volver al contexto principal
driver.switch_to.default_content()

# Estrategia 1: Botón de cerrar
close_buttons = driver.find_elements(
    By.CSS_SELECTOR,
    "button.ui-dialog-titlebar-close"
)

if close_buttons:
    close_buttons[0].click()
else:
    # Estrategia 2: Tecla ESC
    from selenium.webdriver.common.keys import Keys
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

time.sleep(2)  # Esperar a que modal cierre
```

---

## Código Completo Funcional

### Estructura del Descargador

```python
class SINCAScraper:
    def __init__(self, download_dir):
        self.download_dir = download_dir
        self.driver = self._init_driver()

    def _init_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def find_pm25_icons(self):
        """Encuentra todos los íconos de gráficos PM2.5"""
        all_iframe_links = self.driver.find_elements(
            By.CSS_SELECTOR,
            "a.iframe"
        )

        pm25_links = []
        for link in all_iframe_links:
            href = link.get_attribute('href')
            title = link.get_attribute('title')

            if 'PM25' in href or 'MP 2,5' in title:
                station = title.split('|')[-1].strip() if '|' in title else "Unknown"
                pm25_links.append({
                    'station': station,
                    'element': link,
                })

        return pm25_links

    def download_station_data(self, icon_info):
        """Descarga datos de una estación"""
        station = icon_info['station']

        # Paso 1: Click en ícono con fallback
        self.driver.execute_script(
            "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
            icon_info['element']
        )
        time.sleep(1)

        try:
            icon_info['element'].click()
        except:
            self.driver.execute_script(
                "arguments[0].click();",
                icon_info['element']
            )

        time.sleep(3)

        # Paso 2: Navegar a iframe y frame "right"
        iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
        if not iframes:
            return False

        self.driver.switch_to.frame(iframes[0])
        time.sleep(5)  # Esperar carga de frameset

        try:
            right_frame = self.driver.find_element(By.NAME, "right")
            self.driver.switch_to.frame(right_frame)
            time.sleep(3)
        except:
            self.driver.switch_to.default_content()
            return False

        # Paso 3: Buscar y hacer click en botón CSV
        csv_buttons = self.driver.find_elements(
            By.PARTIAL_LINK_TEXT,
            "Excel CSV"
        )

        if not csv_buttons:
            self.driver.switch_to.default_content()
            return False

        # Encontrar botón visible
        for btn in csv_buttons:
            if btn.is_displayed():
                btn.click()
                time.sleep(5)  # Esperar descarga
                break

        # Paso 4: Cerrar modal
        self.driver.switch_to.default_content()

        # Intentar botón de cerrar, sino ESC
        try:
            close_btn = self.driver.find_element(
                By.CSS_SELECTOR,
                "button.ui-dialog-titlebar-close"
            )
            close_btn.click()
        except:
            from selenium.webdriver.common.keys import Keys
            webdriver.ActionChains(self.driver).send_keys(
                Keys.ESCAPE
            ).perform()

        time.sleep(2)
        return True

    def download_all(self):
        """Descarga datos de todas las estaciones"""
        self.driver.get("https://sinca.mma.gob.cl/index.php/region/index/id/M")
        time.sleep(5)

        pm25_icons = self.find_pm25_icons()

        success_count = 0
        for i, icon_info in enumerate(pm25_icons, 1):
            logger.info(f"[{i}/{len(pm25_icons)}] {icon_info['station']}")

            if self.download_station_data(icon_info):
                success_count += 1
                logger.info(f"✓ Success")
            else:
                logger.warning(f"✗ Failed")

            time.sleep(3)  # Pausa entre estaciones

        logger.info(f"Downloaded {success_count}/{len(pm25_icons)} stations")
        self.driver.quit()

# Uso
scraper = SINCAScraper(download_dir="data/external")
scraper.download_all()
```

---

## Patrones y Mejores Prácticas

### 1. Exploración Progresiva

```python
# Fase 1: Explorar manualmente con script simple
# Fase 2: Crear script de debug dedicado
# Fase 3: Integrar en downloader principal
```

### 2. Logging Detallado

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 3. Screenshots en Puntos Clave

```python
def take_screenshot(self, name):
    filepath = f"logs/{name}_{self.station}.png"
    self.driver.save_screenshot(filepath)
    logger.info(f"Screenshot: {filepath}")

# Usar en puntos críticos
self.take_screenshot("before_click")
icon.click()
self.take_screenshot("after_click")
```

### 4. Múltiples Estrategias de Búsqueda

```python
def find_element_robustly(driver, strategies):
    """Prueba múltiples estrategias hasta encontrar elemento"""
    for strategy_name, by, value in strategies:
        try:
            elements = driver.find_elements(by, value)
            if elements:
                logger.debug(f"Found with strategy: {strategy_name}")
                return elements
        except:
            continue
    return []

# Uso
strategies = [
    ("Partial link", By.PARTIAL_LINK_TEXT, "Excel CSV"),
    ("XPath href", By.XPATH, "//a[contains(@href, 'xcl')]"),
    ("XPath text", By.XPATH, "//a[contains(text(), 'CSV')]"),
]

elements = find_element_robustly(driver, strategies)
```

### 5. Esperas Adaptativas

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Espera explícita (recomendada)
wait = WebDriverWait(driver, 10)
element = wait.until(
    EC.presence_of_element_located((By.ID, "my-element"))
)

# Espera fija solo cuando no hay alternativa
time.sleep(5)  # Para contenido que carga dinámicamente sin señal
```

### 6. Manejo de Errores con Continue

```python
for item in items:
    try:
        process(item)
    except Exception as e:
        logger.error(f"Error processing {item}: {e}")
        take_screenshot(f"error_{item}")
        continue  # Continuar con siguiente item
```

---

## Debugging Checklist

Cuando un scraper no funciona, seguir este checklist:

### ✓ Nivel 1: Verificación Básica
- [ ] ¿La página carga correctamente en navegador normal?
- [ ] ¿El elemento es visible en el HTML?
- [ ] ¿Hay JavaScript que modifique la página después de cargar?

### ✓ Nivel 2: Selectores
- [ ] ¿El selector CSS/XPath es correcto?
- [ ] ¿El elemento existe en el contexto actual (frame/iframe)?
- [ ] ¿Hay múltiples elementos con el mismo selector?

### ✓ Nivel 3: Contexto
- [ ] ¿Estoy en el iframe/frame correcto?
- [ ] ¿El elemento está en un shadow DOM?
- [ ] ¿Hay ventanas/pestañas múltiples?

### ✓ Nivel 4: Timing
- [ ] ¿Esperé suficiente tiempo para que cargue?
- [ ] ¿Hay animaciones que interfieren?
- [ ] ¿El elemento aparece después de una acción del usuario?

### ✓ Nivel 5: Interacción
- [ ] ¿Otro elemento está bloqueando el click?
- [ ] ¿Necesito scroll para hacer visible el elemento?
- [ ] ¿JavaScript click funciona cuando Selenium click falla?

---

## Scripts de Debug Útiles

### Script 1: Explorador de Iframes

```python
def explore_iframes(driver, level=0):
    """Explora recursivamente todos los iframes"""
    indent = "  " * level

    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    print(f"{indent}Found {len(iframes)} iframe(s) at level {level}")

    for i, iframe in enumerate(iframes):
        src = iframe.get_attribute('src')
        print(f"{indent}Iframe {i}: {src[:80]}...")

        driver.switch_to.frame(iframe)
        explore_iframes(driver, level + 1)
        driver.switch_to.parent_frame()
```

### Script 2: Contador de Elementos

```python
def count_elements_by_tag(driver):
    """Cuenta todos los elementos por tag"""
    script = """
    var tags = {};
    var all = document.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
        var tag = all[i].tagName.toLowerCase();
        tags[tag] = (tags[tag] || 0) + 1;
    }
    return tags;
    """

    counts = driver.execute_script(script)

    print("Element counts:")
    for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")
```

### Script 3: Buscador de Texto

```python
def find_text_in_page(driver, search_text):
    """Encuentra dónde aparece un texto específico"""
    script = f"""
    var search = "{search_text}";
    var results = [];
    var all = document.querySelectorAll('*');

    for (var i = 0; i < all.length; i++) {{
        var text = all[i].textContent;
        if (text.includes(search) && text.length < 200) {{
            results.push({{
                tag: all[i].tagName,
                text: text.substring(0, 100),
                id: all[i].id,
                className: all[i].className
            }});
        }}
    }}

    return results;
    """

    results = driver.execute_script(script)

    print(f"Found '{search_text}' in {len(results)} elements:")
    for r in results[:10]:  # Primeros 10
        print(f"  {r['tag']}: {r['text'][:50]}...")
```

---

## Aplicación a Otras Páginas

### Checklist de Adaptación

Para adaptar esta metodología a otra página web:

1. **Análisis inicial**
   - [ ] Inspeccionar estructura HTML con DevTools
   - [ ] Identificar si usa iframes/frames
   - [ ] Determinar tipo de elementos (links, botones, etc.)
   - [ ] Verificar si hay JavaScript dinámico

2. **Crear script de exploración**
   - [ ] Copiar `debug_iframe_content.py` como template
   - [ ] Adaptar URL objetivo
   - [ ] Modificar selectores de búsqueda
   - [ ] Ejecutar y analizar resultados

3. **Desarrollar selectores**
   - [ ] Probar selectores en consola de Chrome
   - [ ] Crear múltiples estrategias de búsqueda
   - [ ] Validar con JavaScript cuando sea necesario

4. **Implementar navegación**
   - [ ] Mapear estructura de iframes
   - [ ] Implementar cambios de contexto
   - [ ] Agregar esperas apropiadas

5. **Integrar descarga**
   - [ ] Configurar directorio de descarga
   - [ ] Implementar detección de archivos nuevos
   - [ ] Manejar cierres de modal

6. **Testing y refinamiento**
   - [ ] Probar con headless y sin headless
   - [ ] Agregar logging detallado
   - [ ] Implementar manejo de errores
   - [ ] Optimizar tiempos de espera

---

## Lecciones Aprendidas

### ✅ Hacer
1. **Explorar antes de programar** - Invertir tiempo en entender la estructura ahorra horas de debugging
2. **Usar JavaScript como herramienta de exploración** - Es más poderoso que Selenium para inspección
3. **Screenshots en cada paso crítico** - Permiten debugging post-mortem
4. **Múltiples estrategias de búsqueda** - Aumenta robustez del scraper
5. **Esperas generosas para contenido dinámico** - Framesets antiguos cargan lento
6. **Click con JavaScript como fallback** - Más confiable que Selenium click
7. **Logging detallado con niveles** - INFO para progreso, DEBUG para detalles
8. **Scripts de debug separados** - Más rápido que modificar script principal

### ❌ Evitar
1. **Asumir estructura HTML sin verificar** - Siempre inspeccionar primero
2. **Usar solo selectores simples** - Pueden fallar con cambios mínimos
3. **Esperas fijas muy cortas** - Causan fallos intermitentes
4. **Ignorar el contexto de iframes** - La causa #1 de "element not found"
5. **Click directo sin verificar visibilidad** - Causa ElementClickIntercepted
6. **No manejar errores individuales** - Un fallo no debe detener todo
7. **Depender solo de elementos visibles** - A veces están ocultos pero clickeables
8. **No limpiar el contexto** - Siempre volver a default_content

---

## Métricas de Éxito

Para el caso de SINCA:
- ✅ **Tasa de éxito**: 76.9% (10/13 estaciones)
- ✅ **Datos descargados**: ~1.2 MB (>1000 registros por estación)
- ✅ **Tiempo de ejecución**: ~30 minutos para 13 estaciones
- ✅ **Estabilidad**: 0 crashes, solo 3 fallos de descarga individuales
- ✅ **Código reutilizable**: Script aplicable a otras regiones de SINCA

---

## Referencias

- Selenium Documentation: https://www.selenium.dev/documentation/
- ChromeDriver: https://chromedriver.chromium.org/
- Webdriver Manager: https://github.com/SergeyPirogov/webdriver_manager

---

## Conclusión

Esta metodología proporciona un framework robusto para automatizar descargas desde páginas web complejas con iframes anidados. La clave del éxito está en:

1. **Exploración metódica** antes de programar
2. **Múltiples estrategias** de búsqueda y acción
3. **Manejo robusto de errores** con continue
4. **Logging y screenshots** para debugging
5. **Testing incremental** con scripts dedicados

El resultado es un scraper confiable que puede adaptarse fácilmente a otros sitios con estructuras similares.
