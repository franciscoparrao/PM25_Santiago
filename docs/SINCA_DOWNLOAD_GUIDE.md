# Guía de Descarga de Datos SINCA

**Sistema de Información Nacional de Calidad del Aire (SINCA)**
**Ministerio del Medio Ambiente - Chile**

---

## 📋 Información General

### ¿Qué es SINCA?

SINCA es la plataforma oficial del gobierno chileno que consolida mediciones de calidad del aire de todas las estaciones de monitoreo del país. Para Santiago, hay **32 estaciones de monitoreo** que miden PM2.5, PM10, O₃, NO₂, SO₂, y CO en tiempo real.

### Datos que Necesitamos

- **Contaminante**: PM2.5 (Material Particulado Fino)
- **Región**: Metropolitana de Santiago
- **Período**: 2019-01-01 a 2025-11-10
- **Frecuencia**: Horaria (para agregar a diaria)
- **Formato**: CSV

---

## 🌐 Método 1: Descarga Manual desde Web (RECOMENDADO)

### Paso 1: Acceder al Portal

1. Ir a: **https://sinca.mma.gob.cl/**
2. Click en **"Descarga de Datos"** en el menú superior
3. O directamente: **https://sinca.mma.gob.cl/index.php/datos/descarga**

### Paso 2: Configurar la Descarga

**Filtros a aplicar:**

```
┌─────────────────────────────────────────────────┐
│ CONFIGURACIÓN DE DESCARGA                       │
├─────────────────────────────────────────────────┤
│ Región:          Metropolitana de Santiago      │
│ Contaminante:    MP2.5 (Material Particulado)   │
│ Fecha Inicio:    01/01/2019                     │
│ Fecha Fin:       10/11/2025                     │
│ Frecuencia:      Horaria                        │
│ Formato:         CSV                            │
└─────────────────────────────────────────────────┘
```

### Paso 3: Seleccionar Estaciones

**Estaciones en Santiago (32 totales):**

#### Zona Norte:
- Colina
- Quilicura
- Independencia
- Cerro Navia
- Pudahuel

#### Zona Centro:
- Santiago Centro
- Parque O'Higgins
- Las Condes
- Providencia
- Ñuñoa
- La Reina
- Vitacura

#### Zona Sur:
- El Bosque
- La Florida
- Puente Alto
- San Bernardo

#### Zona Oeste:
- Lo Prado
- Maipú
- Cerrillos
- Talagante
- Peñaflor
- Melipilla

**Recomendación:** Seleccionar **TODAS las estaciones** disponibles para maximizar cobertura espacial.

### Paso 4: Descargar

1. Click en **"Generar Descarga"**
2. El sistema generará un archivo CSV (puede tardar varios minutos)
3. Guardar el archivo como:
   ```
   data/external/sinca_pm25_raw.csv
   ```

### Formato Esperado

El CSV descargado tendrá este formato:

```csv
fecha,hora,estacion,region,contaminante,valor,unidad
2019-01-01,00:00,Cerrillos,Metropolitana,MP2.5,25.3,µg/m³N
2019-01-01,01:00,Cerrillos,Metropolitana,MP2.5,23.1,µg/m³N
...
```

**Columnas importantes:**
- `fecha`: Fecha de la medición
- `hora`: Hora de la medición (00:00 a 23:00)
- `estacion`: Nombre de la estación
- `valor`: Concentración de PM2.5 en µg/m³

---

## 🐍 Método 2: Descarga Programática con Python

Si la descarga manual es muy lenta o falla, puedes usar web scraping:

### Opción A: Script Automático (Simple)

Ya tenemos un script preparado: `src/data_acquisition/sinca_scraper.py`

```bash
python3 src/data_acquisition/sinca_scraper.py \
    --start-date 2019-01-01 \
    --end-date 2025-11-10 \
    --contaminant PM25
```

### Opción B: Descarga por Chunks (Para períodos largos)

```python
# Descargar año por año para evitar timeouts
python3 src/data_acquisition/sinca_scraper.py --start-date 2019-01-01 --end-date 2019-12-31
python3 src/data_acquisition/sinca_scraper.py --start-date 2020-01-01 --end-date 2020-12-31
python3 src/data_acquisition/sinca_scraper.py --start-date 2021-01-01 --end-date 2021-12-31
# ... etc
```

Luego combinar los archivos:

```bash
cat data/external/sinca_2019.csv > data/external/sinca_pm25_raw.csv
tail -n +2 data/external/sinca_2020.csv >> data/external/sinca_pm25_raw.csv
tail -n +2 data/external/sinca_2021.csv >> data/external/sinca_pm25_raw.csv
# ... etc
```

---

## 📊 Método 3: API de SINCA (Experimental)

⚠️ **Nota**: SINCA tiene una API pero no está bien documentada y puede ser inestable.

### Endpoint Base

```
https://sinca.mma.gob.cl/index.php/json/listadomapa2k19
```

### Parámetros

- `timestamp`: Fecha/hora en formato Unix
- `estaciones`: IDs de estaciones separados por coma

### Ejemplo con curl:

```bash
curl "https://sinca.mma.gob.cl/index.php/json/listadomapa2k19?timestamp=1546300800" \
  -H "User-Agent: Mozilla/5.0" \
  > sinca_response.json
```

**Problema**: La API solo devuelve datos recientes (últimas 24-48 horas), no históricos.

---

## 🔧 Método 4: Descarga desde Archivo Consolidado (Si está disponible)

SINCA ocasionalmente publica datasets consolidados en:

- **Portal de Datos Abiertos**: https://datos.gob.cl/
- **Búsqueda**: "SINCA PM2.5 Santiago"

Si encuentras un dataset consolidado:

```bash
# Descargar directamente
wget https://datos.gob.cl/dataset/[id]/sinca-pm25-santiago.csv \
  -O data/external/sinca_pm25_raw.csv
```

---

## ✅ Verificación de Datos Descargados

Una vez descargado, verifica el archivo:

```bash
# Ver primeras líneas
head -20 data/external/sinca_pm25_raw.csv

# Contar registros
wc -l data/external/sinca_pm25_raw.csv

# Ver estaciones únicas
cut -d',' -f3 data/external/sinca_pm25_raw.csv | sort -u

# Ver rango de fechas
cut -d',' -f1 data/external/sinca_pm25_raw.csv | sort | uniq | head -5
cut -d',' -f1 data/external/sinca_pm25_raw.csv | sort | uniq | tail -5
```

### Estadísticas Esperadas

```
Período:           2019-01-01 a 2025-11-10 (2,506 días)
Estaciones:        ~32 estaciones
Mediciones/hora:   24 mediciones/día × 32 estaciones = 768 mediciones/día
Total esperado:    ~1,900,000 registros (con datos faltantes: ~1,500,000)
Tamaño archivo:    ~80-120 MB
```

---

## 🚨 Problemas Comunes y Soluciones

### Problema 1: El sitio SINCA es lento / timeout

**Solución:**
- Descargar por períodos más cortos (6 meses en vez de 6 años)
- Intentar en horarios de menor tráfico (madrugada)
- Usar descarga programática con reintentos

### Problema 2: Datos faltantes en períodos específicos

**Normal**: SINCA tiene gaps por:
- Mantenimiento de estaciones
- Fallas técnicas
- Calibraciones

**Solución:** Aceptar los datos disponibles, luego interpolar o excluir esos períodos.

### Problema 3: Formato inconsistente

**Solución:** Nuestro script de preprocessing manejará:
- Diferentes formatos de fecha
- Nombres de columnas inconsistentes
- Valores faltantes (-999, NaN, etc.)

### Problema 4: Coordenadas de estaciones no están en el CSV

**Solución:** Tenemos las coordenadas en `config/config.yaml`. Si faltan:

```bash
# Extraer coordenadas desde el mapa interactivo de SINCA
# O usar este listado aproximado:

Cerrillos:      -33.50, -70.71
El Bosque:      -33.56, -70.67
La Florida:     -33.52, -70.60
Las Condes:     -33.37, -70.52
Pudahuel:       -33.42, -70.75
# etc...
```

---

## 📝 Script de Verificación Rápida

Guarda esto como `check_sinca.py`:

```python
import pandas as pd

# Cargar datos
df = pd.read_csv('data/external/sinca_pm25_raw.csv')

print(f"Total records: {len(df):,}")
print(f"Date range: {df['fecha'].min()} to {df['fecha'].max()}")
print(f"Stations: {df['estacion'].nunique()}")
print(f"\nStation list:")
print(df['estacion'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nPM2.5 statistics:")
print(df['valor'].describe())
```

Ejecutar:

```bash
python3 check_sinca.py
```

---

## 🎯 Siguiente Paso Después de la Descarga

Una vez que tengas `data/external/sinca_pm25_raw.csv`:

```bash
# 1. Verificar datos
python3 check_sinca.py

# 2. Ejecutar preprocessing
python3 src/data_preprocessing/01_clean_sinca.py

# 3. Match espacial-temporal con datos satelitales
python3 src/data_preprocessing/02_spatial_matching.py

# 4. Crear dataset master
python3 src/data_preprocessing/03_create_master_dataset.py
```

---

## 📚 Recursos Adicionales

### Documentación Oficial

- **SINCA Portal**: https://sinca.mma.gob.cl/
- **Metodología**: https://sinca.mma.gob.cl/index.php/pagina/index/id/metodologia
- **Manual de Usuario**: https://sinca.mma.gob.cl/archivos/MANUAL_SINCA.pdf

### Datos Alternativos (Backup)

Si SINCA no está disponible:

1. **OpenAQ**: https://openaq.org/ (tiene datos de Santiago)
2. **IQAIR**: https://www.iqair.com/chile/santiago (datos recientes)
3. **Datos históricos académicos**: Contactar DICTUC o Universidad de Chile

### Contacto SINCA

- Email: sinca@mma.gob.cl
- Teléfono: +56 2 2573 5600

---

## ✨ Tips Pro

### 1. Descarga incremental

Si ya tienes datos hasta 2023, solo descarga 2024-2025:

```bash
python3 src/data_acquisition/sinca_scraper.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-10 \
    --output data/external/sinca_2024_2025.csv
```

### 2. Validación cruzada

Compara medias mensuales de SINCA con valores de MODIS AOD para detectar inconsistencias.

### 3. Backup automático

```bash
# Después de descargar, hacer backup
cp data/external/sinca_pm25_raw.csv \
   data/external/backups/sinca_pm25_$(date +%Y%m%d).csv
```

---

## ⏱️ Tiempo Estimado

| Método | Tiempo | Dificultad |
|--------|--------|------------|
| **Web Manual** | 15-30 minutos | Fácil ⭐ |
| **Script Python (año a año)** | 1-2 horas | Media ⭐⭐ |
| **Script Python (bulk)** | 30-60 minutos | Media ⭐⭐ |
| **API** | Variable | Difícil ⭐⭐⭐ |

**Recomendación**: Empezar con descarga manual del sitio web. Si falla, usar script Python.

---

**Última actualización**: 2025-11-12
**Autor**: Claude Code Assistant
**Proyecto**: PM2.5 Santiago - Air Quality Prediction
