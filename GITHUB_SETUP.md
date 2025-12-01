# Instrucciones para Crear el Repositorio GitHub

Este documento explica cómo preparar y subir el repositorio a GitHub para que coincida con lo declarado en el paper.

## URL Declarada en el Paper

```
https://github.com/franciscoparraUSACH/PM25_Santiago
```

## Pasos para Crear el Repositorio

### 1. Crear el Repositorio en GitHub

1. Ir a https://github.com/new
2. **Repository name**: `PM25_Santiago`
3. **Description**: `PM2.5 Forecasting and Spatial Interpolation for Santiago, Chile - Code for Parra & Astudillo (2025)`
4. **Visibility**: Public
5. **NO inicializar** con README, .gitignore o LICENSE (ya los tenemos)
6. Click "Create repository"

### 2. Preparar el Directorio Local

```bash
cd /home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago

# Renombrar archivos para GitHub
mv README.md README_DEVELOPMENT.md        # Guardar el README de desarrollo
mv README_GITHUB.md README.md             # Usar el README profesional
mv .gitignore .gitignore_development      # Guardar gitignore de desarrollo
mv .gitignore_github .gitignore           # Usar gitignore para GitHub
```

### 3. Inicializar Git y Subir

```bash
# Inicializar repositorio
git init

# Agregar archivos
git add .

# Verificar qué se va a incluir
git status

# Commit inicial
git commit -m "Initial commit: PM2.5 forecasting and spatial interpolation code

Paper: Parra & Astudillo (2025), Environmental Modelling & Software
- XGBoost temporal forecasting (R² = 0.76)
- Regression Kriging spatial interpolation (R² = 0.89)
- Walk-forward validation (2,161 iterations)
- LOSO-CV for spatial validation"

# Agregar remote
git remote add origin https://github.com/franciscoparraUSACH/PM25_Santiago.git

# Subir
git branch -M main
git push -u origin main
```

### 4. Verificar Contenido

Después de subir, verificar que el repositorio contenga:

```
PM25_Santiago/
├── README.md                    ✓ (profesional, para paper)
├── LICENSE                      ✓ (MIT)
├── requirements.txt             ✓ (versiones flexibles)
├── requirements_paper.txt       ✓ (versiones exactas del paper)
├── .gitignore                   ✓
├── src/
│   ├── temporal/
│   │   ├── forecasting.py      ✓
│   │   ├── temporal_models.py  ✓
│   │   └── ...
│   ├── spatial/
│   │   ├── regression_kriging.py ✓
│   │   └── ...
│   ├── data_acquisition/       ✓
│   ├── data_processing/        ✓
│   └── utils/                  ✓
├── data/
│   ├── README.md               ✓
│   └── processed/              ✓ (datasets procesados)
└── results/
    └── figures/                ✓ (figuras del paper)
```

## Archivos que NO Deben Subirse

El `.gitignore` excluye automáticamente:
- `data/raw/` - Datos crudos (muy grandes, usuarios deben descargar)
- `elsarticle/` - Manuscrito LaTeX (separado del código)
- `cache/`, `logs/` - Archivos temporales
- `*.log`, `*.zip` - Archivos generados
- Archivos de desarrollo (`test_*.py`, `debug_*.py`, etc.)

## Verificación Final

Después de subir, visitar:
```
https://github.com/franciscoparraUSACH/PM25_Santiago
```

Y verificar que:
1. El README se muestra correctamente
2. La estructura de carpetas es clara
3. Los archivos `.py` principales están presentes
4. Los datasets en `data/processed/` están disponibles
5. La LICENSE MIT está visible

## Tamaño Estimado del Repositorio

- Código Python: ~500 KB
- Datasets procesados: ~48 MB
- Figuras: ~5 MB
- **Total**: ~55 MB (dentro del límite de GitHub de 100 MB por archivo)

## Notas Importantes

1. **Usuario correcto**: El paper declara `franciscoparraUSACH`, verificar que el usuario de GitHub sea exactamente ese.

2. **Datasets grandes**: Si algún archivo excede 100 MB, usar Git LFS o subirlo a Zenodo/Figshare y enlazar.

3. **Actualizar paper si es necesario**: Si la estructura real difiere de lo declarado, actualizar la sección "Software and Data Availability" del paper.
