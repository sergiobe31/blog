# Análisis de Regímenes de Mercado

Proyecto de análisis de regímenes de mercado usando técnicas de clustering (K-Means, GMM) aplicadas al SPY.

## Estructura del Proyecto

```
blog_regimenes/
├── data/                    # Datos descargados (no versionados)
├── figures/                 # Figuras exportadas para el blog
│   ├── post1/              # Notebook 1: Identificación
│   ├── post2/              # Notebook 2: Limitaciones
│   └── post3/              # Notebook 3: Backtest
├── notebooks/
│   ├── 01_identificacion_regimenes.ipynb
│   ├── 02_limitaciones_modelo.ipynb
│   └── 03_backtest_extensiones.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py             # Descarga y preparación de datos
│   ├── features.py         # Cálculo de features (Oliva, alternativas)
│   ├── regimes.py          # K-Means, GMM, matriz de transición
│   └── plotting.py         # Funciones de visualización
├── config.py               # Parámetros globales
├── requirements.txt        # Dependencias
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso en Google Colab

Cada notebook es independiente y puede ejecutarse en Colab:

```python
!pip install -q yfinance statsmodels networkx
!git clone https://github.com/tu-usuario/blog_regimenes.git
%cd blog_regimenes/notebooks
```

## Notebooks

### 1. Identificación de Regímenes
- Descarga de datos SPY
- Features de Oliva (momentum, volatilidad)
- Selección de K (codo, silhouette)
- Análisis de centroides
- Matriz de transición
- Gaussian Mixture Models

### 2. Limitaciones del Modelo
- Propiedades de la matriz de transición
- Test de memoria de duración
- Volatility clustering (ACF)
- Fat tails y comparación con distribuciones
- Drift IS vs OOS

### 3. Backtest y Extensiones
- Estrategia basada en regímenes
- Walk-forward validation (5 años, re-entrenar cada 12 meses)
- Comparación vs Buy & Hold
- Generación de series sintéticas
- Extensiones futuras (Jump Models, ML)

## Parámetros (config.py)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| TICKER | SPY | Activo analizado |
| START_DATE | 2000-01-01 | Inicio de datos |
| END_DATE | 2023-12-31 | Fin de datos |
| TRAIN_END | 2020-12-31 | Split IS/OOS |
| K_REGIMES | 4 | Número de regímenes |
| RANDOM_STATE | 42 | Reproducibilidad |

## Autor

Sergio Berganzo de Miguel
Máster en ML/DL - Instituto BME + UPM
