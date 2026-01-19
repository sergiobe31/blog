"""
Módulo de análisis de regímenes de mercado.

Componentes:
    - data: Descarga y preparación de datos
    - features: Cálculo de features (Oliva, alternativas)
    - regimes: K-Means, GMM, matriz de transición
    - plotting: Funciones de visualización
"""

from . import data
from . import features
from . import regimes
from . import plotting

__all__ = ["data", "features", "regimes", "plotting"]
