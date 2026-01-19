"""
Configuración global del proyecto de análisis de regímenes de mercado.
Centraliza parámetros para garantizar consistencia entre notebooks.
"""

# =============================================================================
# DATOS
# =============================================================================
TICKER = "SPY"
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# Split In-Sample / Out-of-Sample
TRAIN_END = "2020-12-31"

# =============================================================================
# FEATURES
# =============================================================================
WINDOW_SIZES = [5, 10, 20, 30, 40, 50]  # Días para momentum y volatilidad

# =============================================================================
# MODELO
# =============================================================================
K_REGIMES = 4  # Número de regímenes (ajustar tras análisis de codo/silhouette)
RANDOM_STATE = 42  # Reproducibilidad

# =============================================================================
# BACKTEST (Notebook 3)
# =============================================================================
WALK_FORWARD_RETRAIN_MONTHS = 12  # Re-entrenar cada N meses
WALK_FORWARD_WINDOW_YEARS = 5     # Ventana rolling de entrenamiento

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# Colores por régimen (consistentes en todos los gráficos)
REGIME_COLORS = {
    0: "#2ecc71",  # Verde - Bull tranquilo
    1: "#e74c3c",  # Rojo - Bear volátil
    2: "#3498db",  # Azul - Transición/Neutro
    3: "#f39c12",  # Naranja - Bull volátil
}

REGIME_NAMES = {
    0: "Bull Tranquilo",
    1: "Bear Volátil",
    2: "Neutro",
    3: "Bull Volátil",
}
