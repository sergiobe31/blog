"""
Módulo de identificación de regímenes de mercado.

Incluye:
    - K-Means clustering
    - Gaussian Mixture Models
    - Matriz de transición
    - Métricas de evaluación
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Optional

import sys
sys.path.append("..")
from config import K_REGIMES, RANDOM_STATE


def standardize_features(
    features: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Estandariza features (media=0, std=1).

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame con features
    scaler : StandardScaler, optional
        Scaler pre-entrenado (para OOS)

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        (features_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(features)
    else:
        scaled_values = scaler.transform(features)

    features_scaled = pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=features.columns,
    )

    return features_scaled, scaler


def fit_kmeans(
    features: pd.DataFrame,
    k: int = K_REGIMES,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """
    Entrena K-Means y asigna regímenes.

    Parameters
    ----------
    features : pd.DataFrame
        Features estandarizados
    k : int
        Número de clusters
    random_state : int
        Semilla para reproducibilidad

    Returns
    -------
    tuple[KMeans, np.ndarray]
        (modelo, labels)
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(features)

    return model, labels


def fit_gmm(
    features: pd.DataFrame,
    k: int = K_REGIMES,
    random_state: int = RANDOM_STATE,
) -> tuple[GaussianMixture, np.ndarray]:
    """
    Entrena Gaussian Mixture Model y asigna regímenes.

    Parameters
    ----------
    features : pd.DataFrame
        Features estandarizados
    k : int
        Número de componentes
    random_state : int
        Semilla para reproducibilidad

    Returns
    -------
    tuple[GaussianMixture, np.ndarray, np.ndarray]
        (modelo, labels)
    """
    model = GaussianMixture(
        n_components=k,
        random_state=random_state,
        covariance_type="full",
    )
    model.fit(features)
    labels = model.predict(features)

    return model, labels


def evaluate_k_range(
    features: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Evalúa K-Means para un rango de K (codo y silhouette).

    Parameters
    ----------
    features : pd.DataFrame
        Features estandarizados
    k_range : range
        Rango de valores de K a evaluar
    random_state : int
        Semilla para reproducibilidad

    Returns
    -------
    pd.DataFrame
        DataFrame con inertia y silhouette score para cada K
    """
    results = []

    for k in k_range:
        model, labels = fit_kmeans(features, k=k, random_state=random_state)
        sil_score = silhouette_score(features, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": sil_score,
        })

    return pd.DataFrame(results)


def compute_transition_matrix(labels: np.ndarray, k: int = K_REGIMES) -> np.ndarray:
    """
    Calcula matriz de transición empírica entre regímenes.

    Parameters
    ----------
    labels : np.ndarray
        Array de etiquetas de régimen
    k : int
        Número de regímenes

    Returns
    -------
    np.ndarray
        Matriz de transición (k x k), filas suman 1
    """
    transitions = np.zeros((k, k))

    for i in range(len(labels) - 1):
        from_regime = labels[i]
        to_regime = labels[i + 1]
        transitions[from_regime, to_regime] += 1

    # Normalizar filas para obtener probabilidades
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Evitar división por cero
    transition_matrix = transitions / row_sums

    return transition_matrix


def get_regime_statistics(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int = K_REGIMES,
) -> pd.DataFrame:
    """
    Calcula estadísticas de retornos por régimen.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'returns'
    labels : np.ndarray
        Array de etiquetas de régimen
    k : int
        Número de regímenes

    Returns
    -------
    pd.DataFrame
        Estadísticas por régimen
    """
    # Alinear labels con df
    df_aligned = df.iloc[-len(labels):].copy()
    df_aligned["regime"] = labels

    stats = []
    for regime in range(k):
        regime_returns = df_aligned[df_aligned["regime"] == regime]["returns"]

        stats.append({
            "regime": regime,
            "count": len(regime_returns),
            "pct_time": len(regime_returns) / len(df_aligned) * 100,
            "mean_return_annual": regime_returns.mean() * 252,
            "std_return_annual": regime_returns.std() * (252 ** 0.5),
            "sharpe": (regime_returns.mean() * 252) / (regime_returns.std() * (252 ** 0.5)),
            "min_return": regime_returns.min(),
            "max_return": regime_returns.max(),
        })

    return pd.DataFrame(stats)


def compute_centroid_distances(
    features: pd.DataFrame,
    model: KMeans,
) -> pd.DataFrame:
    """
    Calcula distancia de cada observación a todos los centroides.

    Parameters
    ----------
    features : pd.DataFrame
        Features estandarizados
    model : KMeans
        Modelo K-Means entrenado

    Returns
    -------
    pd.DataFrame
        DataFrame con distancias a cada centroide
    """
    centroids = model.cluster_centers_
    distances = {}

    for i, centroid in enumerate(centroids):
        distances[f"dist_centroid_{i}"] = np.linalg.norm(
            features.values - centroid, axis=1
        )

    return pd.DataFrame(distances, index=features.index)
