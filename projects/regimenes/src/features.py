"""
Módulo de cálculo de features para identificación de regímenes.

Features de Oliva:
    - Momentum: retorno acumulado en ventana
    - Volatilidad: desviación estándar de retornos en ventana

Features alternativas:
    - Downside Deviation: volatilidad solo de retornos negativos
    - Average Return: media de retornos en ventana
    - Sortino Ratio: retorno medio / downside deviation
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
sys.path.append("..")
from config import WINDOW_SIZES


def compute_momentum(returns: pd.Series, window: int) -> pd.Series:
    """
    Calcula momentum como retorno acumulado en ventana.

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos logarítmicos
    window : int
        Tamaño de la ventana en días

    Returns
    -------
    pd.Series
        Momentum (suma de retornos en ventana)
    """
    return returns.rolling(window=window).sum()


def compute_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Calcula volatilidad como desviación estándar en ventana.

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos logarítmicos
    window : int
        Tamaño de la ventana en días

    Returns
    -------
    pd.Series
        Volatilidad (std de retornos en ventana)
    """
    return returns.rolling(window=window).std()


def compute_downside_deviation(returns: pd.Series, window: int) -> pd.Series:
    """
    Calcula Downside Deviation (volatilidad de retornos negativos).

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos logarítmicos
    window : int
        Tamaño de la ventana en días

    Returns
    -------
    pd.Series
        Downside deviation
    """
    negative_returns = returns.clip(upper=0)
    return negative_returns.rolling(window=window).std()


def compute_average_return(returns: pd.Series, window: int) -> pd.Series:
    """
    Calcula retorno medio en ventana.

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos logarítmicos
    window : int
        Tamaño de la ventana en días

    Returns
    -------
    pd.Series
        Media de retornos en ventana
    """
    return returns.rolling(window=window).mean()


def compute_sortino_ratio(returns: pd.Series, window: int) -> pd.Series:
    """
    Calcula Sortino Ratio (retorno medio / downside deviation).

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos logarítmicos
    window : int
        Tamaño de la ventana en días

    Returns
    -------
    pd.Series
        Sortino ratio
    """
    avg_return = compute_average_return(returns, window)
    downside_dev = compute_downside_deviation(returns, window)

    # Evitar división por cero
    return avg_return / downside_dev.replace(0, np.nan)


def compute_oliva_features(
    df: pd.DataFrame,
    windows: list[int] = WINDOW_SIZES,
) -> pd.DataFrame:
    """
    Calcula features de Oliva (momentum y volatilidad) para múltiples ventanas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'returns'
    windows : list[int]
        Lista de tamaños de ventana

    Returns
    -------
    pd.DataFrame
        DataFrame con features de Oliva
    """
    returns = df["returns"]
    features = pd.DataFrame(index=df.index)

    for w in windows:
        features[f"momentum_{w}d"] = compute_momentum(returns, w)
        features[f"volatility_{w}d"] = compute_volatility(returns, w)

    return features.dropna()


def compute_alternative_features(
    df: pd.DataFrame,
    windows: list[int] = WINDOW_SIZES,
) -> pd.DataFrame:
    """
    Calcula features alternativas para múltiples ventanas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'returns'
    windows : list[int]
        Lista de tamaños de ventana

    Returns
    -------
    pd.DataFrame
        DataFrame con features alternativas
    """
    returns = df["returns"]
    features = pd.DataFrame(index=df.index)

    for w in windows:
        features[f"downside_dev_{w}d"] = compute_downside_deviation(returns, w)
        features[f"avg_return_{w}d"] = compute_average_return(returns, w)
        features[f"sortino_{w}d"] = compute_sortino_ratio(returns, w)

    return features.dropna()


def compute_all_features(
    df: pd.DataFrame,
    windows: list[int] = WINDOW_SIZES,
) -> pd.DataFrame:
    """
    Calcula todos los features (Oliva + alternativas).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'returns'
    windows : list[int]
        Lista de tamaños de ventana

    Returns
    -------
    pd.DataFrame
        DataFrame con todos los features
    """
    oliva = compute_oliva_features(df, windows)
    alternative = compute_alternative_features(df, windows)

    # Alinear índices
    common_index = oliva.index.intersection(alternative.index)

    return pd.concat([oliva.loc[common_index], alternative.loc[common_index]], axis=1)
