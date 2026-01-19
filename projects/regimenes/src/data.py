"""
Módulo de descarga y preparación de datos.
"""

import pandas as pd
import yfinance as yf
from typing import Optional

import sys
sys.path.append("..")
from config import TICKER, START_DATE, END_DATE, TRAIN_END


def download_spy_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Descarga datos históricos de un ticker usando yfinance.

    Parameters
    ----------
    ticker : str
        Símbolo del activo (default: SPY)
    start : str
        Fecha inicio en formato 'YYYY-MM-DD'
    end : str
        Fecha fin en formato 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas OHLCV y 'returns' (retornos logarítmicos)
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Aplanar MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calcular retornos logarítmicos
    df["returns"] = pd.np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    return df


def train_test_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en conjuntos In-Sample y Out-of-Sample.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos completos
    train_end : str
        Fecha de corte para el conjunto de entrenamiento

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test) - Datos IS y OOS
    """
    df_train = df.loc[:train_end].copy()
    df_test = df.loc[train_end:].iloc[1:].copy()  # Excluir fecha de corte

    return df_train, df_test


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Genera resumen estadístico de los datos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'returns'

    Returns
    -------
    dict
        Diccionario con estadísticas descriptivas
    """
    returns = df["returns"]

    return {
        "n_observations": len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "mean_return_annual": returns.mean() * 252,
        "std_return_annual": returns.std() * (252 ** 0.5),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "min_return": returns.min(),
        "max_return": returns.max(),
    }
