"""
Módulo de visualización para análisis de regímenes.

Todas las funciones guardan figuras automáticamente si se especifica save_path.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional

import sys
sys.path.append("..")
from config import REGIME_COLORS, REGIME_NAMES, FIGURE_DPI, FIGURE_FORMAT


def setup_plotting_style():
    """Configura estilo global de matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = FIGURE_DPI
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def save_figure(fig, save_path: Optional[str], tight: bool = True):
    """Helper para guardar figuras."""
    if save_path:
        if tight:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
        else:
            fig.savefig(save_path, dpi=FIGURE_DPI,
                       facecolor="white", edgecolor="none")
        print(f"Figura guardada: {save_path}")


def plot_correlation_heatmap(
    features: pd.DataFrame,
    title: str = "Correlación entre Features",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Genera heatmap de correlación entre features.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame con features
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title)
    plt.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_elbow_silhouette(
    eval_df: pd.DataFrame,
    title: str = "Selección de K",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Genera gráfico de codo y silhouette side by side.

    Parameters
    ----------
    eval_df : pd.DataFrame
        DataFrame con columnas 'k', 'inertia', 'silhouette'
    title : str
        Título general
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Método del codo
    axes[0].plot(eval_df["k"], eval_df["inertia"], "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Número de Clusters (K)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Método del Codo")
    axes[0].set_xticks(eval_df["k"])

    # Silhouette
    axes[1].plot(eval_df["k"], eval_df["silhouette"], "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Número de Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xticks(eval_df["k"])

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_centroids_radar(
    centroids: np.ndarray,
    feature_names: list[str],
    title: str = "Centroides por Régimen",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Genera radar chart de centroides.

    Parameters
    ----------
    centroids : np.ndarray
        Array de centroides (k x n_features)
    feature_names : list[str]
        Nombres de los features
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, centroid in enumerate(centroids):
        values = centroid.tolist()
        values += values[:1]  # Cerrar el círculo

        ax.plot(
            angles, values,
            "o-", linewidth=2,
            label=REGIME_NAMES.get(i, f"Régimen {i}"),
            color=REGIME_COLORS.get(i, f"C{i}"),
        )
        ax.fill(angles, values, alpha=0.25, color=REGIME_COLORS.get(i, f"C{i}"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=9)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_transition_graph(
    transition_matrix: np.ndarray,
    threshold: float = 0.05,
    title: str = "Grafo de Transiciones",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Visualiza matriz de transición como grafo dirigido.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Matriz de transición (k x k)
    threshold : float
        Probabilidad mínima para mostrar arista
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    k = len(transition_matrix)
    G = nx.DiGraph()

    # Añadir nodos
    for i in range(k):
        G.add_node(i, label=REGIME_NAMES.get(i, f"R{i}"))

    # Añadir aristas con peso > threshold
    for i in range(k):
        for j in range(k):
            if transition_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=transition_matrix[i, j])

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.circular_layout(G)
    node_colors = [REGIME_COLORS.get(i, f"C{i}") for i in range(k)]

    # Dibujar nodos
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9,
    )

    # Dibujar etiquetas
    labels = {i: REGIME_NAMES.get(i, f"R{i}") for i in range(k)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight="bold")

    # Dibujar aristas con grosor proporcional al peso
    edges = G.edges(data=True)
    for u, v, d in edges:
        weight = d["weight"]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=weight * 5,
            alpha=0.7,
            edge_color="gray",
            connectionstyle="arc3,rad=0.1" if u != v else "arc3,rad=0.3",
            arrows=True,
            arrowsize=20,
        )

        # Añadir etiqueta de probabilidad
        edge_pos = (pos[u] + pos[v]) / 2
        if u == v:
            edge_pos = pos[u] + np.array([0, 0.3])
        ax.annotate(
            f"{weight:.2f}",
            xy=edge_pos,
            fontsize=9,
            ha="center",
        )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_regime_timeseries(
    df: pd.DataFrame,
    labels: np.ndarray,
    price_col: str = "Close",
    title: str = "Serie Temporal por Régimen",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Visualiza serie temporal coloreada por régimen.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos de precios
    labels : np.ndarray
        Array de etiquetas de régimen
    price_col : str
        Columna con precios
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Alinear labels con df
    df_plot = df.iloc[-len(labels):].copy()
    df_plot["regime"] = labels

    # Dibujar cada régimen con su color
    for regime in sorted(df_plot["regime"].unique()):
        mask = df_plot["regime"] == regime
        ax.scatter(
            df_plot.index[mask],
            df_plot[price_col][mask],
            c=REGIME_COLORS.get(regime, f"C{regime}"),
            label=REGIME_NAMES.get(regime, f"Régimen {regime}"),
            s=10,
            alpha=0.7,
        )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_acf_comparison(
    returns: pd.Series,
    lags: int = 50,
    title: str = "ACF: Retornos vs |Retornos|",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Compara ACF de retornos vs ACF de |retornos| (volatility clustering).

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos
    lags : int
        Número de lags a mostrar
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_acf(returns.dropna(), ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title("ACF de Retornos")
    axes[0].set_xlabel("Lag")

    plot_acf(returns.abs().dropna(), ax=axes[1], lags=lags, alpha=0.05)
    axes[1].set_title("ACF de |Retornos|")
    axes[1].set_xlabel("Lag")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_distribution_comparison(
    returns: pd.Series,
    title: str = "Distribución de Retornos",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Compara distribución empírica vs Normal vs t-Student.

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    returns_clean = returns.dropna()
    mu, sigma = returns_clean.mean(), returns_clean.std()

    # Histograma con ajustes
    axes[0].hist(returns_clean, bins=100, density=True, alpha=0.7, label="Empírica")

    x = np.linspace(returns_clean.min(), returns_clean.max(), 1000)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal")

    # Ajustar t-Student
    df_t, loc_t, scale_t = stats.t.fit(returns_clean)
    axes[0].plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), "g-", lw=2, label=f"t-Student (df={df_t:.1f})")

    axes[0].set_xlabel("Retorno")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Histograma + Ajustes")
    axes[0].legend()
    axes[0].set_xlim(-0.1, 0.1)

    # QQ-plot vs Normal
    stats.probplot(returns_clean, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-Plot vs Normal")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_equity_curves(
    equity_strategy: pd.Series,
    equity_benchmark: pd.Series,
    title: str = "Estrategia vs Buy & Hold",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Compara curvas de equity de estrategia vs benchmark.

    Parameters
    ----------
    equity_strategy : pd.Series
        Equity de la estrategia
    equity_benchmark : pd.Series
        Equity del benchmark (buy & hold)
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(equity_strategy.index, equity_strategy.values,
            label="Estrategia", linewidth=2, color="#2ecc71")
    ax.plot(equity_benchmark.index, equity_benchmark.values,
            label="Buy & Hold", linewidth=2, color="#3498db", alpha=0.7)

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor del Portfolio")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig
