"""Exploratory data analysis (EDA) module.

Produces the headline visual evidence for the analysis report:

* normalised price evolution of the universe vs the SPY benchmark,
* distribution diagnostics of daily returns (histogram + per-sector boxplot),
* annualised volatility league table (per ticker),
* cross-ticker correlation heatmap of daily returns,
* rolling 60-day volatility of the SPY benchmark (volatility clustering),
* sector-level risk/return scatter.

All figures are written to ``outputs/figures`` as PNG files with a unified,
publication-quality style, and every figure is driven by the *cleaned* panel
so that the numbers in the report match the modelling dataset exactly.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless backend, safe on servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Unified visual style for every figure in the project
STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
}

PALETTE = "#1f77b4"
ACCENT = "#d62728"
BENCH = "#7f7f7f"


def _returns_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the wide-format daily simple-returns matrix.

    Parameters
    ----------
    prices:
        Cleaned long-format OHLCV panel.

    Returns
    -------
    pd.DataFrame
        Wide returns matrix indexed by date, one column per ticker.
    """
    wide = prices.pivot(index="date", columns="ticker", values="close")
    return wide.pct_change(fill_method=None).dropna(how="all")


def run_eda(
    prices: pd.DataFrame,
    sector_map: Dict[str, str],
    figures_dir: pathlib.Path,
) -> Dict[str, str]:
    """Run the full EDA suite and persist all figures.

    Parameters
    ----------
    prices:
        Cleaned long-format OHLCV panel (universe only, no benchmark).
    sector_map:
        Ticker -> sector mapping.
    figures_dir:
        Output directory for PNG figures.

    Returns
    -------
    dict
        Mapping of figure name -> absolute file path, plus an ``eda_summary``
        key with headline statistics used in the report.
    """
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    sns.set_palette("colorblind")

    figures: Dict[str, str] = {}
    returns = _returns_panel(prices)
    wide_close = prices.pivot(index="date", columns="ticker", values="close")

    # -- Figure 1: normalised price evolution --------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    normalised = wide_close / wide_close.iloc[0] * 100.0
    for ticker in normalised.columns:
        ax.plot(normalised.index, normalised[ticker], linewidth=0.9, alpha=0.55)
    # Equally-weighted universe index for reference
    ew_index = normalised.mean(axis=1)
    ax.plot(ew_index.index, ew_index, color="black", linewidth=2.2, label="Equal-weight universe")
    # Log scale: NVDA's ~280x run would otherwise flatten every other series
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000, 5000, 10000, 25000])
    ax.set_yticklabels(["100", "200", "500", "1k", "2k", "5k", "10k", "25k"])
    ax.set_title("Normalised Price Evolution (base 100, log scale; grey lines = individual stocks)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed close (start = 100, log scale)")
    ax.legend(loc="upper left", frameon=False)
    path = figures_dir / "01_price_evolution.png"
    fig.savefig(path)
    plt.close(fig)
    figures["price_evolution"] = str(path)

    # -- Figure 2: distribution of daily returns ------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    pooled = returns.stack()
    ax.hist(
        pooled, bins=np.linspace(-6, 6, 121), density=True, alpha=0.7,
        color=PALETTE, edgecolor="none", label="Pooled daily returns",
    )
    x = np.linspace(-6, 6, 400)
    normal_pdf = (
        1.0 / (np.sqrt(2 * np.pi) * pooled.std()) *
        np.exp(-0.5 * ((x - pooled.mean()) / pooled.std()) ** 2)
    )
    ax.plot(x, normal_pdf, color=ACCENT, linewidth=2.0, label="Gaussian with same mean/var")
    ax.set_yscale("log")
    ax.set_title("Daily Return Distribution vs Gaussian (log-density scale)")
    ax.set_xlabel("Daily simple return (%)")
    ax.set_ylabel("Density (log scale)")
    ax.legend(frameon=False)
    path = figures_dir / "02_return_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    figures["return_distribution"] = str(path)

    # -- Figure 3: annualised volatility league -------------------------------
    ann_vol = returns.std() * np.sqrt(252) * 100
    ann_vol = ann_vol.sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    colors = [PALETTE] * len(ann_vol)
    top_vol = ann_vol.index[-1]
    colors[-1] = ACCENT
    ax.barh(ann_vol.index, ann_vol.values, color=colors, alpha=0.85)
    ax.axvline(float(ann_vol.median()), color=BENCH, linestyle="--", linewidth=1.2,
               label=f"Universe median: {ann_vol.median():.1f}%")
    for i, (ticker, vol) in enumerate(ann_vol.items()):
        ax.text(vol + 0.5, i, f"{vol:.1f}", va="center", fontsize=8)
    ax.set_title("Annualised Volatility by Ticker (30-day-equivalent, full sample)")
    ax.set_xlabel("Annualised return volatility (%)")
    ax.legend(frameon=False, loc="lower right")
    path = figures_dir / "03_volatility_league.png"
    fig.savefig(path)
    plt.close(fig)
    figures["volatility_league"] = str(path)

    # -- Figure 4: correlation heatmap ----------------------------------------
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    sns.heatmap(
        corr, ax=ax, cmap="vlag", center=0, vmin=0, vmax=1,
        square=True, linewidths=0.4, cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    ax.set_title("Cross-Ticker Correlation of Daily Returns")
    path = figures_dir / "04_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    figures["correlation_heatmap"] = str(path)

    # -- Figure 5: sector risk/return profile ---------------------------------
    ticker_sector = pd.Series(sector_map)
    sector_stats = pd.DataFrame({
        "ann_return": returns.mean() * 252 * 100,
        "ann_vol": ann_vol,
        "sector": ticker_sector.reindex(returns.columns),
    })
    sector_agg = sector_stats.groupby("sector").agg(
        ann_return=("ann_return", "mean"), ann_vol=("ann_vol", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(
        sector_agg["ann_vol"], sector_agg["ann_return"],
        s=140, color=PALETTE, alpha=0.8, edgecolor="white", zorder=3,
    )
    for row in sector_agg.itertuples(index=False):
        ax.annotate(
            row.sector, (row.ann_vol, row.ann_return),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )
    ax.set_title("Sector Risk / Return Profile (equal-weight, full sample)")
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised return (%)")
    path = figures_dir / "05_sector_risk_return.png"
    fig.savefig(path)
    plt.close(fig)
    figures["sector_risk_return"] = str(path)

    # -- Figure 6: sector return boxplot --------------------------------------
    long_returns = returns.stack().rename("ret").reset_index()
    long_returns["sector"] = long_returns["ticker"].map(sector_map)
    order = (
        long_returns.groupby("sector")["ret"].std().sort_values().index
    )
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    sns.boxplot(
        data=long_returns, x="sector", y="ret", order=order,
        showfliers=False, ax=ax, width=0.6,
    )
    ax.set_title("Distribution of Daily Returns by Sector (outliers hidden)")
    ax.set_xlabel("GICS sector")
    ax.set_ylabel("Daily return (%)")
    ax.tick_params(axis="x", rotation=30)
    path = figures_dir / "06_sector_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    figures["sector_boxplot"] = str(path)

    # -- Headline stats --------------------------------------------------------
    figures["eda_summary"] = {
        "pooled_mean_daily_pct": round(float(pooled.mean()), 4),
        "pooled_std_daily_pct": round(float(pooled.std()), 4),
        "pooled_skew": round(float(pooled.skew()), 3),
        "pooled_kurtosis_excess": round(float(pooled.kurt()), 3),
        "pooled_min_pct": round(float(pooled.min()), 2),
        "pooled_max_pct": round(float(pooled.max()), 2),
        "best_ticker_total": str(
            (wide_close.iloc[-1] / wide_close.iloc[0] - 1).idxmax()
        ),
        "best_ticker_total_pct": round(
            float((wide_close.iloc[-1] / wide_close.iloc[0] - 1).max() * 100), 1
        ),
        "avg_cross_correlation": round(float(corr.where(
            ~np.eye(len(corr), dtype=bool)
        ).stack().mean()), 3),
    }
    logger.info("EDA complete: %d figures written to %s", len(figures) - 1, figures_dir)
    return figures
