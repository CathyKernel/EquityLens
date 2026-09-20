"""Exploratory data analysis (EDA) module.

Produces the headline visual evidence for the analysis report:

* normalised price evolution of the universe vs the SPY benchmark,
* distribution diagnostics of daily returns (histogram + per-sector boxplot),
* annualised volatility by GICS sector (mean bar + per-ticker range),
* average cross-sector correlation heatmap of daily returns,
* rolling 60-day volatility of the SPY benchmark (volatility clustering),
* sector-level risk/return scatter.

With a 60+ stock universe, per-ticker charts become unreadable at report
size, so the headline figures aggregate to the sector level while the
per-ticker volatility league is still exported as structured data
(``vol_league_top`` / ``vol_league_bottom``) for tabular presentation.

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
from matplotlib.lines import Line2D
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
    "font.size": 11,
    "axes.titlesize": 12.5,
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
        ax.plot(
            normalised.index, normalised[ticker],
            linewidth=0.7, alpha=0.30, color="grey",
        )
    # Equally-weighted universe index for reference
    ew_index = normalised.mean(axis=1)
    ax.plot(ew_index.index, ew_index, color="black", linewidth=2.2, label="Equal-weight universe")
    # Log scale: NVDA's ~280x run would otherwise flatten every other series
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000, 5000, 10000, 25000])
    ax.set_yticklabels(["100", "200", "500", "1k", "2k", "5k", "10k", "25k"])
    ax.set_title(
        "Normalised Price Evolution (base 100, log scale; "
        f"grey lines = {normalised.shape[1]} individual stocks)"
    )
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

    # -- Figure 3: annualised volatility by sector ----------------------------
    # At 66 tickers a per-ticker bar chart is unreadable at report size, so
    # the headline figure aggregates to the sector level (mean bar + min/max
    # whiskers across that sector's tickers); the per-ticker league is
    # exported below as structured data for the report's league table.
    ann_vol = (returns.std() * np.sqrt(252) * 100).sort_values()
    ticker_sector = pd.Series(sector_map).reindex(ann_vol.index)
    sector_vol = (
        ann_vol.groupby(ticker_sector)
        .agg(mean_vol="mean", min_vol="min", max_vol="max")
        .sort_values("mean_vol")
    )
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    y_pos = np.arange(len(sector_vol))
    ax.barh(
        y_pos, sector_vol["mean_vol"],
        xerr=[
            sector_vol["mean_vol"] - sector_vol["min_vol"],
            sector_vol["max_vol"] - sector_vol["mean_vol"],
        ],
        color=PALETTE, alpha=0.85, height=0.62,
        error_kw={"ecolor": BENCH, "elinewidth": 1.2, "capsize": 3},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_vol.index)
    for i, row in enumerate(sector_vol.itertuples(index=False)):
        ax.text(
            row.max_vol + 1.5, i, f"{row.mean_vol:.1f}%",
            va="center", fontsize=9, color="#333333",
        )
    ax.axvline(
        float(ann_vol.median()), color=ACCENT, linestyle="--", linewidth=1.2,
        label=f"Universe median ticker: {ann_vol.median():.1f}%",
    )
    ax.set_title(
        "Annualised Volatility by GICS Sector "
        "(bar = sector mean, whiskers = min/max ticker)"
    )
    ax.set_xlabel("Annualised return volatility (%)")
    ax.legend(frameon=False, loc="lower right")
    path = figures_dir / "03_volatility_league.png"
    fig.savefig(path)
    plt.close(fig)
    figures["volatility_league"] = str(path)

    # -- Figure 4: average cross-sector correlation heatmap -------------------
    corr = returns.corr()
    sec = pd.Series(sector_map).reindex(corr.columns)
    sectors = sorted(sec.dropna().unique())
    sec_corr = pd.DataFrame(index=sectors, columns=sectors, dtype=float)
    for a in sectors:
        for b in sectors:
            block = corr.loc[sec == a, sec == b].values
            if a == b:
                # within-sector average excludes the diagonal of ones
                block = block[~np.eye(block.shape[0], dtype=bool)]
            sec_corr.loc[a, b] = float(np.nanmean(block))
    fig, ax = plt.subplots(figsize=(9, 7.6), constrained_layout=True)
    sns.heatmap(
        sec_corr.astype(float), ax=ax, cmap="vlag", center=0, vmin=0, vmax=1,
        square=True, linewidths=0.6, annot=True, fmt=".2f",
        annot_kws={"size": 8.5},
        cbar_kws={"shrink": 0.8, "label": "Average pairwise correlation"},
    )
    ax.set_title(
        "Average Correlation of Daily Returns within and across GICS Sectors"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
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
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    ax.scatter(
        sector_agg["ann_vol"], sector_agg["ann_return"],
        s=140, color=PALETTE, alpha=0.8, edgecolor="white", zorder=3,
    )
    # Number the points and map them in an outside legend: sector names are
    # too long to place at the points without collisions. Points near the top
    # get their number placed *below* so it cannot clip at the axes boundary.
    ax.margins(y=0.10)
    y_lo = float(sector_agg["ann_return"].min())
    y_hi = float(sector_agg["ann_return"].max())
    for i, row in enumerate(sector_agg.itertuples(index=False)):
        near_top = row.ann_return > y_lo + 0.88 * (y_hi - y_lo)
        ax.annotate(
            str(i + 1), (row.ann_vol, row.ann_return),
            textcoords="offset points", xytext=(6, -11 if near_top else 4),
            fontsize=8.5, fontweight="bold", color="#333333",
        )
    handles = [
        Line2D(
            [0], [0], linestyle="none", marker="o",
            markerfacecolor=PALETTE, markeredgecolor="white", markersize=7,
            label=f"{i + 1}  {row.sector}",
        )
        for i, row in enumerate(sector_agg.itertuples(index=False))
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8.5, handletextpad=0.4,
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
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    sns.boxplot(
        data=long_returns, x="sector", y="ret", order=order,
        showfliers=False, ax=ax, width=0.6,
    )
    # Wrap long sector names onto two lines and rotate vertically: with 11
    # categories, upright or diagonal labels collide; vertical never does.
    ax.set_xticklabels(
        [s.replace(" ", "\n") if len(s) > 10 else s for s in order],
        rotation=90, ha="center", fontsize=8,
    )
    ax.set_title("Distribution of Daily Returns by Sector (outliers hidden)")
    ax.set_xlabel("GICS sector")
    ax.set_ylabel("Daily return (%)")
    path = figures_dir / "06_sector_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    figures["sector_boxplot"] = str(path)

    # -- Headline stats --------------------------------------------------------
    league = ann_vol.sort_values(ascending=False)
    vol_league_top = [
        {
            "ticker": str(t),
            "sector": str(ticker_sector[t]),
            "ann_vol_pct": round(float(v), 1),
        }
        for t, v in league.head(10).items()
    ]
    vol_league_bottom = [
        {
            "ticker": str(t),
            "sector": str(ticker_sector[t]),
            "ann_vol_pct": round(float(v), 1),
        }
        for t, v in league.tail(5).sort_values().items()
    ]
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
        "vol_league_top": vol_league_top,
        "vol_league_bottom": vol_league_bottom,
    }
    logger.info("EDA complete: %d figures written to %s", len(figures) - 1, figures_dir)
    return figures
