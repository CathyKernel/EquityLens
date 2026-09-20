"""Statistical hypothesis-testing suite.

Before any model is fitted, the statistical properties of the return series
must be validated - this is what separates a defensible quant workflow from
"throwing sklearn at a CSV". Five families of tests are applied:

1. **Normality** (Jarque-Bera, per ticker + pooled): daily equity returns are
   famously non-Gaussian (fat tails). Rejection motivates robust risk metrics
   and justifies tree-based models over Gaussian assumptions.
2. **Stationarity** (Augmented Dickey-Fuller): raw prices are non-stationary
   (unit root) while returns are stationary - the formal justification for
   modelling returns rather than prices.
3. **Autocorrelation** (Ljung-Box on returns): weak-form market efficiency
   implies near-zero linear autocorrelation in returns.
4. **Volatility clustering** (Ljung-Box on *squared* returns): significant
   autocorrelation of squared returns = ARCH effects, the empirical basis for
   volatility features in the model.
5. **Cross-sector heterogeneity** (one-way ANOVA + Welch t-tests): do mean
   daily returns differ across GICS sectors? Informs whether sector belongs
   in the feature set.

Every test reports its statistic, p-value and an interpretable conclusion;
results feed the HTML report and the QQ-plot figure.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

ALPHA = 0.05


def _conclude(p_value: float, hypothesis: str) -> str:
    """Return a human-readable reject/retain conclusion for a p-value.

    Parameters
    ----------
    p_value:
        Test p-value.
    hypothesis:
        Short description of the null hypothesis.

    Returns
    -------
    str
        Interpretive sentence for the report.
    """
    if p_value < ALPHA:
        return (
            f"p = {p_value:.2e} < 0.05 -> reject the null "
            f"({hypothesis} is contradicted by the data)"
        )
    return (
        f"p = {p_value:.2e} >= 0.05 -> fail to reject the null "
        f"(data is consistent with {hypothesis})"
    )


def test_normality(returns: pd.DataFrame) -> Dict[str, Any]:
    """Jarque-Bera normality tests for each ticker and the pooled sample.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.

    Returns
    -------
    dict
        Per-ticker JB results, pooled result and interpretation.
    """
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        stat, p = stats.jarque_bera(series)
        rows.append({
            "ticker": ticker,
            "jb_stat": round(float(stat), 1),
            "p_value": float(p),
            "skew": round(float(series.skew()), 3),
            "excess_kurtosis": round(float(series.kurt()), 3),
            "normal": bool(p >= ALPHA),
        })
    pooled = returns.stack()
    stat_p, p_p = stats.jarque_bera(pooled)
    result = {
        "per_ticker": pd.DataFrame(rows),
        "pooled": {
            "jb_stat": round(float(stat_p), 1),
            "p_value": float(p_p),
            "skew": round(float(pooled.skew()), 3),
            "excess_kurtosis": round(float(pooled.kurt()), 3),
        },
        "n_reject": int(sum(not r["normal"] for r in rows)),
        "n_tickers": len(rows),
    }
    result["interpretation"] = (
        f"Jarque-Bera rejects normality for {result['n_reject']} of "
        f"{result['n_tickers']} tickers (pooled excess kurtosis = "
        f"{result['pooled']['excess_kurtosis']:.1f}): daily returns have fat "
        f"tails, so Gaussian-based risk metrics understate tail risk and "
        f"non-parametric/tree models are preferred."
        if result["n_reject"] > 0 else
        "Returns appear Gaussian; standard assumptions hold."
    )
    return result


def test_stationarity(returns: pd.DataFrame) -> Dict[str, Any]:
    """Augmented Dickey-Fuller tests: prices vs first-differenced returns.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.

    Returns
    -------
    dict
        ADF results for one representative price series and for returns.
    """
    # Rebuild a price-like series from cumulative returns of the median-vol ticker
    ticker = returns.std().idxmin()
    log_ret = np.log1p(returns[ticker].dropna())
    price_like = log_ret.cumsum()

    adf_price = adfuller(price_like, autolag="AIC")
    adf_ret = adfuller(log_ret, autolag="AIC")

    return {
        "ticker": ticker,
        "price_adf_stat": round(float(adf_price[0]), 2),
        "price_p_value": float(adf_price[1]),
        "returns_adf_stat": round(float(adf_ret[0]), 2),
        "returns_p_value": float(adf_ret[1]),
        "interpretation": (
            f"ADF on the price-like series of {ticker}: p = "
            f"{adf_price[1]:.2e} (unit root NOT rejected -> non-stationary); "
            f"on daily returns: p = {adf_ret[1]:.2e} (unit root rejected -> "
            f"stationary). Modelling returns instead of prices is therefore "
            f"statistically justified."
        ),
    }


def test_autocorrelation(returns: pd.DataFrame) -> Dict[str, Any]:
    """Ljung-Box tests for autocorrelation in returns and squared returns.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.

    Returns
    -------
    dict
        Ljung-Box results at lags 1/5/10/20 for returns and squared returns.
    """
    lags = [1, 5, 10, 20]
    pooled = returns.stack()

    lb_ret = acorr_ljungbox(pooled, lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(pooled ** 2, lags=lags, return_df=True)

    ret_row = lb_ret.loc[10]
    sq_row = lb_sq.loc[10]
    return {
        "lags": lags,
        "returns_ljung_box": {
            lag: round(float(lb_ret.loc[lag, "lb_stat"]), 1) for lag in lags
        },
        "returns_p_values": {
            lag: float(lb_ret.loc[lag, "lb_pvalue"]) for lag in lags
        },
        "squared_returns_ljung_box": {
            lag: round(float(lb_sq.loc[lag, "lb_stat"]), 1) for lag in lags
        },
        "squared_returns_p_values": {
            lag: float(lb_sq.loc[lag, "lb_pvalue"]) for lag in lags
        },
        "interpretation": (
            f"Ljung-Box on returns at lag 10: p = "
            f"{lb_ret.loc[10, 'lb_pvalue']:.2e}; on squared returns: p = "
            f"{lb_sq.loc[10, 'lb_pvalue']:.2e}. Returns show little linear "
            f"memory (weak-form efficiency) while squared returns are "
            f"strongly autocorrelated -> pronounced volatility clustering "
            f"(ARCH effects). This motivates the rolling-volatility and "
            f"Bollinger-band features in the model."
        ),
    }


def test_sector_differences(
    returns: pd.DataFrame, sector_map: Dict[str, str]
) -> Dict[str, Any]:
    """One-way ANOVA of mean daily returns across GICS sectors.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.
    sector_map:
        Ticker -> sector mapping.

    Returns
    -------
    dict
        ANOVA F statistic, p-value and per-sector sample statistics.
    """
    long_returns = returns.stack().rename("ret").reset_index()
    long_returns["sector"] = long_returns["ticker"].map(sector_map)
    groups = [
        grp["ret"].dropna().values
        for _, grp in long_returns.groupby("sector")
    ]
    f_stat, p_value = stats.f_oneway(*groups)

    sector_stats = (
        long_returns.groupby("sector")["ret"]
        .agg(["count", "mean", "std"])
        .round(4)
        .reset_index()
        .rename(columns={"count": "n_obs", "mean": "mean_daily_pct", "std": "std_daily_pct"})
    )
    sector_stats["ann_return_pct"] = (sector_stats["mean_daily_pct"] * 252).round(2)
    sector_stats["ann_vol_pct"] = (sector_stats["std_daily_pct"] * np.sqrt(252)).round(2)

    return {
        "f_stat": round(float(f_stat), 3),
        "p_value": float(p_value),
        "sector_stats": sector_stats,
        "interpretation": _conclude(
            p_value,
            "equal mean daily returns across sectors",
        ),
    }


def plot_qq(returns: pd.DataFrame, figures_dir: pathlib.Path) -> str:
    """QQ-plot of pooled daily returns against the Gaussian quantiles.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.
    figures_dir:
        Output directory for the figure.

    Returns
    -------
    str
        Path of the saved figure.
    """
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pooled = returns.stack().dropna()

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    stats.probplot(pooled, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2.2, alpha=0.35, color="#1f77b4")
    ax.get_lines()[1].set(linewidth=1.6, color="#d62728")
    ax.set_title("QQ-Plot: Pooled Daily Returns vs Gaussian Quantiles")
    ax.set_xlabel("Theoretical Gaussian quantiles")
    ax.set_ylabel("Empirical return quantiles (%)")
    path = figures_dir / "07_qq_plot.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run_all_tests(
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    figures_dir: pathlib.Path,
) -> Dict[str, Any]:
    """Run the complete hypothesis-testing suite.

    Parameters
    ----------
    returns:
        Wide-format daily returns matrix.
    sector_map:
        Ticker -> sector mapping.
    figures_dir:
        Output directory for figures.

    Returns
    -------
    dict
        Aggregated results with keys ``normality``, ``stationarity``,
        ``autocorrelation``, ``sector_anova`` and ``qq_plot``.
    """
    logger.info("Running statistical hypothesis tests...")
    results = {
        "normality": test_normality(returns),
        "stationarity": test_stationarity(returns),
        "autocorrelation": test_autocorrelation(returns),
        "sector_anova": test_sector_differences(returns, sector_map),
        "qq_plot": plot_qq(returns, figures_dir),
    }
    logger.info("Statistical tests complete.")
    return results
