"""Vectorised backtesting of the ML trading signal.

Strategy
--------
Signals are produced on the sub-sampled modelling calendar (every
``sampling_step`` trading days, aligned with the label horizon). At the
close of each rebalance date ``t`` the best model produces ``P(up)`` for
every ticker in the universe. The strategy holds an **equal-weight long
position** in every ticker whose probability exceeds
``probability_threshold`` (cash otherwise) and earns the close-to-close
return from ``t`` to ``t + horizon`` trading days. Because sample dates are
exactly ``horizon`` days apart, the holding windows tile the timeline with
no overlap - a genuine weekly-rebalanced portfolio.

Implementation notes
--------------------
* **Strict signal alignment**: ``signal[t]`` is produced with data up to the
  close of ``t`` and earns the forward ``[t, t+horizon]`` close-to-close
  return computed from the *daily* price panel - there is no same-bar
  look-ahead.
* **Transaction costs**: every change in position (0->1 or 1->0) between two
  consecutive rebalance dates is charged ``transaction_cost_bps`` basis
  points, approximating a low-cost institutional execution.
* **Benchmark**: SPY buy-and-hold measured over the identical rebalance
  windows, so the comparison is like-for-like.
* Performance statistics are computed on per-period returns and annualised
  with ``252 / horizon`` periods per year.
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

# Unified figure style (matches the EDA module so the backtest equity curve
# shares the same visual language as every other chart in the report)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

logger = logging.getLogger(__name__)


def _performance_metrics(
    period_returns: pd.Series, rf_annual: float, periods_per_year: float,
    label: str,
) -> Dict[str, Any]:
    """Compute performance statistics from per-period returns.

    Parameters
    ----------
    period_returns:
        Strategy return per rebalance period (fraction, not percent).
    rf_annual:
        Annual risk-free rate used for the Sharpe ratio.
    periods_per_year:
        Number of rebalance periods per year (e.g. 252/5 ~ 50.4).
    label:
        Strategy label for logging.

    Returns
    -------
    dict
        Total return, CAGR, annualised vol, Sharpe, max drawdown, hit rate.
    """
    rets = period_returns.dropna()
    n_periods = len(rets)
    n_years = max(n_periods / periods_per_year, 1e-9)

    equity = (1.0 + rets).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    ann_vol = float(rets.std() * np.sqrt(periods_per_year))
    ann_ret = float(rets.mean() * periods_per_year)
    sharpe = float((ann_ret - rf_annual) / (ann_vol if ann_vol > 0 else np.nan))

    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    hit_rate = float((rets > 0).mean())

    stats = {
        "label": label,
        "total_return_pct": round(100.0 * total_return, 2),
        "cagr_pct": round(100.0 * cagr, 2),
        "ann_vol_pct": round(100.0 * ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
        "hit_rate_pct": round(100.0 * hit_rate, 2),
        "n_periods": int(n_periods),
    }
    logger.info("Backtest [%s]: %s", label, stats)
    return stats


def run_backtest(
    test_frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    daily_prices: pd.DataFrame,
    cfg,
    figures_dir: pathlib.Path,
) -> Dict[str, Any]:
    """Backtest the probability-threshold strategy vs the SPY benchmark.

    Parameters
    ----------
    test_frame:
        Hold-out test rows enriched with ``proba_up`` from the best model
        (contains ``ticker, date, close`` on the rebalance calendar).
    benchmark:
        Long-format OHLCV frame for SPY covering at least the test window.
    daily_prices:
        Cleaned daily OHLCV panel of the universe (used to compute exact
        forward close-to-close returns over the holding horizon).
    cfg:
        Pipeline config namespace (``cfg.backtest`` and
        ``cfg.features.target_horizon`` are used).
    figures_dir:
        Output directory for the equity-curve figure.

    Returns
    -------
    dict
        Strategy and benchmark stats plus figure paths and a period table.
    """
    b = cfg.backtest
    threshold = float(b.probability_threshold)
    cost = float(b.transaction_cost_bps) / 10_000.0
    rf = float(b.risk_free_rate)
    horizon = int(cfg.features.target_horizon)
    periods_per_year = 252.0 / horizon

    # -- Daily wide close matrix (rectangular panel after cleaning) -----------
    wide_close = daily_prices.pivot(index="date", columns="ticker", values="close")
    wide_close = wide_close.sort_index()
    date_pos = {d: i for i, d in enumerate(wide_close.index)}

    # Forward horizon-day return for every daily date, from the daily panel
    fwd_ret_daily = wide_close.shift(-horizon) / wide_close - 1.0

    # -- Signals on the rebalance calendar ------------------------------------
    sig = test_frame[["date", "ticker", "proba_up"]].copy()
    sig["position"] = (sig["proba_up"] >= threshold).astype(float)
    rebalance_dates = np.sort(sig["date"].unique())

    # Lookup forward returns at each rebalance date (per ticker)
    fwd_at_rebalance = fwd_ret_daily.reindex(rebalance_dates)
    positions = sig.pivot(index="date", columns="ticker", values="position")
    positions = positions.reindex(index=rebalance_dates).fillna(0.0)

    # Last rebalance date has no forward window -> drop it from returns
    valid_dates = [d for d in rebalance_dates if d in fwd_ret_daily.index]
    if valid_dates:
        last_valid = valid_dates[-1]
        has_forward = fwd_at_rebalance.notna().any(axis=1)
    else:
        has_forward = pd.Series(dtype=bool)

    gross_ret = (fwd_at_rebalance * positions).sum(axis=1) / positions.sum(
        axis=1
    ).replace(0, np.nan)
    # Dates where nothing is selected -> stay in cash (0% return)
    gross_ret = gross_ret.fillna(0.0)

    # -- Transaction costs: turnover between consecutive rebalance dates ------
    turnover = positions.diff().abs().sum(axis=1) / len(positions.columns)
    turnover.iloc[0] = positions.iloc[0].abs().sum() / len(positions.columns)
    net_ret = gross_ret - turnover * cost

    # Drop trailing date(s) without a complete forward window
    period_table = pd.DataFrame({
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "n_long": positions.sum(axis=1).astype(int),
        "n_assets": len(positions.columns),
        "turnover": turnover,
    })
    period_table["exposure_pct"] = 100.0 * period_table["n_long"] / period_table["n_assets"]
    if has_forward.any():
        last_complete = has_forward[has_forward].index[-1]
        period_table = period_table.loc[:last_complete]

    strategy_returns = period_table["net_ret"]
    strategy_equity = (1.0 + strategy_returns).cumprod()
    strategy_equity = pd.concat(
        [pd.Series([1.0], index=[strategy_returns.index[0] - pd.Timedelta(days=1)]),
         strategy_equity]
    )

    # -- Benchmark: SPY over the identical rebalance windows ------------------
    bench_close = benchmark.set_index("date")["close"].sort_index()
    bench_ret = []
    for d in strategy_returns.index:
        i = date_pos.get(d)
        if i is None or i + horizon >= len(wide_close.index):
            bench_ret.append(np.nan)
            continue
        # forward SPY close via the daily calendar (intraday alignment is fine)
        fwd_date = wide_close.index[i + horizon]
        bench_ret.append(bench_close.asof(fwd_date) / bench_close.asof(d) - 1.0)
    bench_returns = pd.Series(bench_ret, index=strategy_returns.index).dropna()
    bench_equity = (1.0 + bench_returns).cumprod()
    bench_equity = pd.concat(
        [pd.Series([1.0], index=[bench_returns.index[0] - pd.Timedelta(days=1)]),
         bench_equity]
    )

    stats_strategy = _performance_metrics(
        strategy_returns, rf, periods_per_year, "ML strategy"
    )
    stats_benchmark = _performance_metrics(
        bench_returns, rf, periods_per_year, "SPY buy & hold"
    )
    stats_strategy["avg_exposure_pct"] = round(
        float(period_table["exposure_pct"].mean()), 1
    )
    stats_strategy["avg_turnover_pct_per_period"] = round(
        100.0 * float(period_table["turnover"].mean()), 2
    )

    # -- Equity curve figure ---------------------------------------------------
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(
        strategy_equity.index, strategy_equity.values,
        linewidth=2.0, color="#1f77b4",
        label="ML signal strategy (net of costs)",
    )
    ax.plot(
        bench_equity.index, bench_equity.values,
        linewidth=2.0, color="#7f7f7f", linestyle="--", label="SPY buy & hold",
    )
    ax.set_title(
        f"Out-of-sample Backtest ({horizon}-day rebalancing) - "
        f"long when P(up) >= {threshold:.2f}, "
        f"{int(b.transaction_cost_bps)} bps per turnover unit"
    )
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Portfolio value (start = 1.0)")
    ax.legend(frameon=False, loc="upper left")
    path = figures_dir / "11_backtest_equity.png"
    fig.savefig(path)
    plt.close(fig)

    return {
        "strategy": stats_strategy,
        "benchmark": stats_benchmark,
        "equity_curve_fig": str(path),
        "period_table": period_table,
    }
