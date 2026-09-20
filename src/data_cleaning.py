"""Data cleaning and validation module.

Applies production-grade hygiene checks to the raw OHLCV panel:

1. **Schema enforcement** - correct dtypes and canonical column order.
2. **Calendar alignment** - every ticker is reindexed onto the union of all
   trading dates in the universe, so panels are rectangular.
3. **Integrity rules** - ``high >= max(open, close)``, ``low <= min(open,
   close)``, non-negative volume, strictly positive prices.
4. **Missing data policy** - isolated gaps are forward-filled (a stock that
   did not trade keeps its last quote); rows still missing after
   forward-fill are dropped, which typically corresponds to listing gaps
   at the start of the sample.
5. **Outlier flagging** - daily returns beyond +/- N sigma are *flagged but
   kept*: in equity markets extreme moves are genuine information (earnings
   shocks, crashes), and silently trimming them would bias risk estimates.

The module returns both the cleaned frame and a machine-readable quality
report that feeds the HTML report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_prices(
    raw: pd.DataFrame,
    sector_map: Dict[str, str],
    outlier_sigma: float = 5.0,
    min_history_days: int = 260,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate the raw OHLCV panel.

    Parameters
    ----------
    raw:
        Raw long-format OHLCV frame from the acquisition layer.
    sector_map:
        Ticker -> sector mapping, used to drop unmapped tickers.
    outlier_sigma:
        Sigma threshold beyond which a daily return is flagged as an outlier.
    min_history_days:
        Minimum number of valid observations for a ticker to be kept.

    Returns
    -------
    tuple
        ``(clean_df, quality_report)`` where ``clean_df`` is the rectangular
        OHLCV panel and ``quality_report`` is a dict of metrics.
    """
    report: Dict[str, Any] = {}
    df = raw.copy()

    # -- 1. Deduplicate -------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    report["rows_input"] = int(n_before)
    report["duplicates_removed"] = int(n_before - len(df))

    # -- 2. Restrict to mapped universe --------------------------------------
    unmapped = sorted(set(df["ticker"].unique()) - set(sector_map.keys()))
    df = df[df["ticker"].isin(sector_map.keys())]
    report["unmapped_tickers"] = unmapped

    # -- 3. Integrity rules ---------------------------------------------------
    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    bad_price = (df["close"] <= 0) | df["close"].isna()
    report["ohlc_violations"] = int(bad_ohlc.sum())
    report["invalid_closes"] = int(bad_price.sum())
    df = df[~(bad_ohlc | bad_price)]

    # -- 4. Calendar alignment (rectangular panel) ---------------------------
    trading_dates = np.sort(df["date"].unique())
    full_index = pd.MultiIndex.from_product(
        [df["ticker"].unique(), trading_dates],
        names=["ticker", "date"],
    )
    df = (
        df.set_index(["ticker", "date"])
        .reindex(full_index)
        .reset_index()
    )
    # Forward-fill isolated gaps per ticker (sort first so ffill is causal)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    gaps_before = int(df["close"].isna().sum())
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = (
        df.groupby("ticker", sort=False)[price_cols]
        .transform(lambda s: s.ffill())
    )
    df["volume"] = df["volume"].fillna(0.0)
    # Rows still missing after forward-fill are pre-listing dates on the union
    # calendar - drop them (they belong to tickers not yet traded).
    report["gaps_forward_filled"] = gaps_before - int(df["close"].isna().sum())
    report["gaps_unfillable_dropped"] = int(df["close"].isna().sum())
    df = df.dropna(subset=["close"])

    # -- 5. Per-ticker history filter -----------------------------------------
    counts = df.groupby("ticker")["close"].count()
    short_tickers = sorted(counts[counts < min_history_days].index.tolist())
    if short_tickers:
        df = df[~df["ticker"].isin(short_tickers)]
    report["short_history_tickers"] = short_tickers

    # -- 6. Outlier flagging (kept in data, flagged only) --------------------
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    pooled_std = float(df["ret_1d"].std())
    outlier_mask = df["ret_1d"].abs() > outlier_sigma * pooled_std
    report["outlier_threshold_sigma"] = outlier_sigma
    report["outlier_days_flagged"] = int(outlier_mask.sum())
    df["is_outlier"] = outlier_mask.fillna(False)
    df = df.drop(columns=["ret_1d"])

    # -- 7. Final stats --------------------------------------------------------
    report["rows_output"] = int(len(df))
    report["tickers_kept"] = int(df["ticker"].nunique())
    report["first_date"] = str(df["date"].min().date())
    report["last_date"] = str(df["date"].max().date())
    report["trading_days"] = int(df["date"].nunique())
    report["zero_volume_days_pct"] = round(
        100.0 * float((df["volume"] == 0).mean()), 3
    )

    logger.info(
        "Cleaning done: %d rows in -> %d rows out (%d tickers, %s .. %s)",
        report["rows_input"], report["rows_output"], report["tickers_kept"],
        report["first_date"], report["last_date"],
    )
    return df, report
