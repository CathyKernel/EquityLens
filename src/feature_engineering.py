"""Feature engineering: technical indicators and the modelling target.

Builds a tidy panel of model-ready features from the cleaned OHLCV data.

Label design (why weekly?)
--------------------------
Next-day direction is dominated by microstructure noise: empirically the
best cross-sectional models barely beat a coin flip at that horizon, and
cross-validated AUC hovers around 0.50. This project therefore targets the
**direction of the next `target_horizon` trading days (one trading week)**,
where momentum and volatility signals carry genuine information. To keep
the labels statistically honest, only every `sampling_step`-th trading date
enters the dataset, so consecutive labels never overlap - otherwise
CV scores would be inflated by label correlation.

Leakage discipline (the single most important design constraint)
-----------------------------------------------------------------
* Every feature at row ``(ticker, t)`` uses **only information available at
  the close of day t**: lagged returns, rolling windows ending at t, and
  cross-sectional ranks computed over the universe **as of t**.
* The target is ``sign(close_{t+h} / close_t - 1)`` where ``h`` is the
  target horizon - strictly *future* information, never a feature.
* Rolling statistics are computed inside ``groupby(ticker)`` so windows
  never bleed across tickers.
* The first ``warmup_days`` rows of each ticker are discarded (indicator
  burn-in), and rows without an available forward close are dropped.

Feature families
----------------
- lagged multi-horizon returns (1/5/10/21 trading days)
- trend: SMA/EMA price ratios, SMA(10)-SMA(50) spread
- momentum: RSI(14), MACD histogram (normalised), 52-week-high distance
- volatility: 10d & 30d annualised realised vol, Bollinger %B and bandwidth
- microstructure: daily range, close location value, volume z-score
- cross-sectional: per-date volatility rank (market-regime proxy)
- target: next-week direction (1 = up, 0 = down)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "target_up"

FEATURE_COLUMNS = [
    "ret_lag_1", "ret_lag_5", "ret_lag_10", "ret_lag_21",
    "close_to_sma10", "close_to_sma20", "close_to_sma50", "sma_spread_10_50",
    "ema_ratio_12", "rsi_14", "macd_hist_norm", "dist_52w_high",
    "vol_10d_ann", "vol_30d_ann", "bollinger_pct_b", "bollinger_bandwidth",
    "daily_range", "close_location", "volume_zscore_20", "vol_cs_rank",
]

CATEGORICAL_ENCODINGS = {
    "sector": "sector",  # one-hot, see build_dataset()
}


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style EMA of gains/losses).

    Parameters
    ----------
    series:
        Close-price series for a single ticker.
    period:
        RSI lookback window.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period, min_periods=period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0).clip(0.0, 100.0)


def build_features(
    prices: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Compute the full technical-feature panel (no target yet).

    Parameters
    ----------
    prices:
        Cleaned long-format OHLCV panel.
    cfg:
        Pipeline config namespace (``cfg.features`` is used).

    Returns
    -------
    pd.DataFrame
        Feature panel with columns ``ticker, date, close`` + features.
    """
    f = cfg.features
    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    grp = df.groupby("ticker", sort=False)

    # -- Trailing returns over multiple horizons -------------------------------
    # pct_change(periods=k) at row t = close_t / close_{t-k} - 1: the k-day
    # return ending at t, fully known at the close of day t (no leakage).
    for lag in f.return_lags:
        df[f"ret_lag_{lag}"] = grp["close"].pct_change(
            periods=int(lag), fill_method=None
        )

    # -- Trend: price vs moving averages --------------------------------------
    for window in f.sma_windows:
        sma = grp["close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        df[f"close_to_sma{window}"] = df["close"] / sma - 1.0
    sma10 = grp["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    sma50 = grp["close"].transform(lambda s: s.rolling(50, min_periods=50).mean())
    df["sma_spread_10_50"] = (sma10 - sma50) / sma50
    ema12 = grp["close"].transform(
        lambda s: s.ewm(span=f.ema_span, min_periods=f.ema_span, adjust=False).mean()
    )
    df["ema_ratio_12"] = df["close"] / ema12 - 1.0

    # -- Momentum: RSI, MACD, 52-week distance --------------------------------
    df["rsi_14"] = grp["close"].transform(lambda s: _rsi(s, f.rsi_period))
    macd_fast = grp["close"].transform(
        lambda s: s.ewm(span=f.macd.fast, adjust=False).mean()
    )
    macd_slow = grp["close"].transform(
        lambda s: s.ewm(span=f.macd.slow, adjust=False).mean()
    )
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.groupby(df["ticker"], sort=False).transform(
        lambda s: s.ewm(span=int(f.macd.signal), adjust=False).mean()
    )
    # Normalise the histogram by price scale so it is comparable across stocks
    df["macd_hist_norm"] = 100.0 * (macd_line - macd_signal) / df["close"]
    roll_max = grp["close"].transform(
        lambda s: s.rolling(int(f.high52_window), min_periods=60).max()
    )
    df["dist_52w_high"] = df["close"] / roll_max - 1.0

    # -- Volatility: realised vol + Bollinger ---------------------------------
    log_ret = np.log(df["close"]).groupby(df["ticker"], sort=False).diff()
    for window in f.volatility_windows:
        df[f"vol_{window}d_ann"] = (
            log_ret.pow(2)
            .groupby(df["ticker"], sort=False)
            .transform(lambda s, w=window: s.rolling(w, min_periods=w // 2).mean())
            .mul(252.0)
            .pow(0.5)
            * 100.0
        )
    win = f.bollinger.window
    roll_mean = grp["close"].transform(lambda s, w=win: s.rolling(w, min_periods=w).mean())
    roll_std = grp["close"].transform(lambda s, w=win: s.rolling(w, min_periods=w).std())
    upper = roll_mean + f.bollinger.num_std * roll_std
    lower = roll_mean - f.bollinger.num_std * roll_std
    df["bollinger_pct_b"] = (df["close"] - lower) / (upper - lower)
    df["bollinger_bandwidth"] = (upper - lower) / roll_mean

    # -- Microstructure: range, close location, volume z-score ----------------
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    vol_mean = grp["volume"].transform(
        lambda s, w=f.volume_window: s.rolling(w, min_periods=w).mean().shift(1)
    )
    vol_std = grp["volume"].transform(
        lambda s, w=f.volume_window: s.rolling(w, min_periods=w).std().shift(1)
    )
    # Volume z-score uses *shifted* rolling stats -> strictly causal
    df["volume_zscore_20"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # -- Cross-sectional rank (market regime proxy) ---------------------------
    # Rank of 30d volatility within the universe AS OF day t (no future data)
    df["vol_cs_rank"] = df.groupby("date")["vol_30d_ann"].rank(pct=True)

    keep = ["ticker", "date", "close"] + FEATURE_COLUMNS
    return df[keep]


def build_dataset(
    prices: pd.DataFrame,
    cfg,
    sector_map: dict,
) -> pd.DataFrame:
    """Attach the forward-horizon direction target and finalise the set.

    The target is the sign of the forward ``target_horizon``-day return.
    Rows are then sub-sampled to every ``sampling_step``-th trading date so
    that consecutive labels are non-overlapping.

    Parameters
    ----------
    prices:
        Cleaned long-format OHLCV panel.
    cfg:
        Pipeline config namespace.
    sector_map:
        Ticker -> sector mapping (one-hot encoded as features).

    Returns
    -------
    pd.DataFrame
        Model-ready dataset with features, one-hot sector dummies and
        ``target_up``; warm-up rows and rows without a forward close are
        dropped, and the frame is sorted chronologically.
    """
    features = build_features(prices, cfg)

    horizon = int(cfg.features.target_horizon)
    step = int(cfg.features.sampling_step)

    # Target: direction of the forward `horizon`-day return (strictly future)
    fwd_close = features.groupby("ticker", sort=False)["close"].shift(-horizon)
    next_ret = fwd_close / features["close"] - 1.0
    features[TARGET] = (next_ret > 0).astype("float64")
    # Rows without an available forward close (end of sample) are dropped
    features = features.dropna(subset=[TARGET])

    # Warm-up filter: drop the first `warmup_days` rows of every ticker
    # (rolling-indicator burn-in) via a per-ticker cumulative counter.
    warmup = int(cfg.features.warmup_days)
    within_ticker_pos = features.groupby("ticker", sort=False).cumcount()
    features = features[within_ticker_pos >= warmup]

    # Non-overlapping label sampling: keep every `step`-th trading date of
    # the GLOBAL calendar so sample dates align across tickers.
    all_dates = np.sort(features["date"].unique())
    sample_dates = set(all_dates[::step])
    features = features[features["date"].isin(sample_dates)]

    # One-hot sector features (fit on the full universe definition - static)
    features["sector"] = features["ticker"].map(sector_map)
    sector_dummies = pd.get_dummies(
        features["sector"], prefix="sector", drop_first=True, dtype="float64"
    )
    features = pd.concat([features.drop(columns=["sector"]), sector_dummies], axis=1)

    # Drop any residual NaN feature rows (early indicator burn-in)
    features = features.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    feature_cols = FEATURE_COLUMNS + [
        c for c in features.columns if c.startswith("sector_")
    ]

    logger.info(
        "Modelling dataset: %d rows, %d tickers, %d feature columns, "
        "%d-day horizon, sampled every %d days "
        "(target balance: %.1f%% up-periods)",
        len(features), features["ticker"].nunique(), len(feature_cols),
        horizon, step, 100.0 * features[TARGET].mean(),
    )
    return features
