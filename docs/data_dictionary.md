# Data Dictionary

This document describes every field produced by the pipeline: raw OHLCV
columns, SQLite warehouse tables, engineered features and the modelling
target. All prices are **split- and dividend-adjusted** (Yahoo Finance,
`auto_adjust=True`), expressed in USD.

## 1. Raw layer — `data/raw/<TICKER>.parquet` and `daily_prices` (SQLite)

| Field | Type | Description |
|---|---|---|
| `ticker` | TEXT | Ticker symbol, uppercase (e.g. `AAPL`, `MSFT`). 66 constituents + `SPY` benchmark. |
| `date` | DATE | Trading date (NYSE calendar), timezone-naive. |
| `open` | REAL | Adjusted opening price of the session. |
| `high` | REAL | Adjusted highest price of the session. |
| `low` | REAL | Adjusted lowest price of the session. |
| `close` | REAL | Adjusted closing price of the session. |
| `volume` | REAL | Shares traded during the session. |

**Data-quality flags added during cleaning** (`is_outlier`): `1` when the
session's daily return exceeds ±5 pooled standard deviations. Outliers are
*flagged but kept* — extreme equity moves are genuine information.

## 2. SQLite warehouse — `database/market_data.db`

| Table | Grain | Description |
|---|---|---|
| `daily_prices` | ticker × date | Canonical OHLCV panel (primary key: ticker, date). |
| `ticker_sector` | ticker | Static ticker → GICS sector mapping (11 sectors). |

**Analytical views** (produced as query results in
`src/sql_layer.extract_all_aggregates`, not persisted):

| Query | Grain | Key fields |
|---|---|---|
| `monthly_returns` | ticker × month | `monthly_return_pct` — month-over-month close change via `LAG()` window. |
| `rolling_volatility` | ticker × date | `rolling_vol_30d_pct` — annualised 30-day realised vol via window frame. |
| `annual_summary` | ticker × year | `annual_return_pct`, `avg_daily_volume_m` (join with sectors). |
| `sector_aggregate` | sector | `mean_daily_ret_pct`, `ann_vol_pct`, `annualised_sharpe` (pure-SQL Sharpe). |
| `volatility_league` | ticker | Per-ticker annualised vol, Sharpe and sector label, ranked. |

## 3. Feature layer — modelling dataset

All features are **strictly causal**: computable at the close of day `t`
using only information up to `t`. Values are floats unless noted.

### Trailing returns (momentum)

| Feature | Description |
|---|---|
| `ret_lag_1` | 1-day trailing return: `close_t / close_{t-1} - 1`. |
| `ret_lag_5` | 5-day trailing return (one trading week). |
| `ret_lag_10` | 10-day trailing return (two weeks). |
| `ret_lag_21` | 21-day trailing return (one trading month). |

### Trend

| Feature | Description |
|---|---|
| `close_to_sma10` | `close_t / SMA10_t - 1` — distance to the 10-day mean. |
| `close_to_sma20` | Distance to the 20-day simple moving average. |
| `close_to_sma50` | Distance to the 50-day simple moving average. |
| `sma_spread_10_50` | `(SMA10 - SMA50) / SMA50` — classic golden-cross proxy. |
| `ema_ratio_12` | Distance to the 12-day exponential moving average. |

### Momentum oscillators

| Feature | Description |
|---|---|
| `rsi_14` | Relative Strength Index (Wilder EMA), bounded [0, 100]. |
| `macd_hist_norm` | MACD histogram (EMA12 − EMA26 minus its 9-day EMA), normalised by price ×100. |
| `dist_52w_high` | `close_t / rolling_252d_max_t - 1` — distance to the 52-week high. |

### Volatility

| Feature | Description |
|---|---|
| `vol_10d_ann` | Annualised realised volatility from 10-day squared log-returns (×100). |
| `vol_30d_ann` | Same over a 30-day window. |
| `bollinger_pct_b` | Bollinger %B: position of the close within the 20-day ±2σ band. |
| `bollinger_bandwidth` | Band width divided by the 20-day mean — squeeze detection. |

### Microstructure

| Feature | Description |
|---|---|
| `daily_range` | `(high_t − low_t) / close_t` — intraday range. |
| `close_location` | Where the close sits inside the day's range: `(close − low) / (high − low)`. |
| `volume_zscore_20` | Volume z-score vs the *shifted* 20-day rolling mean/std (strictly causal). |
| `vol_cs_rank` | Cross-sectional percentile rank of `vol_30d_ann` within the universe **as of day t** (regime proxy). |

### Sector dummies

`sector_<Name>` — one-hot GICS sector indicators (10 columns + dropped
baseline, from the static 11-sector map).

### Target

| Field | Description |
|---|---|
| `target_up` | **1** if the forward 5-trading-day return `close_{t+5} / close_t − 1` is positive, else **0**. Sampled on every 5th trading date so labels never overlap. |

## 4. Output artefacts

| Artefact | Description |
|---|---|
| `outputs/results.json` | Stage-by-stage metrics: data-quality report, EDA summary, test statistics, CV/test model metrics, backtest stats. |
| `outputs/report.pdf` | Print-ready 17-page PDF research report (ReportLab): cover, clickable TOC, 11 embedded figures, formatted result tables. |
| `outputs/figures/*.png` | The 11 figures referenced by the report and README. |
