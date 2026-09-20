# EquityLens

**S&P 500 Quantitative Market Analysis & Machine-Learning Trading Signals**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![tests](https://github.com/CathyKernel/EquityLens/actions/workflows/tests.yml/badge.svg)](https://github.com/CathyKernel/EquityLens/actions/workflows/tests.yml)

An end-to-end quantitative research pipeline over **10+ years of daily data for
66 large-cap S&P 500 constituents across all 11 GICS sectors (177,738 cleaned
daily observations)**: market-data ingestion, an SQLite analytical warehouse
driven by window functions and CTEs, rigorous data-quality validation,
exploratory analysis, a formal statistical hypothesis-testing suite,
leakage-safe feature engineering, next-week direction classification with
time-series cross-validation, and an out-of-sample backtest net of transaction
costs — wrapped up in a **print-ready 17-page PDF research report**.

> This is a research/educational project. It is **not** investment advice.

---

## Key results (real pipeline output, reproducible)

| Metric | Value |
|---|---|
| Universe | 66 S&P 500 stocks, all 11 GICS sectors + SPY benchmark |
| Data window | 2016-01-04 → 2026-09-18 (2,693 trading sessions) |
| Clean sample | 177,738 daily observations; 660 outlier days flagged at 5σ (kept, not deleted) |
| Modelling sample | 34,782 non-overlapping weekly observations, 30 feature columns (20 causal technical features + 10 sector one-hot) |
| Best classifier (hold-out) | XGBoost — **ROC-AUC 0.515**, accuracy 52.6% vs 53.2% up-period base rate |
| Out-of-sample strategy Sharpe | **1.74** vs 1.28 for SPY buy-and-hold (net of 2 bps costs) |
| Out-of-sample CAGR / max drawdown | **25.7% / -10.5%** vs 19.4% / -14.0% for SPY, at 52.7% average exposure |
| Fat-tail evidence | Jarque-Bera rejects normality for **66/66** tickers (pooled excess kurtosis ≈ 18.2) |
| Volatility clustering | Ljung-Box on squared returns: p ≈ 0 at lags 1–20 (ARCH effects) |
| Sector heterogeneity | One-way ANOVA across 11 sectors: F = 2.29, **p = 0.011** |

The honest headline: cross-validated AUC ≈ 0.50 (2016–2024) is consistent with
weak-form market efficiency — daily-direction predictability is essentially
nil. At the **weekly horizon** with non-overlapping labels, momentum and
volatility features carry a thin but real edge, and a selective
probability-threshold policy converts that edge into meaningfully better
risk-adjusted performance out of sample. The methodology — not the returns —
is the product of this repository.

## Pipeline architecture

```mermaid
flowchart LR
    A[yfinance API\n66 tickers x 10y] --> B[Parquet cache\ndata/raw]
    B --> C[SQLite warehouse\nCTEs + window functions]
    C --> D[Cleaning & validation\nquality report]
    D --> E[EDA\n6 figures]
    D --> F[Hypothesis tests\nJB / ADF / Ljung-Box / ANOVA]
    D --> G[Feature engineering\n20 technical + 10 sector features\nnon-overlapping labels]
    G --> H[Model comparison\nLogReg / RF / XGBoost\nTimeSeriesSplit CV]
    H --> I[Backtest\nweekly rebalance\n2 bps costs]
    E --> J[PDF report\noutputs/report.pdf]
    F --> J
    H --> J
    I --> J
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/CathyKernel/EquityLens.git
cd EquityLens

# 2. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline (~5-10 min; downloads 67 tickers on first run)
python run_pipeline.py

# Useful variants
python run_pipeline.py --skip-download   # reuse the Parquet cache
python run_pipeline.py --stage eda       # run stages up to a given point
python -m pytest tests/ -v               # unit tests (leakage guards)
```

Outputs land in `outputs/`:

| Artefact | Description |
|---|---|
| `outputs/report.pdf` | **17-page print-ready PDF research report** (cover, clickable TOC, 11 embedded figures, formatted result tables) |
| `outputs/figures/*.png` | 11 publication-quality charts |
| `outputs/results.json` | Machine-readable summary of every stage |
| `database/market_data.db` | SQLite warehouse with the analytical tables |
| `data/raw/*.parquet` | Cached OHLCV bars per ticker |

## Methodology highlights

These are the design decisions an interviewer is most likely to probe —
each is implemented deliberately and covered by unit tests:

1. **No look-ahead bias.** Every feature at `(ticker, t)` uses only data
   available at the close of day `t`; the label is the direction of the
   forward 5-trading-day return. `tests/test_pipeline.py::TestNoLookAhead`
   proves that truncating the future leaves features at `t` bit-identical.
2. **Non-overlapping labels.** The modelling set keeps every 5th trading
   date, so consecutive weekly labels never overlap — otherwise
   cross-validation scores would be inflated by label autocorrelation.
3. **Time-series-aware validation.** `TimeSeriesSplit` folds inside the
   training period (never fitting on the future to predict the past); the
   final 20% of dates are an untouched chronological hold-out.
4. **Honest baselines.** Accuracy is reported against the majority-class
   (up-period) base rate; the backtest benchmark (SPY) is measured over
   identical rebalance windows, including transaction costs.
5. **Cost-aware backtesting.** 2 basis points are charged per unit of
   position turnover; exposure, turnover, hit rate and drawdown are all
   reported, not just the equity curve.
6. **SQL as a first-class citizen.** Rolling volatility, monthly returns
   and Sharpe-by-sector are computed in SQLite with window functions and
   CTEs (see `src/sql_layer.py`), mirroring a real warehouse workflow.
7. **Outliers are flagged, never silently deleted.** A 5-sigma rule marks
   660 extreme sessions across 66 tickers and 10+ years; they stay in the
   sample because extreme moves are genuine market information.

## Repository layout

```
equitylens/
├── run_pipeline.py            # one-command orchestration (9 stages)
├── config.yaml                # single source of truth for all parameters
├── src/
│   ├── data_acquisition.py    # yfinance download + Parquet cache + retries
│   ├── sql_layer.py           # SQLite DDL + analytical SQL catalogue
│   ├── data_cleaning.py       # rectangularisation, validation, outliers
│   ├── eda.py                 # 6 exploratory figures + summary stats
│   ├── statistical_tests.py   # JB / ADF / Ljung-Box / ANOVA suite
│   ├── feature_engineering.py # 20 technical features + sector one-hot + forward-horizon label
│   ├── modeling.py            # 3 classifiers, TimeSeriesSplit CV, figures
│   ├── backtest.py            # weekly-rebalance backtest vs SPY
│   └── report_generator.py    # ReportLab → print-ready PDF report
├── tests/
│   └── test_pipeline.py       # leakage guards + schema/target invariants
├── docs/
│   └── data_dictionary.md     # every raw & feature field documented
├── .github/workflows/         # CI: pytest on every push/PR
├── data/raw/                  # Parquet cache (created at runtime)
├── database/                  # SQLite warehouse (created at runtime)
└── outputs/                   # figures, results.json, report.pdf
```

## Statistical findings worth citing

- **Fat tails:** pooled daily returns show excess kurtosis ≈ 18.2 and
  Jarque-Bera rejects normality for all 66 tickers — Gaussian risk models
  materially understate tail risk, which favours tree-based learners and
  non-parametric statistics.
- **Volatility clustering:** Ljung-Box on *squared* returns is highly
  significant (p ≈ 0 at all lags) while linear return autocorrelation is
  weak — classic ARCH effects, which motivate the realised-volatility and
  Bollinger features.
- **Stationarity:** ADF rejects a unit root in returns but not in prices —
  the formal justification for modelling returns rather than price levels.
- **Sector heterogeneity:** ANOVA across the 11 GICS sectors is significant
  (F = 2.29, p = 0.011), supporting the inclusion of sector dummies.
- **Cross-sectional structure:** the average pairwise return correlation is
  0.34 — diversified enough for sector effects to be identifiable, but
  clearly driven by one dominant market factor.

## Configuration

Everything is parameterised in `config.yaml`: the ticker universe and
sector map, the data window, feature parameters (RSI period, MACD spans,
Bollinger width, sampling step), model hyperparameters, and backtest
settings (threshold, costs, risk-free rate). No magic numbers in code.

## Limitations

- 66 large-cap tickers is a curated cross-section of the index, not the
  full 500; results may not generalise to small caps or other markets.
- Close-to-close labels ignore overnight gaps and execution slippage.
- The out-of-sample window (Aug 2024 → Sep 2026) was a strong bull market;
  the edge is regime-dependent, and the CV period (2016–2024) shows
  cross-validated AUC ≈ 0.49, i.e. little exploitable signal there.
- No hyperparameter search, ensembling, regime features, or macro overlay.

## Data source & attribution

Daily OHLCV bars are downloaded via
[yfinance](https://github.com/ranaroussi/yfinance) from Yahoo Finance
(split- and dividend-adjusted, `auto_adjust=True`). Cached Parquet files are
for research convenience only.

## License

[MIT](LICENSE)
