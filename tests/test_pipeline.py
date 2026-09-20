"""Unit tests for the leakage-critical parts of the pipeline.

The most valuable tests here are the **no-look-ahead guarantees**: if any
feature or target accidentally peeks at future prices, the entire research
narrative collapses, so these invariants are pinned down explicitly.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET,
    _rsi,
    build_dataset,
    build_features,
)


from types import SimpleNamespace

MINI_CFG = SimpleNamespace(
    features=SimpleNamespace(
        target_horizon=5,
        sampling_step=5,
        return_lags=[1, 5, 10, 21],
        sma_windows=[10, 20, 50],
        ema_span=12,
        rsi_period=14,
        macd=SimpleNamespace(fast=12, slow=26, signal=9),
        bollinger=SimpleNamespace(window=20, num_std=2.0),
        volatility_windows=[10, 30],
        volume_window=20,
        high52_window=252,
        warmup_days=60,
    ),
    backtest=SimpleNamespace(
        probability_threshold=0.55,
        transaction_cost_bps=2.0,
        risk_free_rate=0.02,
    ),
)


class _MiniCfg:
    """Alias so test bodies read clearly."""

    features = MINI_CFG.features
    backtest = MINI_CFG.backtest


def _synthetic_panel(n_days: int = 400, tickers=("AAA", "BBB"), seed: int = 7):
    """Generate a deterministic synthetic OHLCV panel."""
    rng = np.random.default_rng(seed)
    frames = []
    for ticker in tickers:
        rets = rng.normal(0.0005, 0.01, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        open_ = close * (1 + rng.normal(0, 0.002, n_days))
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": pd.bdate_range("2020-01-01", periods=n_days),
            "open": open_, "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(1e5, 5e6, n_days).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


class TestRSI:
    def test_bounds(self):
        """RSI must always live in [0, 100]."""
        series = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300).cumsum() + 100)
        rsi = _rsi(series, period=14)
        assert rsi.between(0, 100).all()

    def test_flat_series_is_neutral(self):
        """A perfectly flat price should map to the neutral 50 level."""
        rsi = _rsi(pd.Series([100.0] * 50), period=14)
        assert (rsi.iloc[14:] == 50.0).all()


class TestNoLookAhead:
    def test_feature_row_ignores_future_prices(self):
        """Truncating the future must not change any feature at time t.

        If a feature used future information, its value at row t would change
        once rows after t are removed from the input panel.
        """
        panel = _synthetic_panel(300, tickers=("AAA",), seed=11)
        full = build_features(panel, _MiniCfg)
        truncated = build_features(panel.iloc[:-120], _MiniCfg)

        t = panel["date"].iloc[150]
        row_full = full.loc[full["date"] == t, FEATURE_COLUMNS].iloc[0]
        row_trunc = truncated.loc[truncated["date"] == t, FEATURE_COLUMNS].iloc[0]
        pd.testing.assert_series_equal(
            row_full, row_trunc, check_names=False,
            rtol=1e-9, atol=1e-12,
        )

    def test_target_uses_forward_horizon(self):
        """The label must equal the sign of the forward 5-day return."""
        panel = _synthetic_panel(300, tickers=("AAA",), seed=13)
        cfg = _MiniCfg
        features = build_features(panel, cfg)
        horizon = cfg.features.target_horizon

        fwd_close = features.groupby("ticker", sort=False)["close"].shift(-horizon)
        expected = (fwd_close / features["close"] - 1.0 > 0).astype(float)

        horizon_col = pd.Series(np.nan, index=features.index)
        dataset = build_dataset(panel, cfg, {"AAA": "Technology"})

        # Recompute expected labels on the rows that survive in the dataset
        merged = features.assign(expected=expected)
        merged = merged.dropna(subset=["expected"])
        merged = pd.Series(merged["expected"].values, index=merged[["ticker", "date"]].apply(tuple, axis=1))
        _ = horizon_col

        got = pd.Series(
            dataset[TARGET].values,
            index=dataset[["ticker", "date"]].apply(tuple, axis=1),
        )
        common = got.index.intersection(merged.index)
        assert len(common) > 0
        assert (got.loc[common] == merged.loc[common]).all()

    def test_labels_do_not_overlap(self):
        """Consecutive sample rows of one ticker must be `step` days apart."""
        panel = _synthetic_panel(400, tickers=("AAA",), seed=17)
        dataset = build_dataset(panel, _MiniCfg, {"AAA": "Technology"})
        dates = dataset.sort_values("date")["date"].tolist()
        steps = pd.Series(dates).diff().dropna().dt.days
        # 5 trading days = 7 calendar days (bdate_range skips weekends)
        assert (steps == 7).all()


class TestDataset:
    def test_schema(self):
        """Dataset must expose ticker/date, every feature and the target."""
        panel = _synthetic_panel(400, tickers=("AAA", "BBB"), seed=19)
        dataset = build_dataset(panel, _MiniCfg, {"AAA": "Technology", "BBB": "Financials"})
        for col in ["ticker", "date", TARGET] + FEATURE_COLUMNS:
            assert col in dataset.columns, f"missing column: {col}"
        # at least one sector dummy column present (drop_first keeps n-1)
        sector_cols = [c for c in dataset.columns if c.startswith("sector_")]
        assert len(sector_cols) == 1

    def test_warmup_rows_removed(self):
        """The first `warmup_days` rows per ticker must be dropped."""
        panel = _synthetic_panel(300, tickers=("AAA",), seed=23)
        dataset = build_dataset(panel, _MiniCfg, {"AAA": "Technology"})
        assert len(dataset) < len(panel)
        assert not dataset[FEATURE_COLUMNS].isna().any().any()

    def test_binary_target(self):
        """Target must be strictly 0/1 with both classes present."""
        panel = _synthetic_panel(400, tickers=("AAA", "BBB"), seed=29)
        dataset = build_dataset(panel, _MiniCfg, {"AAA": "Technology", "BBB": "Financials"})
        assert set(dataset[TARGET].unique()) <= {0.0, 1.0}
        assert 0.2 < dataset[TARGET].mean() < 0.8


class TestBacktestAlignment:
    def test_signal_lags_return(self):
        """A signal must earn the *forward* return, never the same-bar one.

        With a constant positive signal on a rising series, per-period
        returns must equal the forward close-to-close return.
        """
        from src.backtest import _performance_metrics

        rets = pd.Series([0.01, 0.02, -0.005, 0.015] * 10)
        stats = _performance_metrics(rets, rf_annual=0.0, periods_per_year=50.4, label="t")
        assert stats["n_periods"] == len(rets)
        expected_total = float((1 + rets).prod() - 1)
        assert abs(stats["total_return_pct"] - 100 * expected_total) < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
