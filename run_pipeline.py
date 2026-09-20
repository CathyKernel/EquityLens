#!/usr/bin/env python3
"""EquityLens - end-to-end pipeline runner.

Executes the full research workflow in deterministic order:

1.  acquire  - download OHLCV bars for the S&P 500 universe (+ SPY benchmark)
2.  warehouse - load into SQLite and run the analytical SQL layer
3.  clean    - rectangularise, validate, forward-fill, flag outliers
4.  eda      - exploratory figures and headline statistics
5.  stats    - hypothesis-testing suite (normality, stationarity, ARCH, ANOVA)
6.  features - technical indicator panel + next-day direction target
7.  model    - cross-validate, evaluate on chronological hold-out
8.  backtest - probability-threshold strategy net of costs vs SPY
9.  report   - render the print-ready PDF analysis report

Usage
-----
    python run_pipeline.py                    # full run
    python run_pipeline.py --skip-download    # reuse cached Parquet data
    python run_pipeline.py --stage eda        # run a single stage (debug)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, resolve_path
from src.data_acquisition import load_or_download
from src.data_cleaning import clean_prices
from src.eda import run_eda
from src.feature_engineering import FEATURE_COLUMNS, build_dataset
from src.modeling import run_modeling
from src.report_generator import generate_report
from src.sql_layer import extract_all_aggregates, load_to_sqlite
from src.statistical_tests import run_all_tests

STAGES = [
    "acquire", "warehouse", "clean", "eda", "stats",
    "features", "model", "backtest", "report",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``skip_download`` and ``stage`` fields.
    """
    parser = argparse.ArgumentParser(description="EquityLens pipeline")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="reuse cached Parquet files instead of hitting Yahoo Finance",
    )
    parser.add_argument(
        "--stage", choices=STAGES, default=None,
        help="run a single stage only (implies cached download)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the pipeline end-to-end and persist all artefacts.

    Parameters
    ----------
    argv:
        Optional CLI argument list.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    cfg = load_config()

    # Resolve output locations once
    figures_dir = resolve_path(cfg.outputs.figures_dir)
    results_path = resolve_path(cfg.outputs.results_path)
    report_path = resolve_path(cfg.outputs.report_path)
    db_path = resolve_path(cfg.database.path)
    cache_dir = resolve_path(cfg.data.cache_dir)

    single_stage = args.stage
    # YAML mappings become SimpleNamespace; convert back to a plain dict
    sector_map = dict(vars(cfg.sector_map))
    universe_tickers = list(cfg.universe.tickers)
    if single_stage:
        # Run every stage from the start up to and including the requested one
        active_stages = STAGES[: STAGES.index(single_stage) + 1]
    elif args.skip_download:
        active_stages = ["warehouse"] + STAGES[1:]  # acquire handled via cache
    else:
        active_stages = STAGES
    t0 = time.time()
    results: Dict[str, Any] = {}

    # ---- 1. Acquire --------------------------------------------------------
    if "acquire" in active_stages or args.skip_download:
        universe = universe_tickers + [cfg.universe.benchmark]
        if args.skip_download:
            # Cache-only mode: drop tickers that have no Parquet file yet
            available = [t for t in universe if (cache_dir / f"{t}.parquet").exists()]
            missing = [t for t in universe if t not in available]
            if missing:
                logger.warning("No cached data for %s; they will be skipped.", missing)
            universe = available
        raw = load_or_download(
            tickers=universe,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            cache_dir=cache_dir,
        )
        results["raw_rows"] = int(len(raw))
        # Cache the benchmark separately for the backtest stage
        raw_bench = raw[raw["ticker"] == cfg.universe.benchmark].reset_index(drop=True)

    # ---- 2. Warehouse (SQLite + analytical SQL) -----------------------------
    if "warehouse" in active_stages:
        universe_only = raw[raw["ticker"].isin(universe_tickers)]
        rows_loaded = load_to_sqlite(universe_only, sector_map, db_path)
        sql_aggregates = extract_all_aggregates(db_path)
        results["sql_rows_loaded"] = int(rows_loaded)
        results["sql_aggregate_names"] = list(sql_aggregates.keys())

    # ---- 3. Clean ------------------------------------------------------------
    if "clean" in active_stages:
        universe_only = raw[raw["ticker"].isin(universe_tickers)]
        clean, quality = clean_prices(
            universe_only, sector_map,
            outlier_sigma=float(cfg.cleaning.outlier_sigma),
            min_history_days=int(cfg.cleaning.min_history_days),
        )
        results["quality"] = quality

    # ---- 4. EDA ---------------------------------------------------------------
    if "eda" in active_stages:
        eda_out = run_eda(clean, sector_map, figures_dir)
        eda_summary = eda_out.pop("eda_summary")
        results["eda"] = eda_summary
        results["figures_eda"] = eda_out

    # ---- 5. Statistical tests --------------------------------------------------
    if "stats" in active_stages:
        returns_wide = clean.pivot(index="date", columns="ticker", values="close").pct_change(
            fill_method=None
        )
        stats_out = run_all_tests(
            returns_wide, sector_map, figures_dir
        )
        results["stats"] = {
            k: v for k, v in stats_out.items() if k != "qq_plot"
        }
        results["figures_stats"] = {"qq_plot": stats_out["qq_plot"]}

    # ---- 6. Features -----------------------------------------------------------
    if "features" in active_stages:
        dataset = build_dataset(clean, cfg, sector_map)
        n_sector_dummies = len(
            [c for c in dataset.columns if c.startswith("sector_")]
        )
        results["dataset"] = {
            "rows": int(len(dataset)),
            "tickers": int(dataset["ticker"].nunique()),
            "n_features": len(FEATURE_COLUMNS) + n_sector_dummies,
            "up_period_share": round(float(dataset["target_up"].mean()), 4),
        }

    # ---- 7. Model -----------------------------------------------------------------
    if "model" in active_stages:
        model_out = run_modeling(dataset, cfg, figures_dir)
        results["modeling"] = {
            "best_model": model_out["best_model"],
            "test_metrics": model_out["test_metrics"],
            "cv": model_out["cv"].to_dict(orient="records"),
        }

    # ---- 8. Backtest ------------------------------------------------------------------
    if "backtest" in active_stages:
        bt_out = run_backtest(
            model_out["test_frame"], raw_bench, clean, cfg, figures_dir
        )
        results["backtest"] = {
            "strategy": bt_out["strategy"],
            "benchmark": bt_out["benchmark"],
        }

    # ---- 9. Report -----------------------------------------------------------------------
    if "report" in active_stages:
        context = _build_report_context(
            cfg, clean, dataset, results, model_out, bt_out, sql_aggregates
        )
        generate_report(context, report_path)

    # ---- Persist machine-readable results ------------------------------------------------
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str))
    elapsed = time.time() - t0
    logger.info("Pipeline finished in %.1f s. Results: %s", elapsed, results_path)
    return 0


def _build_report_context(
    cfg, clean: pd.DataFrame, dataset: pd.DataFrame,
    results: Dict[str, Any], model_out: Dict[str, Any], bt_out: Dict[str, Any],
    sql_aggregates: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Assemble the context dictionary for the PDF report renderer.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    clean:
        Cleaned OHLCV panel.
    dataset:
        Model-ready feature dataset.
    results:
        Aggregated pipeline results dict.
    model_out:
        Modelling stage output.
    bt_out:
        Backtest stage output.
    sql_aggregates:
        Analytical SQL extract frames from the warehouse stage.

    Returns
    -------
    dict
        Report context: raw values and DataFrames; all presentation
        (formatting, tables, figures) is handled by the report generator.
    """
    quality = results["quality"]
    stats = results["stats"]
    test_metrics = results["modeling"]["test_metrics"]
    best_name = results["modeling"]["best_model"]
    strategy = results["backtest"]["strategy"]
    benchmark = results["backtest"]["benchmark"]
    sector_map = dict(vars(cfg.sector_map))

    sector_counts = list(
        pd.Series(sector_map).value_counts().sort_index().items()
    )

    # -- SQL extracts surfaced in the report -----------------------------------
    sector_agg = sql_aggregates["sector_aggregate"]
    league = (
        sql_aggregates["volatility_league"]
        .sort_values("annual_vol_pct", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    annual = (
        sql_aggregates["annual_summary"]
        .groupby("year")
        .agg(
            mean_annual_return_pct=("annual_return_pct", "mean"),
            share_positive_pct=(
                "annual_return_pct", lambda s: 100.0 * (s > 0).mean()
            ),
        )
        .reset_index()
        .sort_values("year")
    )
    annual["mean_annual_return_pct"] = annual["mean_annual_return_pct"].round(1)
    annual["share_positive_pct"] = annual["share_positive_pct"].round(0)

    n_sector_dummies = len([c for c in dataset.columns if c.startswith("sector_")])
    feature_families = [
        (
            "Momentum",
            "Short-horizon return lags capture recent performance drift.",
            4,
            "ret_lag_1, ret_lag_5, ret_lag_10, ret_lag_21",
        ),
        (
            "Trend",
            "Price versus moving averages and the 52-week high encodes "
            "medium-term trend.",
            6,
            "close_to_sma10/20/50, sma_spread_10_50, ema_ratio_12, "
            "dist_52w_high",
        ),
        (
            "Oscillators",
            "Overbought / oversold pressure indicators.",
            2,
            "rsi_14, macd_hist_norm",
        ),
        (
            "Volatility",
            "Realised volatility and Bollinger positioning proxy the risk "
            "regime (ARCH evidence from Section 5).",
            6,
            "vol_10d_ann, vol_30d_ann, bollinger_pct_b, "
            "bollinger_bandwidth, daily_range, close_location",
        ),
        (
            "Volume",
            "Abnormal trading activity confirms or contradicts price moves.",
            2,
            "volume_zscore_20, vol_cs_rank",
        ),
        (
            "Sector one-hot",
            "GICS membership absorbs cross-sectional heterogeneity (ANOVA "
            "evidence from Section 5).",
            n_sector_dummies,
            "GICS sectors, drop_first encoding",
        ),
    ]

    figures = {
        **results.get("figures_eda", {}),
        **results.get("figures_stats", {}),
        **{k: v for k, v in model_out.get("figures", {}).items()
           if k != "best_model"},
        "backtest": bt_out["equity_curve_fig"],
    }

    return {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "window": {
            "start": quality["first_date"],
            "end": quality["last_date"],
            "sessions": quality["trading_days"],
        },
        "universe": {
            "size": int(clean["ticker"].nunique()),
            "sectors": len(sector_counts),
            "benchmark": cfg.universe.benchmark,
        },
        "quality": quality,
        "sector_counts": sector_counts,
        "sql": {
            "sector_aggregate": sector_agg,
            "volatility_league": league,
            "annual_summary": annual,
        },
        "eda": results["eda"],
        "stats": stats,
        "dataset": results["dataset"],
        "feature_families": feature_families,
        "modeling": {
            "best_model": best_name,
            "cv": pd.DataFrame(results["modeling"]["cv"]),
            "test_metrics": test_metrics,
        },
        "backtest": {"strategy": strategy, "benchmark": benchmark},
        "params": {
            "threshold": float(cfg.backtest.probability_threshold),
            "cost_bps": int(cfg.backtest.transaction_cost_bps),
            "risk_free_rate": float(cfg.backtest.risk_free_rate),
            "horizon": int(cfg.features.target_horizon),
            "sampling_step": int(cfg.features.sampling_step),
            "cv_folds": int(cfg.model.cv_folds),
            "test_ratio": float(cfg.model.test_ratio),
        },
        "figures": figures,
    }


if __name__ == "__main__":
    sys.exit(main())
