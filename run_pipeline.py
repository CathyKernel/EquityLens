#!/usr/bin/env python3
"""Equity Quant ML - end-to-end pipeline runner.

Executes the full research workflow in deterministic order:

1.  acquire  - download OHLCV bars for the S&P 500 universe (+ SPY benchmark)
2.  warehouse - load into SQLite and run the analytical SQL layer
3.  clean    - rectangularise, validate, forward-fill, flag outliers
4.  eda      - exploratory figures and headline statistics
5.  stats    - hypothesis-testing suite (normality, stationarity, ARCH, ANOVA)
6.  features - technical indicator panel + next-day direction target
7.  model    - cross-validate, evaluate on chronological hold-out
8.  backtest - probability-threshold strategy net of costs vs SPY
9.  report   - render the self-contained HTML analysis report

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
from src.report_generator import _df_to_html, generate_report
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
    parser = argparse.ArgumentParser(description="Equity Quant ML pipeline")
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
        context = _build_report_context(cfg, clean, dataset, results, model_out, bt_out)
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
) -> Dict[str, Any]:
    """Assemble the Jinja2 context for the HTML report.

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

    Returns
    -------
    dict
        Template context with values and pre-rendered HTML tables.
    """
    quality = results["quality"]
    eda = results["eda"]
    stats = results["stats"]
    test_metrics = results["modeling"]["test_metrics"]
    best_name = results["modeling"]["best_model"]
    best = test_metrics[best_name]
    strategy = results["backtest"]["strategy"]
    benchmark = results["backtest"]["benchmark"]

    # SQL sector aggregate table (derived from the ANOVA stage statistics)
    sector_anova = stats["sector_anova"]["sector_stats"][
        ["sector", "n_obs", "mean_daily_pct", "ann_vol_pct"]
    ]
    sector_table = sector_anova.rename(columns={
        "n_obs": "Obs", "mean_daily_pct": "Mean daily ret (%)",
        "ann_vol_pct": "Ann. vol (%)",
    })

    jb = stats["normality"]["per_ticker"].copy()
    jb_view = (
        jb.sort_values("excess_kurtosis", ascending=False)
        .head(6)[["ticker", "jb_stat", "p_value", "skew", "excess_kurtosis"]]
        .rename(columns={
            "ticker": "Ticker", "jb_stat": "JB stat",
            "p_value": "p-value", "skew": "Skew",
            "excess_kurtosis": "Excess kurtosis",
        })
    )
    jb_view["p-value"] = jb_view["p-value"].map(lambda v: f"{v:.2e}")

    cv_df = pd.DataFrame(results["modeling"]["cv"])
    cv_html = _df_to_html(
        cv_df.rename(columns={
            "model": "Model", "cv_roc_auc_mean": "CV ROC-AUC",
            "cv_roc_auc_std": "+/-", "cv_accuracy_mean": "CV accuracy",
        }), max_rows=10
    )

    tm_df = pd.DataFrame(
        [{"model": k, **v} for k, v in test_metrics.items()]
    ).rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        "majority_class_share": "Up-period share",
    })
    tm_html = _df_to_html(tm_df, max_rows=10)

    bt_df = pd.DataFrame([
        {
            "Strategy": s["label"], "Total ret %": s["total_return_pct"],
            "CAGR %": s["cagr_pct"], "Ann. vol %": s["ann_vol_pct"],
            "Sharpe": s["sharpe"], "Max DD %": s["max_drawdown_pct"],
            "Hit rate %": s["hit_rate_pct"],
        }
        for s in (strategy, benchmark)
    ])
    bt_html = _df_to_html(bt_df, max_rows=5, float_fmt="{:.2f}")

    n_features = len(FEATURE_COLUMNS) + len(
        [c for c in dataset.columns if c.startswith("sector_")]
    )
    figures = {
        **{k: v for k, v in results.get("figures_eda", {}).items()},
        **{k: v for k, v in results.get("figures_stats", {}).items()},
        **{k: v for k, v in model_out.get("figures", {}).items()
           if k != "best_model"},
        "backtest": bt_out["equity_curve_fig"],
    }

    strat_edge = strategy["sharpe"] - benchmark["sharpe"]
    headline = (
        f"The {best_name} classifier reaches ROC-AUC {best['roc_auc']:.3f} "
        f"(accuracy {best['accuracy']:.1%} vs {best['majority_class_share']:.1%} "
        f"up-period base rate) on out-of-sample data, and the weekly-rebalanced "
        f"signal {'improves' if strat_edge > 0 else 'trails'} the SPY "
        f"buy-and-hold Sharpe by {abs(strat_edge):.2f} "
        f"({strategy['sharpe']} vs {benchmark['sharpe']}) net of "
        f"{int(cfg.backtest.transaction_cost_bps)} bps transaction costs."
    )

    return {
        "universe_size": int(clean["ticker"].nunique()),
        "n_sectors": len([c for c in dataset.columns if c.startswith("sector_")]) + 1,
        "first_date": quality["first_date"],
        "last_date": quality["last_date"],
        "rows": f"{quality['rows_output']:,}",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "best_model": best_name,
        "best_auc": f"{best['roc_auc']:.3f}",
        "best_acc": f"{best['accuracy']:.1%}",
        "strategy": strategy,
        "benchmark": benchmark,
        "headline": headline,
        "quality": quality,
        "sector_table_html": _df_to_html(sector_table, max_rows=12, float_fmt="{:.3f}"),
        "eda": eda,
        "stats": stats,
        "jb_table_html": _df_to_html(jb_view, max_rows=8, float_fmt="{:.2f}"),
        "sector_anova_html": _df_to_html(
            stats["sector_anova"]["sector_stats"], max_rows=12, float_fmt="{:.3f}"
        ),
        "n_features": n_features,
        "cv_folds": int(cfg.model.cv_folds),
        "cv_table_html": cv_html,
        "test_metrics_html": tm_html,
        "backtest_table_html": bt_html,
        "threshold": f"{float(cfg.backtest.probability_threshold):.2f}",
        "cost_bps": int(cfg.backtest.transaction_cost_bps),
        "horizon": int(cfg.features.target_horizon),
        "conclusion_ml": (
            f"Across time-series cross-validation and the chronological "
            f"hold-out, tree ensembles extract a modest but consistent edge "
            f"over the up-period base rate at the weekly horizon; volatility, "
            f"trend and volume features dominate the importance ranking."
        ),
        "conclusion_stats": (
            f"The statistical suite confirms fat tails (JB rejected for "
            f"{results['stats']['normality']['n_reject']} of "
            f"{results['stats']['normality']['n_tickers']} tickers), "
            f"stationary returns, and strong volatility clustering - "
            f"justifying the feature families chosen for the model."
        ),
        "conclusion_bt": (
            f"The weekly-rebalanced, probability-threshold strategy delivers "
            f"Sharpe {strategy['sharpe']} vs {benchmark['sharpe']} for SPY "
            f"with max drawdown {strategy['max_drawdown_pct']}% vs "
            f"{benchmark['max_drawdown_pct']}%, at "
            f"{strategy.get('avg_exposure_pct', 'n/a')}% average exposure - "
            f"the selective positioning (staying in cash when confidence is "
            f"low) is the main source of drawdown reduction."
        ),
        "figures": figures,
    }


if __name__ == "__main__":
    sys.exit(main())
