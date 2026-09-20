"""SQLite data-warehouse layer.

Persists the OHLCV panel and a sector mapping into a local SQLite database,
then exposes **analytical SQL queries** (CTEs, window functions, joins and
aggregations) that produce model-ready and report-ready aggregates.

Why a SQL layer at all?
------------------------
In production quant stacks, analysts rarely touch raw files: market data
lives in a warehouse and is extracted through SQL. This module mirrors that
workflow so the pipeline demonstrates the SQL literacy expected in data
science interviews: window functions (``OVER (PARTITION BY ... ORDER BY
...)``), common table expressions, index hints and grouped aggregation.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DDL_PRICES = """
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker  TEXT    NOT NULL,
    date    DATE    NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
"""

DDL_SECTORS = """
CREATE TABLE IF NOT EXISTS ticker_sector (
    ticker TEXT PRIMARY KEY,
    sector TEXT NOT NULL
);
"""

IDX_PRICES_DATE = (
    "CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices (date);"
)


def load_to_sqlite(
    prices: pd.DataFrame,
    sector_map: Dict[str, str],
    db_path: pathlib.Path,
) -> int:
    """Persist the price panel and sector mapping into SQLite.

    Parameters
    ----------
    prices:
        Long-format OHLCV frame.
    sector_map:
        Mapping of ticker -> GICS sector.
    db_path:
        Target SQLite database file. Parent directories are created.

    Returns
    -------
    int
        Number of price rows written.
    """
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(DDL_PRICES)
        conn.execute(DDL_SECTORS)
        conn.execute(IDX_PRICES_DATE)

        records = list(
            prices[["ticker", "date", "open", "high", "low", "close", "volume"]]
            .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
            .itertuples(index=False, name=None)
        )
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )

        sector_rows = [
            (str(t), str(s)) for t, s in sector_map.items()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO ticker_sector VALUES (?, ?)", sector_rows
        )

        row_count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        logger.info(
            "SQLite warehouse loaded: %d price rows, %d sector mappings, db=%s",
            row_count, len(sector_rows), db_path,
        )
    return int(row_count)


# ---------------------------------------------------------------------------
# Analytical queries
# ---------------------------------------------------------------------------

SQL_MONTHLY_RETURNS = """
WITH month_end AS (
    SELECT
        ticker,
        date,
        close,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, strftime('%Y-%m', date)
            ORDER BY date DESC
        ) AS rn
    FROM daily_prices
),
monthly_close AS (
    SELECT ticker, strftime('%Y-%m', date) AS month, close
    FROM month_end WHERE rn = 1
),
lagged AS (
    SELECT
        ticker,
        month,
        close,
        LAG(close) OVER (PARTITION BY ticker ORDER BY month) AS prev_close
    FROM monthly_close
)
SELECT
    ticker,
    month,
    100.0 * (close / prev_close - 1.0) AS monthly_return_pct
FROM lagged
WHERE prev_close IS NOT NULL
ORDER BY ticker, month;
"""


SQL_ROLLING_VOLATILITY = """
WITH log_returns AS (
    SELECT
        ticker,
        date,
        LN(close / LAG(close) OVER (
            PARTITION BY ticker ORDER BY date
        )) AS log_ret
    FROM daily_prices
)
SELECT
    ticker,
    date,
    100.0 * SQRT(
        252.0
        * AVG(log_ret * log_ret) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )
    ) AS rolling_vol_30d_pct
FROM log_returns
WHERE log_ret IS NOT NULL
ORDER BY ticker, date;
"""


SQL_ANNUAL_SUMMARY = """
WITH annual_close AS (
    SELECT
        ticker,
        strftime('%Y', date) AS yr,
        -- close on the last trading day of each year, via window ranking
        close,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, strftime('%Y', date)
            ORDER BY date DESC
        ) AS rn,
        AVG(volume) OVER (
            PARTITION BY ticker, strftime('%Y', date)
        ) AS avg_daily_volume
    FROM daily_prices
),
yearly AS (
    SELECT ticker, yr, close, avg_daily_volume FROM annual_close WHERE rn = 1
),
annual_ret AS (
    SELECT
        ticker,
        yr,
        100.0 * (close / LAG(close) OVER (PARTITION BY ticker ORDER BY yr) - 1.0)
            AS annual_return_pct,
        avg_daily_volume
    FROM yearly
)
SELECT
    s.ticker,
    COALESCE(s.sector, 'Unmapped') AS sector,
    a.yr AS year,
    ROUND(a.annual_return_pct, 2) AS annual_return_pct,
    ROUND(a.avg_daily_volume / 1e6, 2) AS avg_daily_volume_m
FROM annual_ret a
JOIN ticker_sector s ON s.ticker = a.ticker
WHERE a.annual_return_pct IS NOT NULL
ORDER BY a.ticker, a.yr;
"""


SQL_SECTOR_AGGREGATE = """
WITH daily_ret AS (
    SELECT
        d.ticker,
        d.date,
        100.0 * (d.close / LAG(d.close) OVER (
            PARTITION BY d.ticker ORDER BY d.date
        ) - 1.0) AS ret_pct
    FROM daily_prices d
)
SELECT
    COALESCE(s.sector, 'Unmapped') AS sector,
    COUNT(r.ret_pct)                                  AS n_obs,
    ROUND(AVG(r.ret_pct), 4)                          AS mean_daily_ret_pct,
    ROUND(100.0 * SQRT(AVG(r.ret_pct * r.ret_pct)), 4) AS daily_vol_pct,
    ROUND(100.0 * SQRT(252.0 * AVG(r.ret_pct * r.ret_pct)), 2) AS ann_vol_pct,
    -- annualised Sharpe with 2% risk-free rate, computed in pure SQL
    ROUND(
        252.0 * (AVG(r.ret_pct) - 0.02 / 252.0)
        / (100.0 * SQRT(252.0 * AVG(r.ret_pct * r.ret_pct))),
        3
    ) AS annualised_sharpe
FROM daily_ret r
JOIN ticker_sector s ON s.ticker = r.ticker
WHERE r.ret_pct IS NOT NULL
GROUP BY COALESCE(s.sector, 'Unmapped')
ORDER BY ann_vol_pct DESC;
"""


# ---------------------------------------------------------------------------
# Headline aggregate: per-ticker volatility league table (joined with sectors)
# ---------------------------------------------------------------------------
SQL_VOLATILITY_LEAGUE = """
WITH daily_ret AS (
    SELECT
        ticker,
        100.0 * (close / LAG(close) OVER (
            PARTITION BY ticker ORDER BY date
        ) - 1.0) AS ret_pct
    FROM daily_prices
)
SELECT
    r.ticker,
    COALESCE(s.sector, 'Unmapped') AS sector,
    COUNT(r.ret_pct) AS n_days,
    ROUND(AVG(r.ret_pct), 4) AS mean_daily_ret_pct,
    ROUND(100.0 * SQRT(AVG(r.ret_pct * r.ret_pct)), 3) AS daily_vol_pct,
    ROUND(100.0 * SQRT(252.0 * AVG(r.ret_pct * r.ret_pct)), 2) AS annual_vol_pct,
    ROUND(
        252.0 * (AVG(r.ret_pct) - 0.02 / 252.0)
        / (100.0 * SQRT(252.0 * AVG(r.ret_pct * r.ret_pct))),
        3
    ) AS annualised_sharpe
FROM daily_ret r
JOIN ticker_sector s ON s.ticker = r.ticker
WHERE r.ret_pct IS NOT NULL
GROUP BY r.ticker, COALESCE(s.sector, 'Unmapped')
ORDER BY annualised_sharpe DESC;
"""


def run_query(db_path: pathlib.Path, sql: str) -> pd.DataFrame:
    """Execute an analytical SQL query and return a DataFrame.

    Parameters
    ----------
    db_path:
        SQLite database file.
    sql:
        SQL statement to execute (SELECT only).

    Returns
    -------
    pd.DataFrame
        Query result set.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, parse_dates=None)


def extract_all_aggregates(db_path: pathlib.Path) -> Dict[str, pd.DataFrame]:
    """Run the full catalogue of analytical queries.

    Parameters
    ----------
    db_path:
        SQLite database file.

    Returns
    -------
    dict
        Mapping of query name -> result DataFrame, with keys
        ``monthly_returns``, ``rolling_volatility``, ``annual_summary``,
        ``sector_aggregate`` and ``volatility_league``.
    """
    catalogue = {
        "monthly_returns": SQL_MONTHLY_RETURNS,
        "rolling_volatility": SQL_ROLLING_VOLATILITY,
        "annual_summary": SQL_ANNUAL_SUMMARY,
        "sector_aggregate": SQL_SECTOR_AGGREGATE,
        "volatility_league": SQL_VOLATILITY_LEAGUE,
    }
    results: Dict[str, pd.DataFrame] = {}
    for name, sql in catalogue.items():
        logger.info("Running SQL query: %s", name)
        results[name] = run_query(db_path, sql)
    return results
