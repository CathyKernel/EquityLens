"""Market data acquisition layer.

Downloads daily OHLCV bars for a universe of S&P 500 constituents from
Yahoo Finance via ``yfinance``, and caches every ticker as a Parquet file
so that repeated runs (and offline re-runs) do not re-hit the network.

Key design decisions
--------------------
* **Auto-adjusted prices**: splits and dividends are adjusted into the price
  series, so no manual corporate-action handling is required.
* **Cache-first strategy**: ``load_or_download`` serves cached Parquet files
  when they are fresh enough, and only falls back to the network otherwise.
* **Robust flattening**: ``yf.download`` returns MultiIndex columns when
  multiple tickers are requested; we normalise everything into a tidy
  long-format DataFrame (one row per ticker-date).
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column order for the long-format OHLCV frame
CANONICAL_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the (Price, Ticker) MultiIndex columns produced by yfinance.

    Parameters
    ----------
    df:
        Raw frame returned by ``yf.download`` (single or multi ticker).

    Returns
    -------
    pd.DataFrame
        Frame with simple string columns (lower-cased price fields).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Two possible layouts: (Price, Ticker) or (Ticker, Price)
        if df.columns.nlevels == 2:
            level0 = {str(x) for x in df.columns.get_level_values(0)}
            if {"Open", "High", "Low", "Close", "Volume"} & level0:
                df = df.stack("Ticker", future_stack=True)
                df.index = df.index.set_names(["Date", "Ticker"])
                df = df.reset_index()
            else:
                df.columns = df.columns.get_level_values(0)
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    df.columns = [str(c).lower() for c in df.columns]
    return df


def download_universe(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str],
    cache_dir: pathlib.Path,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download daily bars for all tickers with caching and retry logic.

    Each ticker is fetched individually and persisted to
    ``<cache_dir>/<TICKER>.parquet``. Tickers that fail after retries are
    skipped with a warning instead of aborting the whole pipeline.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols to download.
    start_date:
        ISO-format start date (inclusive), e.g. ``"2016-01-01"``.
    end_date:
        ISO-format end date, or ``None`` for the latest available session.
    cache_dir:
        Directory used for the Parquet cache.
    max_retries:
        Number of download attempts per ticker before giving up.

    Returns
    -------
    pd.DataFrame
        Long-format OHLCV frame with columns
        ``[ticker, date, open, high, low, close, volume]``.
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(tickers)
    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - environment guard
        raise ImportError(
            "yfinance is required for data download. Install it via "
            "`pip install yfinance` or run the pipeline on cached data."
        )

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker}.parquet"
        if cache_file.exists():
            logger.info("Cache hit for %s, skipping download.", ticker)
            frame = pd.read_parquet(cache_file)
            frames.append(frame)
            continue

        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if raw is None or raw.empty:
                    raise ValueError("empty payload returned")
                frame = _flatten_multiindex(raw)
                frame = _normalise_frame(frame, ticker)
                frame.to_parquet(cache_file, index=False)
                frames.append(frame)
                logger.info(
                    "Downloaded %s: %d rows (%s to %s)",
                    ticker, len(frame),
                    frame["date"].min().date(), frame["date"].max().date(),
                )
                break
            except Exception as exc:  # noqa: BLE001 - network layer is best-effort
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt, max_retries, ticker, exc,
                )
                time.sleep(2.0 * attempt)
        else:
            failed.append(ticker)

    if failed:
        logger.warning("Tickers skipped after retries: %s", ", ".join(failed))
    if not frames:
        raise RuntimeError(
            "No data could be downloaded. Check network access or place "
            "Parquet files in the cache directory."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined[CANONICAL_COLUMNS]


def _normalise_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise a single-ticker frame to the canonical schema.

    Ensures the ticker column is populated, the date column is timezone-naive
    datetime, numeric columns are float64 and the frame is date-sorted.

    Parameters
    ----------
    frame:
        Flattened frame for one ticker.
    ticker:
        Ticker symbol to stamp onto the frame.

    Returns
    -------
    pd.DataFrame
        Canonical single-ticker frame.
    """
    frame = frame.copy()
    if "ticker" not in frame.columns or frame["ticker"].isna().all():
        frame["ticker"] = ticker
    frame["ticker"] = frame["ticker"].astype(str).str.upper()

    date_col = frame["date"] if "date" in frame.columns else frame.index
    frame["date"] = pd.to_datetime(date_col).dt.tz_localize(None)

    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype("float64")
    else:
        frame["volume"] = float("nan")

    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return frame[CANONICAL_COLUMNS]


def load_or_download(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str],
    cache_dir: pathlib.Path,
) -> pd.DataFrame:
    """High-level entry point: serve from cache, else download.

    This is a convenience wrapper around :func:`download_universe` that makes
    the cache-first behaviour explicit to callers.

    Parameters
    ----------
    tickers:
        Ticker symbols to load.
    start_date:
        ISO start date for the requested window.
    end_date:
        ISO end date, or ``None`` for latest.
    cache_dir:
        Parquet cache directory.

    Returns
    -------
    pd.DataFrame
        Long-format OHLCV frame covering the requested universe and window.
    """
    return download_universe(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )
