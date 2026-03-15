"""Historical data fetching utilities for the trading system."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_historical_data(
    adapter: Any,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch OHLCV data via the provided adapter.

    Args:
        adapter: Trading venue adapter implementing get_ohlcv.
        symbol: Trading symbol to fetch.
        timeframe: Candle timeframe string.
        limit: Number of rows to request.

    Returns:
        A validated OHLCV DataFrame or an empty DataFrame on failure.
    """
    try:
        df = adapter.get_ohlcv(symbol, timeframe, limit)
    except Exception as exc:  # pragma: no cover - defensive logging around external I/O
        logger.exception(
            "Failed to fetch OHLCV data for %s (%s, limit=%s): %s",
            symbol,
            timeframe,
            limit,
            exc,
        )
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        logger.warning(
            "Adapter returned non-DataFrame OHLCV payload for %s (%s).",
            symbol,
            timeframe,
        )
        return pd.DataFrame()

    columns_lower = {col.lower(): col for col in df.columns}
    missing = [col for col in _REQUIRED_OHLCV_COLUMNS if col not in columns_lower]
    if missing:
        logger.warning(
            "Missing required OHLCV columns for %s (%s): %s",
            symbol,
            timeframe,
            missing,
        )
        return pd.DataFrame()

    rename_map = {
        columns_lower[col]: col
        for col in _REQUIRED_OHLCV_COLUMNS
        if columns_lower[col] != col
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    return df
