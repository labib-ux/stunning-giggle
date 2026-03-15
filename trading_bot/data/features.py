"""Feature engineering utilities for OHLCV market data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from config import MODEL_SAVE_PATH

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "rsi",
    "macd_line",
    "macd_hist",
    "macd_signal",
    "bb_lower",
    "bb_mid",
    "bb_upper",
    "ema9",
    "ema21",
    "volume_sma20",
    "price_change_pct",
    "high_low_range_pct",
]

_REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and engineered features for OHLCV data.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        DataFrame enriched with engineered features and indicator columns.

    Raises:
        ValueError: If required OHLCV columns are missing or nulls remain after cleanup.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for feature computation.")
        return df.copy()

    missing = [col for col in _REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    data = df.copy()

    try:
        rsi = ta.momentum.RSIIndicator(data["close"], window=14).rsi().rename("rsi")

        macd_calc = ta.trend.MACD(
            data["close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )
        macd_line = macd_calc.macd().rename("macd_line")
        macd_signal = macd_calc.macd_signal().rename("macd_signal")
        macd_hist = macd_calc.macd_diff().rename("macd_hist")

        bb_calc = ta.volatility.BollingerBands(data["close"], window=20, window_dev=2)
        bb_upper = bb_calc.bollinger_hband().rename("bb_upper")
        bb_mid = bb_calc.bollinger_mavg().rename("bb_mid")
        bb_lower = bb_calc.bollinger_lband().rename("bb_lower")

        ema9 = ta.trend.EMAIndicator(data["close"], window=9).ema_indicator().rename("ema9")
        ema21 = (
            ta.trend.EMAIndicator(data["close"], window=21).ema_indicator().rename("ema21")
        )
        volume_sma20 = data["volume"].rolling(window=20).mean().rename("volume_sma20")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to compute indicators: %s", exc)
        raise

    data = data.join(
        [
            rsi,
            macd_line,
            macd_signal,
            macd_hist,
            bb_lower,
            bb_mid,
            bb_upper,
            ema9,
            ema21,
            volume_sma20,
        ]
    )

    data["price_change_pct"] = data["close"].pct_change()
    data["high_low_range_pct"] = (data["high"] - data["low"]) / data["low"]

    rename_map = {
        "RSI_14": "rsi",
        "MACD_12_26_9": "macd_line",
        "MACDh_12_26_9": "macd_hist",
        "MACDs_12_26_9": "macd_signal",
        "BBL_20_2.0": "bb_lower",
        "BBM_20_2.0": "bb_mid",
        "BBU_20_2.0": "bb_upper",
        "EMA_9": "ema9",
        "EMA_21": "ema21",
        "SMA_20": "volume_sma20",
    }
    data.rename(columns=rename_map, inplace=True)

    for col in ["BBB_20_2.0", "BBP_20_2.0"]:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    null_counts = data.isnull().sum()
    if null_counts.sum() != 0:
        remaining = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values remain after cleanup: {remaining}")

    return data


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scale feature columns using a MinMaxScaler.

    Args:
        df: DataFrame containing FEATURE_COLUMNS.
        scaler: Optional pre-fitted MinMaxScaler to use for transformation.

    Returns:
        Tuple of (DataFrame with scaled features, fitted scaler).

    Raises:
        ValueError: If required feature columns are missing.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for feature scaling.")
        return df.copy(), scaler if scaler is not None else MinMaxScaler()

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for scaling: {missing}")

    features = df[FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        logger.info("Fitted new MinMaxScaler for feature scaling.")
    else:
        scaled = scaler.transform(features)

    scaled_df = df.copy()
    scaled_df[FEATURE_COLUMNS] = scaled

    min_val = np.nanmin(scaled)
    max_val = np.nanmax(scaled)
    if min_val < 0.0 or max_val > 1.0:
        logger.warning(
            "Scaled features out of [0, 1] range: min=%s max=%s",
            min_val,
            max_val,
        )

    model_dir = Path(MODEL_SAVE_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "feature_scaler.pkl"
    try:
        dump(scaler, scaler_path)
        logger.info("Saved feature scaler to %s", scaler_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to save feature scaler: %s", exc)
        raise

    return scaled_df, scaler
