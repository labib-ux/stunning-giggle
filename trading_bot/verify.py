#!/usr/bin/env python3
"""
System verification script for the RL trading bot project.
Runs a series of checks and prints PASS/FAIL for each item.
"""

from __future__ import annotations

import sys
from typing import Any


def _print_result(passed: bool, message: str) -> None:
    prefix = "PASS —" if passed else "FAIL —"
    print(f"{prefix} {message}")


def _record_result(passed: bool, message: str, counts: dict[str, int]) -> None:
    counts["total"] += 1
    if passed:
        counts["passed"] += 1
    _print_result(passed, message)


def _try_import(label: str, importer, counts: dict[str, int]) -> tuple[bool, Any]:
    try:
        result = importer()
        _record_result(True, f"imported {label}", counts)
        return True, result
    except Exception as e:
        _record_result(False, f"{label}: {e}", counts)
        return False, None


def main() -> None:
    """
    Run all verification checks in sequence and print a summary.
    """
    counts = {"total": 0, "passed": 0}

    # CHECK 1 — Python version
    try:
        if sys.version_info >= (3, 11):
            _record_result(True, f"Python version OK ({sys.version.split()[0]})", counts)
            python_ok = True
        else:
            _record_result(False, f"Python 3.11+ required, found {sys.version.split()[0]}", counts)
            python_ok = False
    except Exception as e:
        _record_result(False, f"Python version check failed: {e}", counts)
        python_ok = False

    # CHECK 2 — All imports (reported individually)
    config_ok, config_values = _try_import(
        "config constants",
        lambda: __import__(
            "config",
            fromlist=[
                "ALPACA_API_KEY",
                "ALPACA_SECRET_KEY",
                "ALPACA_BASE_URL",
                "DB_PATH",
                "MODEL_SAVE_PATH",
                "INITIAL_CAPITAL",
                "LOG_LEVEL",
            ],
        ),
        counts,
    )

    features_ok, _ = _try_import(
        "data.features (FEATURE_COLUMNS, compute_features)",
        lambda: __import__("data.features", fromlist=["FEATURE_COLUMNS", "compute_features"]),
        counts,
    )

    fetcher_ok, _ = _try_import(
        "data.fetcher (fetch_historical_data)",
        lambda: __import__("data.fetcher", fromlist=["fetch_historical_data"]),
        counts,
    )

    env_ok, _ = _try_import(
        "environment.trading_env (TradingEnvironment)",
        lambda: __import__("environment.trading_env", fromlist=["TradingEnvironment"]),
        counts,
    )

    alpaca_ok, _ = _try_import(
        "adapters.alpaca_adapter (AlpacaAdapter)",
        lambda: __import__("adapters.alpaca_adapter", fromlist=["AlpacaAdapter"]),
        counts,
    )

    base_adapter_ok, _ = _try_import(
        "adapters.base_adapter (BaseAdapter)",
        lambda: __import__("adapters.base_adapter", fromlist=["BaseAdapter"]),
        counts,
    )

    risk_ok, _ = _try_import(
        "risk.manager (RiskManager)",
        lambda: __import__("risk.manager", fromlist=["RiskManager"]),
        counts,
    )

    db_logger_ok, _ = _try_import(
        "database.logger (init_db, log_trade, log_decision)",
        lambda: __import__("database.logger", fromlist=["init_db", "log_trade", "log_decision"]),
        counts,
    )

    agent_ok, _ = _try_import(
        "agent.trainer (TradingAgent)",
        lambda: __import__("agent.trainer", fromlist=["TradingAgent"]),
        counts,
    )

    policy_ok, _ = _try_import(
        "agent.policy (CustomFeatureExtractor)",
        lambda: __import__("agent.policy", fromlist=["CustomFeatureExtractor"]),
        counts,
    )

    scheduler_ok, _ = _try_import(
        "scheduler.jobs (run_trading_cycle)",
        lambda: __import__("scheduler.jobs", fromlist=["run_trading_cycle"]),
        counts,
    )

    # CHECK 3 — API keys present
    try:
        if config_ok:
            from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

            keys_ok = bool(ALPACA_API_KEY) and bool(ALPACA_SECRET_KEY)
            if keys_ok:
                _record_result(True, "Alpaca API keys are set", counts)
            else:
                _record_result(
                    False,
                    "Alpaca API keys not set. Open .env and add your keys.",
                    counts,
                )
        else:
            keys_ok = False
            _record_result(
                False,
                "Alpaca API key check skipped because config import failed",
                counts,
            )
    except Exception as e:
        keys_ok = False
        _record_result(False, f"Alpaca API key check failed: {e}", counts)

    # CHECK 4 — Database initialization
    try:
        if config_ok and db_logger_ok:
            from config import DB_PATH
            from database.logger import init_db

            init_db(DB_PATH)
            _record_result(True, "Database initialization succeeded", counts)
        else:
            _record_result(
                False,
                "Database initialization skipped because imports failed",
                counts,
            )
    except Exception as e:
        _record_result(False, f"Database initialization failed: {e}", counts)

    # CHECK 5 — Data pipeline (requires valid API keys)
    df = None
    df_features = None
    check5_ok = False
    try:
        if keys_ok:
            if alpaca_ok and fetcher_ok and features_ok:
                from adapters.alpaca_adapter import AlpacaAdapter
                from data.fetcher import fetch_historical_data
                from data.features import FEATURE_COLUMNS, compute_features

                adapter = AlpacaAdapter()
                df = fetch_historical_data(adapter, "BTC/USD", "1h", limit=100)
                if df is not None and not df.empty:
                    expected_cols = ["open", "high", "low", "close", "volume"]
                    if list(df.columns) == expected_cols:
                        _record_result(True, "Data pipeline fetch returned valid OHLCV", counts)
                        df_features = compute_features(df)
                        if df_features is not None and not df_features.empty:
                            missing = [c for c in FEATURE_COLUMNS if c not in df_features.columns]
                            if not missing:
                                _record_result(
                                    True,
                                    "compute_features produced all FEATURE_COLUMNS",
                                    counts,
                                )
                                if df_features.isnull().sum().sum() == 0:
                                    _record_result(True, "compute_features produced no NaNs", counts)
                                    check5_ok = True
                                else:
                                    _record_result(False, "compute_features output contains NaNs", counts)
                            else:
                                _record_result(
                                    False,
                                    f"compute_features missing columns: {missing}",
                                    counts,
                                )
                        else:
                            _record_result(False, "compute_features returned empty DataFrame", counts)
                    else:
                        _record_result(
                            False,
                            f"OHLCV columns mismatch. Got {list(df.columns)}",
                            counts,
                        )
                else:
                    _record_result(False, "fetch_historical_data returned empty DataFrame", counts)
            else:
                _record_result(
                    False,
                    "Data pipeline skipped because required imports failed",
                    counts,
                )
        else:
            _record_result(
                False,
                "Data pipeline skipped because Alpaca API keys are not set",
                counts,
            )
    except Exception as e:
        _record_result(False, f"Data pipeline check failed: {e}", counts)

    # CHECK 6 — Environment construction
    try:
        if check5_ok:
            import numpy as np
            from data.features import scale_features
            from environment.trading_env import TradingEnvironment

            df_scaled, _scaler = scale_features(df_features)
            env = TradingEnvironment(
                df=df_scaled,
                initial_capital=10000.0,
                lookback_window=30,
                render_mode=None,
            )
            obs, _info = env.reset()

            if obs is not None and obs.dtype == np.float32:
                if obs.shape == env.observation_space.shape:
                    _record_result(True, "Environment constructed and reset successfully", counts)
                else:
                    _record_result(
                        False,
                        f"Observation shape mismatch. Got {obs.shape}, expected {env.observation_space.shape}",
                        counts,
                    )
            else:
                _record_result(False, "Observation is None or wrong dtype", counts)
        else:
            _record_result(
                False,
                "Environment construction skipped because data pipeline failed",
                counts,
            )
    except Exception as e:
        _record_result(False, f"Environment construction failed: {e}", counts)

    # Summary
    print("==============================")
    print("VERIFICATION SUMMARY")
    print("==============================")
    print(f"{counts['passed']} of {counts['total']} checks passed")

    if counts["passed"] == counts["total"]:
        print("✅ System is ready. Run: python main.py --mode train")
    else:
        print("❌ Fix the failing checks above before running the system.")


if __name__ == "__main__":
    main()
