"""Scheduled job definitions for live trading execution cycles."""

import os
import logging
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from data.fetcher import fetch_historical_data
from data.features import compute_features, scale_features, FEATURE_COLUMNS
from database.logger import log_trade, log_decision
from config import MODEL_SAVE_PATH, INITIAL_CAPITAL

logger = logging.getLogger(__name__)


def run_trading_cycle(
    adapter: object,
    agent: object,
    risk_manager: object,
    session_maker: object,
    symbol: str,
    timeframe: str,
    lookback_window: int = 30,
    initial_capital: float = INITIAL_CAPITAL,
    day_start_value: float = INITIAL_CAPITAL,
) -> None:
    """Run a single trading cycle from data fetch to trade execution.

    Args:
        adapter: Adapter instance used for market data and trading actions.
        agent: Trained agent used for action selection.
        risk_manager: Risk manager providing sizing and circuit breaker checks.
        session_maker: SQLAlchemy session factory for logging to the database.
        symbol: Trading symbol to execute on.
        timeframe: Candle timeframe string.
        lookback_window: Number of timesteps to feed into the model.
        initial_capital: Initial capital for normalization in observations.
        day_start_value: Portfolio value at the start of the trading day.
    """
    try:
        try:
            with open("bot_control.json", "r") as f:
                control = json.load(f)
            if control.get("paused", False):
                logger.info(
                    "Trading cycle skipped — bot is paused via dashboard."
                )
                return
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(
                "Could not read bot_control.json: %s — "
                "proceeding with trading cycle as normal.",
                exc,
            )

        df = fetch_historical_data(
            adapter,
            symbol,
            timeframe,
            limit=lookback_window + 50,
        )

        if df is None or df.empty:
            logger.error(
                "fetch_historical_data returned empty DataFrame "
                "for %s %s — aborting cycle.",
                symbol,
                timeframe,
            )
            return

        df = compute_features(df)

        if df is None or df.empty:
            logger.error(
                "compute_features returned empty DataFrame "
                "for %s — aborting cycle.",
                symbol,
            )
            return

        df = df.tail(lookback_window).copy()

        if len(df) < lookback_window:
            logger.error(
                "Insufficient rows after slicing: got %d, need %d "
                "for %s — aborting cycle.",
                len(df),
                lookback_window,
                symbol,
            )
            return

        scaler_path = os.path.join(
            os.path.dirname(MODEL_SAVE_PATH),
            "feature_scaler.pkl",
        )

        try:
            scaler = joblib.load(scaler_path)
            logger.debug("Feature scaler loaded from: %s", scaler_path)
        except FileNotFoundError:
            logger.critical(
                "Feature scaler not found at '%s'. "
                "You must complete a training run before live inference. "
                "The scaler is created by scale_features() in data/features.py.",
                scaler_path,
            )
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.critical(
                "Unexpected error loading feature scaler: %s",
                exc,
                exc_info=True,
            )
            return

        df, _ = scale_features(df, scaler=scaler)
        if df is None or df.empty:
            logger.error(
                "scale_features returned empty DataFrame for %s — aborting cycle.",
                symbol,
            )
            return

        port = adapter.get_portfolio()
        current_price = adapter.get_current_price(symbol)

        if current_price <= 0:
            logger.error(
                "Invalid current price %.4f for %s — aborting cycle.",
                current_price,
                symbol,
            )
            return

        if risk_manager.check_circuit_breaker(
            current_value=port["total_value"],
            day_start_value=day_start_value,
        ):
            logger.critical(
                "CIRCUIT BREAKER TRIGGERED — daily drawdown limit exceeded. "
                "Skipping trading cycle. "
                "Portfolio: $%.2f | Day start: $%.2f",
                port["total_value"],
                day_start_value,
            )
            return

        window_flat = df[FEATURE_COLUMNS].values.flatten()
        capital_ratio = port["cash"] / initial_capital

        position_qty = 0.0
        entry_price = 0.0
        position_held = 0.0

        for pos in port.get("positions", []):
            if pos.get("symbol") == symbol:
                position_qty = float(pos.get("qty", 0.0))
                entry_price = float(pos.get("avg_entry_price", current_price))
                position_held = 1.0 if position_qty > 0 else 0.0
                break

        unrealized_pnl_pct = (
            (current_price - entry_price) / entry_price
            if position_held > 0 and entry_price > 0
            else 0.0
        )

        obs = np.concatenate(
            [
                window_flat,
                np.array(
                    [capital_ratio, position_held, unrealized_pnl_pct],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        action = agent.predict(obs)

        logger.info(
            "Agent decision: %d (0=HOLD 1=BUY 2=SELL) | "
            "Symbol: %s | Price: $%.4f | "
            "Position held: %s | Capital: $%.2f",
            action,
            symbol,
            current_price,
            bool(position_held),
            port["cash"],
        )

        with session_maker() as session:
            log_decision(
                session,
                {
                    "state_hash": "live",
                    "action": action,
                    "confidence": 1.0,
                    "reward": None,
                },
            )

            if action == 1 and position_held == 0.0:
                size = risk_manager.calculate_position_size(
                    capital=port["cash"],
                    price=current_price,
                )
                if size > 0:
                    order = adapter.submit_order(
                        symbol=symbol,
                        side="buy",
                        qty=size,
                        order_type="market",
                    )
                    log_trade(
                        session,
                        {
                            "adapter": type(adapter).__name__,
                            "symbol": symbol,
                            "side": "buy",
                            "qty": size,
                            "price": current_price,
                            "pnl": None,
                            "portfolio_value": port["total_value"],
                        },
                    )
                    logger.info(
                        "BUY order submitted: %.6f %s @ $%.4f",
                        size,
                        symbol,
                        current_price,
                    )
                else:
                    logger.warning(
                        "BUY signal but calculated position size is 0 "
                        "— order skipped. Capital: $%.2f Price: $%.4f",
                        port["cash"],
                        current_price,
                    )

            elif action == 2 and position_held > 0.0:
                order = adapter.submit_order(
                    symbol=symbol,
                    side="sell",
                    qty=position_qty,
                    order_type="market",
                )
                realized_pnl = (current_price - entry_price) * position_qty
                log_trade(
                    session,
                    {
                        "adapter": type(adapter).__name__,
                        "symbol": symbol,
                        "side": "sell",
                        "qty": position_qty,
                        "price": current_price,
                        "pnl": realized_pnl,
                        "portfolio_value": port["total_value"],
                    },
                )
                logger.info(
                    "SELL order submitted: %.6f %s @ $%.4f | "
                    "Realized PnL: $%.4f",
                    position_qty,
                    symbol,
                    current_price,
                    realized_pnl,
                )

            elif action == 1 and position_held > 0.0:
                logger.debug(
                    "BUY signal ignored — already holding position in %s.",
                    symbol,
                )

            elif action == 2 and position_held == 0.0:
                logger.debug(
                    "SELL signal ignored — no position held in %s.",
                    symbol,
                )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("run_trading_cycle failed: %s", exc, exc_info=True)
        return
