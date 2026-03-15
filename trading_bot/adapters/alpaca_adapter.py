"""Alpaca adapter implementation for trading and market data access."""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from adapters.base_adapter import BaseAdapter

COLUMN_OPEN = "open"
COLUMN_HIGH = "high"
COLUMN_LOW = "low"
COLUMN_CLOSE = "close"
COLUMN_VOLUME = "volume"
COLUMN_SYMBOL = "symbol"
COLUMN_TIMESTAMP = "timestamp"

OHLCV_COLUMNS = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME]

PORTFOLIO_KEY_CASH = "cash"
PORTFOLIO_KEY_POSITIONS = "positions"
PORTFOLIO_KEY_TOTAL_VALUE = "total_value"

POSITION_KEY_SYMBOL = "symbol"
POSITION_KEY_QTY = "qty"
POSITION_KEY_MARKET_VALUE = "market_value"
POSITION_KEY_AVG_ENTRY_PRICE = "avg_entry_price"

ATTR_SYMBOL = "symbol"
ATTR_QTY = "qty"
ATTR_MARKET_VALUE = "market_value"
ATTR_AVG_ENTRY_PRICE = "avg_entry_price"

SIDE_BUY = "buy"
SIDE_SELL = "sell"
ORDER_TYPE_MARKET = "market"
PAPER_ENV_TOKEN = "paper"

TIMEFRAME_1M = "1m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"
DEFAULT_TIMEFRAME_STR = TIMEFRAME_1H
DEFAULT_TIMEFRAME = TimeFrame.Hour

MINUTES_PER_1M = 1
MINUTES_PER_15M = 15
HOURS_PER_1H = 1
HOURS_PER_4H = 4
DAYS_PER_1D = 1

CURRENT_PRICE_LOOKBACK_MINUTES = 5

ZERO_FLOAT = 0.0
EMPTY_STRING = ""
DROP_ERRORS_IGNORE = "ignore"

TIMEFRAME_MAP = {
    TIMEFRAME_1M: TimeFrame.Minute,
    TIMEFRAME_15M: TimeFrame(MINUTES_PER_15M, TimeFrameUnit.Minute),
    TIMEFRAME_1H: TimeFrame.Hour,
    TIMEFRAME_4H: TimeFrame(HOURS_PER_4H, TimeFrameUnit.Hour),
    TIMEFRAME_1D: TimeFrame.Day,
}

SIDE_ENUM_MAP = {
    SIDE_BUY: OrderSide.BUY,
    SIDE_SELL: OrderSide.SELL,
}

TIME_IN_FORCE = TimeInForce.GTC

MSG_UNRECOGNIZED_TIMEFRAME = "Unrecognized timeframe '%s', defaulting to %s"
MSG_GET_OHLCV_FAILED = "get_ohlcv failed for symbol %s"
MSG_GET_CURRENT_PRICE_EMPTY = "get_current_price: no bars returned for symbol %s"
MSG_GET_CURRENT_PRICE_FAILED = "get_current_price failed for symbol %s"
MSG_GET_PORTFOLIO_FAILED = "get_portfolio failed"
MSG_INVALID_SIDE = "submit_order received invalid side '%s'"
MSG_SUBMIT_ORDER_FAILED = "submit_order failed for symbol %s"
MSG_CANCEL_ORDERS_RESULT = "cancel_all_orders result: %s"
MSG_CANCEL_ORDERS_FAILED = "cancel_all_orders failed"
MSG_MARKET_OPEN = "Crypto markets never close; returning True."
MSG_GET_POSITIONS_FAILED = "get_positions failed"


def _empty_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ZERO_FLOAT


class AlpacaAdapter(BaseAdapter):
    """Adapter for Alpaca trading and crypto market data APIs."""

    def __init__(self) -> None:
        """Initialize Alpaca trading and data clients."""
        paper = PAPER_ENV_TOKEN in ALPACA_BASE_URL.lower()
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=paper,
        )
        self.data_client = CryptoHistoricalDataClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
        )
        self.logger = logging.getLogger(__name__)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol.

        Args:
            symbol: Asset symbol.
            timeframe: Timeframe string such as 1m, 1h, 4h, or 1d.
            limit: Number of bars to return.

        Returns:
            DataFrame containing OHLCV bars.
        """
        try:
            if timeframe not in TIMEFRAME_MAP:
                self.logger.warning(
                    MSG_UNRECOGNIZED_TIMEFRAME,
                    timeframe,
                    DEFAULT_TIMEFRAME_STR,
                )
            tf = TIMEFRAME_MAP.get(timeframe, DEFAULT_TIMEFRAME)

            now = datetime.utcnow()
            if timeframe == TIMEFRAME_1M:
                start_time = now - timedelta(minutes=limit * MINUTES_PER_1M)
            elif timeframe == TIMEFRAME_15M:
                start_time = now - timedelta(minutes=limit * MINUTES_PER_15M)
            elif timeframe == TIMEFRAME_1H:
                start_time = now - timedelta(hours=limit * HOURS_PER_1H)
            elif timeframe == TIMEFRAME_4H:
                start_time = now - timedelta(hours=limit * HOURS_PER_4H)
            elif timeframe == TIMEFRAME_1D:
                start_time = now - timedelta(days=limit * DAYS_PER_1D)
            else:
                start_time = now - timedelta(hours=limit * HOURS_PER_1H)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_time,
            )
            bars = self.data_client.get_crypto_bars(request)
            df = bars.df
            df = df.reset_index()
            df = df.drop(columns=[COLUMN_SYMBOL], errors=DROP_ERRORS_IGNORE)
            df = df.set_index(COLUMN_TIMESTAMP)
            df = df[OHLCV_COLUMNS].astype(float)
            return df.tail(limit)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_OHLCV_FAILED, symbol, exc_info=True)
            return _empty_ohlcv_df()

    def get_current_price(self, symbol: str) -> float:
        """Fetch the latest close price for a symbol.

        Args:
            symbol: Asset symbol.

        Returns:
            Latest close price as a float.
        """
        try:
            start_time = datetime.utcnow() - timedelta(
                minutes=CURRENT_PRICE_LOOKBACK_MINUTES
            )
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
            )
            bars = self.data_client.get_crypto_bars(request)
            df = bars.df.reset_index()

            if df.empty:
                self.logger.warning(MSG_GET_CURRENT_PRICE_EMPTY, symbol)
                return ZERO_FLOAT

            return float(df[COLUMN_CLOSE].iloc[-1])
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_CURRENT_PRICE_FAILED, symbol, exc_info=True)
            return ZERO_FLOAT

    def get_portfolio(self) -> dict[str, Any]:
        """Retrieve account cash and total portfolio value.

        Returns:
            Dict containing cash, positions, and total value.
        """
        try:
            account = self.trading_client.get_account()
            return {
                PORTFOLIO_KEY_CASH: float(account.cash),
                PORTFOLIO_KEY_POSITIONS: self.get_positions(),
                PORTFOLIO_KEY_TOTAL_VALUE: float(account.portfolio_value),
            }
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_PORTFOLIO_FAILED, exc_info=True)
            return {
                PORTFOLIO_KEY_CASH: ZERO_FLOAT,
                PORTFOLIO_KEY_POSITIONS: [],
                PORTFOLIO_KEY_TOTAL_VALUE: ZERO_FLOAT,
            }

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = ORDER_TYPE_MARKET,
    ) -> dict[str, Any]:
        """Submit a market order through Alpaca.

        Args:
            symbol: Asset symbol.
            side: Order side (buy or sell).
            qty: Quantity to trade.
            order_type: Order type string.

        Returns:
            Dict representing the submitted order.
        """
        try:
            side_enum = SIDE_ENUM_MAP.get(side.lower())
            if side_enum is None:
                self.logger.error(MSG_INVALID_SIDE, side)
                return {}

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=TIME_IN_FORCE,
            )
            order = self.trading_client.submit_order(request)
            return order.model_dump()
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_SUBMIT_ORDER_FAILED, symbol, exc_info=True)
            return {}

    def cancel_all_orders(self) -> None:
        """Cancel all open orders on Alpaca."""
        try:
            result = self.trading_client.cancel_orders()
            self.logger.info(MSG_CANCEL_ORDERS_RESULT, result)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_CANCEL_ORDERS_FAILED, exc_info=True)

    def is_market_open(self) -> bool:
        """Check if the crypto market is open.

        Returns:
            True because crypto markets operate continuously.
        """
        self.logger.debug(MSG_MARKET_OPEN)
        return True

    def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve current positions from Alpaca.

        Returns:
            List of position dictionaries.
        """
        try:
            positions = self.trading_client.get_all_positions()
            results: list[dict[str, Any]] = []
            for position in positions:
                results.append(
                    {
                        POSITION_KEY_SYMBOL: getattr(position, ATTR_SYMBOL, EMPTY_STRING),
                        POSITION_KEY_QTY: _safe_float(
                            getattr(position, ATTR_QTY, ZERO_FLOAT)
                        ),
                        POSITION_KEY_MARKET_VALUE: _safe_float(
                            getattr(position, ATTR_MARKET_VALUE, ZERO_FLOAT)
                        ),
                        POSITION_KEY_AVG_ENTRY_PRICE: _safe_float(
                            getattr(position, ATTR_AVG_ENTRY_PRICE, ZERO_FLOAT)
                        ),
                    }
                )
            return results
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_POSITIONS_FAILED, exc_info=True)
            return []
