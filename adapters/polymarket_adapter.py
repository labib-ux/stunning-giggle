"""Polymarket adapter implementation for market data and order execution."""

import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Any
from py_clob_client.client import ClobClient
from py_clob_client.credentials import ApiCreds
from config import POLYMARKET_API_KEY, POLYMARKET_PRIVATE_KEY
from adapters.base_adapter import BaseAdapter

POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
POLYGON_CHAIN_ID = 137
REQUEST_TIMEOUT = 10
MAX_TRADES_FETCH = 1000

COLUMN_OPEN = "open"
COLUMN_HIGH = "high"
COLUMN_LOW = "low"
COLUMN_CLOSE = "close"
COLUMN_VOLUME = "volume"
COLUMN_TIMESTAMP = "timestamp"
COLUMN_PRICE = "price"

OHLCV_COLUMNS = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME]

TIMEFRAME_1M = "1m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"
RULE_1M = "1min"
RULE_1H = "1h"
RULE_4H = "4h"
RULE_1D = "1D"
DEFAULT_TIMEFRAME_RULE = RULE_1H

TIMEFRAME_MAP = {
    TIMEFRAME_1M: RULE_1M,
    TIMEFRAME_1H: RULE_1H,
    TIMEFRAME_4H: RULE_4H,
    TIMEFRAME_1D: RULE_1D,
}

HISTORY_KEY = "history"
HISTORY_TIME_KEY = "t"
HISTORY_PRICE_KEY = "p"

TRADE_TIMESTAMP_KEY = "timestamp"
TRADE_PRICE_KEY = "price"
TRADE_SIZE_KEY = "size"

PORTFOLIO_KEY_CASH = "cash"
PORTFOLIO_KEY_POSITIONS = "positions"
PORTFOLIO_KEY_TOTAL_VALUE = "total_value"

POSITION_KEY_CONDITION_ID = "condition_id"
POSITION_KEY_SIDE = "side"
POSITION_KEY_SIZE = "size"
POSITION_KEY_PRICE = "price"

MARKET_KEY_CONDITION_ID = "conditionId"
MARKET_KEY_QUESTION = "question"
MARKET_KEY_END_DATE = "endDate"
MARKET_KEY_BEST_ASK = "bestAsk"
MARKET_KEY_VOLUME_24H = "volume24hr"
MARKETS_ENDPOINT = "markets"

ORDER_KEY_MARKET = "market"
ORDER_KEY_CONDITION_ID = "condition_id"

ACTIVE_PARAM_KEY = "active"
CLOSED_PARAM_KEY = "closed"
LIMIT_PARAM_KEY = "limit"
ACTIVE_PARAM_VALUE = "true"
CLOSED_PARAM_VALUE = "false"
ACTIVE_MARKETS_LIMIT = 50

SIDE_BUY = "buy"
SIDE_SELL = "sell"
DEFAULT_ORDER_TYPE = "market"

PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
ZERO_FLOAT = 0.0
EMPTY_STRING = ""
USDC_WEI_MULTIPLIER = 1_000_000
DTYPE_FLOAT64 = "float64"
BALANCE_KEY = "balance"
API_PASSPHRASE = ""

OUTPUT_KEY_CONDITION_ID = "condition_id"
OUTPUT_KEY_QUESTION = "question"
OUTPUT_KEY_END_DATE_ISO = "end_date_iso"
OUTPUT_KEY_CURRENT_YES_PRICE = "current_yes_price"
OUTPUT_KEY_VOLUME_24H = "volume_24h"

MSG_UNRECOGNIZED_TIMEFRAME = "Unrecognized timeframe '%s', defaulting to %s"
MSG_GET_OHLCV_FAILED = "get_ohlcv failed for symbol %s"
MSG_GET_PORTFOLIO_FAILED = "get_portfolio failed"
MSG_SUBMIT_ORDER_FAILED = "submit_order failed for symbol %s"
MSG_CANCEL_ALL_FAILED = "cancel_all_orders failed"
MSG_MARKET_OPEN = "Polymarket operates 24/7; returning True."
MSG_GET_POSITIONS_FAILED = "get_positions failed"
MSG_EMPTY_ORDER_BOOK = "get_market_odds: empty order book asks for %s"
MSG_GET_MARKET_ODDS_FAILED = "get_market_odds failed for %s"
MSG_GET_ACTIVE_MARKETS_FAILED = "get_active_markets failed"
MSG_CANCEL_ALL_RESULT = "cancel_all_orders result: %s"


def _empty_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ZERO_FLOAT


class PolymarketAdapter(BaseAdapter):
    """Adapter for Polymarket CLOB and Gamma APIs."""

    def __init__(self) -> None:
        """Initialize the Polymarket CLOB client."""
        creds = ApiCreds(
            api_key=POLYMARKET_API_KEY,
            api_secret=POLYMARKET_PRIVATE_KEY,
            api_passphrase=API_PASSPHRASE,
        )
        self.client = ClobClient(
            host=POLYMARKET_HOST,
            chain_id=POLYGON_CHAIN_ID,
            creds=creds,
        )
        self.logger = logging.getLogger(__name__)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV data by combining Gamma price history and CLOB volume.

        Args:
            symbol: Polymarket market identifier.
            timeframe: Timeframe string such as 1m, 1h, 4h, or 1d.
            limit: Number of bars to return.

        Returns:
            DataFrame containing OHLCV bars.
        """
        try:
            rule = TIMEFRAME_MAP.get(timeframe, DEFAULT_TIMEFRAME_RULE)
            if timeframe not in TIMEFRAME_MAP:
                self.logger.warning(
                    MSG_UNRECOGNIZED_TIMEFRAME,
                    timeframe,
                    DEFAULT_TIMEFRAME_RULE,
                )

            url = f"{GAMMA_API_BASE}/{MARKETS_ENDPOINT}/{symbol}"
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            history = response.json().get(HISTORY_KEY, [])

            if history:
                timestamps = [
                    datetime.fromtimestamp(item[HISTORY_TIME_KEY], tz=timezone.utc)
                    for item in history
                ]
                prices = [float(item[HISTORY_PRICE_KEY]) for item in history]
                price_df = pd.DataFrame(
                    {COLUMN_TIMESTAMP: timestamps, COLUMN_PRICE: prices}
                ).set_index(COLUMN_TIMESTAMP)
            else:
                price_df = pd.DataFrame(columns=[COLUMN_TIMESTAMP, COLUMN_PRICE]).set_index(
                    COLUMN_TIMESTAMP
                )

            if price_df.empty:
                price_ohlc = pd.DataFrame(columns=OHLCV_COLUMNS[:-1])
            else:
                price_ohlc = price_df[COLUMN_PRICE].resample(rule).ohlc()
                price_ohlc = price_ohlc.ffill()

            trades = self.client.get_trades(
                market=symbol,
                limit=MAX_TRADES_FETCH,
            )
            trades_df = pd.DataFrame(trades)
            if trades_df.empty:
                volume_sum = pd.Series(dtype=DTYPE_FLOAT64, name=COLUMN_VOLUME)
            else:
                trades_df[COLUMN_TIMESTAMP] = pd.to_datetime(
                    trades_df[TRADE_TIMESTAMP_KEY],
                    utc=True,
                )
                trades_df[TRADE_SIZE_KEY] = trades_df[TRADE_SIZE_KEY].astype(
                    DTYPE_FLOAT64
                )
                trades_df = trades_df.set_index(COLUMN_TIMESTAMP)
                volume_sum = (
                    trades_df[TRADE_SIZE_KEY]
                    .resample(rule)
                    .sum()
                    .rename(COLUMN_VOLUME)
                    .fillna(ZERO_FLOAT)
                )

            final_df = pd.concat([price_ohlc, volume_sum], axis=1)
            final_df = final_df.astype(DTYPE_FLOAT64)
            return final_df.tail(limit)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_OHLCV_FAILED, symbol, exc_info=True)
            return _empty_ohlcv_df()

    def get_portfolio(self) -> dict[str, Any]:
        """Retrieve portfolio cash and open position value.

        Returns:
            Dict containing cash, positions, and total value.
        """
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            balance_response = self.client.get_balance_allowance(params)
            raw_balance = balance_response.get(BALANCE_KEY, ZERO_FLOAT)
            cash_balance = _safe_float(raw_balance) / USDC_WEI_MULTIPLIER

            positions = self.get_positions()
            positions_value = sum(
                _safe_float(pos.get(POSITION_KEY_SIZE, ZERO_FLOAT))
                * _safe_float(pos.get(POSITION_KEY_PRICE, ZERO_FLOAT))
                for pos in positions
            )

            return {
                PORTFOLIO_KEY_CASH: cash_balance,
                PORTFOLIO_KEY_POSITIONS: positions,
                PORTFOLIO_KEY_TOTAL_VALUE: cash_balance + positions_value,
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
        order_type: str = DEFAULT_ORDER_TYPE,
    ) -> dict[str, Any]:
        """Submit a limit order at the current market price.

        Args:
            symbol: Polymarket market identifier.
            side: Order side (buy or sell).
            qty: Quantity to trade.
            order_type: Order type string.

        Returns:
            Dict representing the submitted order response.
        """
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            market_price = self.get_market_odds(symbol)
            if market_price <= ZERO_FLOAT:
                return {}

            side_value = side.lower()
            if side_value == SIDE_BUY:
                side_enum = BUY
            elif side_value == SIDE_SELL:
                side_enum = SELL
            else:
                return {}

            order_args = OrderArgs(
                token_id=symbol,
                price=market_price,
                size=qty,
                side=side_enum,
            )
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)
            return response if isinstance(response, dict) else {}
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_SUBMIT_ORDER_FAILED, symbol, exc_info=True)
            return {}

    def get_current_price(self, symbol: str) -> float:
        """Fetch the current market price for a condition.

        Args:
            symbol: Polymarket condition identifier.

        Returns:
            Latest price as a float.
        """
        try:
            return float(self.get_market_odds(symbol))
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_MARKET_ODDS_FAILED, symbol, exc_info=True)
            return ZERO_FLOAT

    def cancel_all_orders(self) -> None:
        """Cancel all open Polymarket orders."""
        try:
            result = self.client.cancel_all()
            self.logger.info(MSG_CANCEL_ALL_RESULT, result)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_CANCEL_ALL_FAILED, exc_info=True)

    def is_market_open(self) -> bool:
        """Check if the Polymarket CLOB is open.

        Returns:
            True because Polymarket operates continuously.
        """
        self.logger.debug(MSG_MARKET_OPEN)
        return True

    def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve open orders as position-like exposure records.

        Returns:
            List of position dictionaries.
        """
        try:
            orders = self.client.get_orders()
            results: list[dict[str, Any]] = []
            for order in orders:
                condition_id = order.get(
                    ORDER_KEY_CONDITION_ID,
                    order.get(
                        MARKET_KEY_CONDITION_ID,
                        order.get(ORDER_KEY_MARKET, EMPTY_STRING),
                    ),
                )
                results.append(
                    {
                        POSITION_KEY_CONDITION_ID: condition_id,
                        POSITION_KEY_SIDE: order.get(POSITION_KEY_SIDE, EMPTY_STRING),
                        POSITION_KEY_SIZE: _safe_float(
                            order.get(POSITION_KEY_SIZE, ZERO_FLOAT)
                        ),
                        POSITION_KEY_PRICE: _safe_float(
                            order.get(POSITION_KEY_PRICE, ZERO_FLOAT)
                        ),
                    }
                )
            return results
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_POSITIONS_FAILED, exc_info=True)
            return []

    def get_market_odds(self, condition_id: str) -> float:
        """Fetch the best ask price as the current market odds.

        Args:
            condition_id: Polymarket condition identifier.

        Returns:
            Best ask price clamped to [0.0, 1.0].
        """
        try:
            order_book = self.client.get_order_book(condition_id)
            if not order_book.asks:
                self.logger.warning(MSG_EMPTY_ORDER_BOOK, condition_id)
                return ZERO_FLOAT

            best_ask = min(float(ask.price) for ask in order_book.asks)
            clamped = max(PROBABILITY_MIN, min(PROBABILITY_MAX, best_ask))
            return float(clamped)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_MARKET_ODDS_FAILED, condition_id, exc_info=True)
            return ZERO_FLOAT

    def get_active_markets(self, keyword: str) -> list[dict[str, Any]]:
        """Search active markets that match a keyword.

        Args:
            keyword: Keyword filter applied to the market question.

        Returns:
            List of matching market summaries.
        """
        try:
            url = f"{GAMMA_API_BASE}/{MARKETS_ENDPOINT}"
            params = {
                ACTIVE_PARAM_KEY: ACTIVE_PARAM_VALUE,
                CLOSED_PARAM_KEY: CLOSED_PARAM_VALUE,
                LIMIT_PARAM_KEY: ACTIVE_MARKETS_LIMIT,
            }
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            markets = response.json()

            keyword_lower = keyword.lower()
            results: list[dict[str, Any]] = []
            for market in markets:
                question = market.get(MARKET_KEY_QUESTION, EMPTY_STRING)
                if keyword_lower not in question.lower():
                    continue
                results.append(
                    {
                        OUTPUT_KEY_CONDITION_ID: market.get(
                            MARKET_KEY_CONDITION_ID,
                            EMPTY_STRING,
                        ),
                        OUTPUT_KEY_QUESTION: question,
                        OUTPUT_KEY_END_DATE_ISO: market.get(
                            MARKET_KEY_END_DATE,
                            EMPTY_STRING,
                        ),
                        OUTPUT_KEY_CURRENT_YES_PRICE: _safe_float(
                            market.get(MARKET_KEY_BEST_ASK, ZERO_FLOAT)
                        ),
                        OUTPUT_KEY_VOLUME_24H: _safe_float(
                            market.get(MARKET_KEY_VOLUME_24H, ZERO_FLOAT)
                        ),
                    }
                )
            return results
        except Exception:  # pragma: no cover - defensive logging
            self.logger.error(MSG_GET_ACTIVE_MARKETS_FAILED, exc_info=True)
            return []
