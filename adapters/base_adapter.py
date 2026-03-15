"""Abstract adapter interface for market data, portfolio, and execution providers."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

COLUMN_OPEN = "open"
COLUMN_HIGH = "high"
COLUMN_LOW = "low"
COLUMN_CLOSE = "close"
COLUMN_VOLUME = "volume"

OHLCV_COLUMNS = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME]

PORTFOLIO_KEY_CASH = "cash"
PORTFOLIO_KEY_POSITIONS = "positions"
PORTFOLIO_KEY_TOTAL_VALUE = "total_value"

DEFAULT_ORDER_TYPE = "market"


class BaseAdapter(ABC):
    """Define the standard adapter contract for market integrations."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV market data for a symbol.

        Args:
            symbol: Asset symbol or identifier.
            timeframe: Bar timeframe string (for example, 1m, 1h).
            limit: Number of bars to return.

        Returns:
            DataFrame with columns [open, high, low, close, volume].
        """
        raise NotImplementedError

    @abstractmethod
    def get_portfolio(self) -> dict[str, Any]:
        """Retrieve the portfolio summary.

        Returns:
            Dict containing exactly the keys: cash, positions, total_value.
        """
        raise NotImplementedError

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = DEFAULT_ORDER_TYPE,
    ) -> dict[str, Any]:
        """Submit an order to the execution venue.

        Args:
            symbol: Asset symbol or identifier.
            side: Order side (buy or sell).
            qty: Quantity to trade.
            order_type: Order type string.

        Returns:
            Dict representing the submitted order response.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get the latest trade price for a symbol.

        Args:
            symbol: Asset symbol or identifier.

        Returns:
            Latest price as a float.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the current account."""
        raise NotImplementedError

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if the market is open, otherwise False.
        """
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve current open positions or exposure records.

        Returns:
            List of position dictionaries.
        """
        raise NotImplementedError
