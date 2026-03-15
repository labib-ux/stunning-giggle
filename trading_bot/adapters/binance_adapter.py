"""Binance adapter stub implementation."""

from typing import Any

import pandas as pd

from adapters.base_adapter import BaseAdapter

DEFAULT_ORDER_TYPE = "market"


class BinanceAdapter(BaseAdapter):
    """Stub adapter for Binance trading and market data."""

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol.

        Args:
            symbol: Asset symbol.
            timeframe: Timeframe string.
            limit: Number of bars to return.

        Returns:
            DataFrame containing OHLCV bars.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_portfolio(self) -> dict[str, Any]:
        """Retrieve the portfolio summary.

        Returns:
            Dict containing cash, positions, and total value.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = DEFAULT_ORDER_TYPE,
    ) -> dict[str, Any]:
        """Submit an order to Binance.

        Args:
            symbol: Asset symbol.
            side: Order side.
            qty: Quantity to trade.
            order_type: Order type string.

        Returns:
            Dict representing the submitted order.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_current_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol.

        Args:
            symbol: Asset symbol.

        Returns:
            Latest price as a float.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def cancel_all_orders(self) -> None:
        """Cancel all open Binance orders."""
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def is_market_open(self) -> bool:
        """Check if the Binance market is open.

        Returns:
            True if the market is open, otherwise False.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")

    def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve open positions from Binance.

        Returns:
            List of position dictionaries.
        """
        # TODO: Implement using ccxt library — ccxt.binance().method_name()
        raise NotImplementedError("Binance adapter not yet implemented")
