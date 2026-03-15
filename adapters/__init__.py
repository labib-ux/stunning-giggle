"""Pluggable adapter pattern interfaces and implementations for trading integrations."""

from .alpaca_adapter import AlpacaAdapter
from .base_adapter import BaseAdapter
from .binance_adapter import BinanceAdapter
from .polymarket_adapter import PolymarketAdapter

__all__ = ["BaseAdapter", "AlpacaAdapter", "PolymarketAdapter", "BinanceAdapter"]
