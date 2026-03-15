"""
Adapters package — pluggable broker and venue integrations.

Implements the adapter pattern so the RL agent can trade on any
supported platform without changing core logic. Each adapter
implements BaseAdapter and can be swapped transparently.
"""

import logging

from adapters.base_adapter import BaseAdapter
from adapters.alpaca_adapter import AlpacaAdapter
from adapters.binance_adapter import BinanceAdapter

# PolymarketAdapter has an optional dependency on py_clob_client
# which requires web3 and Polygon wallet libraries.
# Wrap in try/except so a missing Polymarket installation does not
# prevent AlpacaAdapter and BinanceAdapter from loading correctly.
try:
    from adapters.polymarket_adapter import PolymarketAdapter
except ImportError as e:
    logging.getLogger(__name__).warning(
        "PolymarketAdapter could not be imported: %s. "
        "This is expected if py_clob_client is not installed. "
        "AlpacaAdapter and BinanceAdapter are unaffected.",
        e,
    )
    PolymarketAdapter = None  # type: ignore[assignment,misc]

__all__ = ["BaseAdapter", "AlpacaAdapter", "PolymarketAdapter", "BinanceAdapter"]
