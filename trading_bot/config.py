"""Central configuration module for the trading system.

All constants flow from this module to avoid hardcoded secrets or
magic numbers throughout the codebase.
"""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def _get_env(key: str, default: str = "", required: bool = False) -> str:
    """Retrieve an environment variable with optional requirement enforcement.

    Args:
        key: The environment variable name.
        default: Value to return if the key is not set.
        required: If True and the key is missing or empty, logs a warning.

    Returns:
        The value as a string.
    """
    value = os.getenv(key, default)
    if required and not value:
        logger.warning(
            "Required environment variable '%s' is not set. "
            "The system may fail when this key is needed. "
            "Set it in your .env file.",
            key,
        )
    return value


# ── Alpaca ───────────────────────────────────────────────────────
ALPACA_API_KEY: str = _get_env("ALPACA_API_KEY", required=True)
ALPACA_SECRET_KEY: str = _get_env("ALPACA_SECRET_KEY", required=True)
ALPACA_BASE_URL: str = _get_env(
    "ALPACA_BASE_URL",
    default="https://paper-api.alpaca.markets",
)

# ── Polymarket ───────────────────────────────────────────────────
POLYMARKET_API_KEY: str = _get_env("POLYMARKET_API_KEY", required=True)
POLYMARKET_PRIVATE_KEY: str = _get_env("POLYMARKET_PRIVATE_KEY", required=True)
POLYGON_RPC_URL: str = _get_env(
    "POLYGON_RPC_URL",
    default="https://polygon-rpc.com",
)

# ── Database ─────────────────────────────────────────────────────
DB_PATH: str = _get_env("DB_PATH", default="./trading_bot.db")

# ── Model Persistence ────────────────────────────────────────────
MODEL_SAVE_PATH: str = _get_env(
    "MODEL_SAVE_PATH",
    default="./models/ppo_agent.zip",
)

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL: str = _get_env("LOG_LEVEL", default="INFO")

# ── Trading Constants ────────────────────────────────────────────
INITIAL_CAPITAL: float = float(_get_env("INITIAL_CAPITAL", default="10000.0"))

MAX_POSITION_SIZE: float = float(_get_env("MAX_POSITION_SIZE", default="0.2"))

MAX_DAILY_DRAWDOWN: float = float(_get_env("MAX_DAILY_DRAWDOWN", default="0.05"))

# ── RL Environment Constants ─────────────────────────────────────
# These are not in .env — they are architectural constants that
# must match the trained model's observation space exactly.
# Changing these requires retraining the agent from scratch.
LOOKBACK_WINDOW: int = 30
COMMISSION_RATE: float = 0.001


def _validate_config() -> None:
    """Run lightweight validation on loaded config values.

    Logs warnings for suspicious values rather than raising exceptions.
    """
    if not 0.0 < MAX_POSITION_SIZE <= 1.0:
        logger.warning(
            "MAX_POSITION_SIZE=%.2f is outside the expected range "
            "(0.0, 1.0]. Position sizing may behave unexpectedly.",
            MAX_POSITION_SIZE,
        )

    if not 0.0 < MAX_DAILY_DRAWDOWN <= 1.0:
        logger.warning(
            "MAX_DAILY_DRAWDOWN=%.2f is outside the expected range "
            "(0.0, 1.0]. Circuit breaker may behave unexpectedly.",
            MAX_DAILY_DRAWDOWN,
        )

    if INITIAL_CAPITAL <= 0:
        logger.warning(
            "INITIAL_CAPITAL=%.2f must be positive. "
            "Position sizing will return 0 for all trades.",
            INITIAL_CAPITAL,
        )

    if LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger.warning(
            "LOG_LEVEL='%s' is not a valid Python logging level. "
            "Defaulting behaviour is unpredictable. "
            "Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
            LOG_LEVEL,
        )

    logger.debug("Configuration loaded and validated successfully.")


_validate_config()

# ── SECURITY REMINDER ────────────────────────────────────────────
# Ensure your .gitignore contains at minimum:
#   .env
#   *.db
#   /models/
#   /tensorboard_logs/
#   __pycache__/
#   *.pyc
# Never commit .env, model weights, or database files.
