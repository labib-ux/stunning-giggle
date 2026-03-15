"""Risk management logic for position sizing and trading safeguards."""

import logging


class RiskManager:
    """Encapsulate position sizing and risk control calculations."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.2,
        max_daily_drawdown: float = 0.05,
    ) -> None:
        """Initialize the risk manager with portfolio constraints.

        Args:
            initial_capital: Starting capital for baseline sizing.
            max_position_pct: Maximum fraction of capital per position.
            max_daily_drawdown: Maximum allowable daily drawdown before halting.
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_drawdown = max_daily_drawdown
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        confidence: float = 1.0,
    ) -> float:
        """Compute position size using fixed fractional sizing.

        Args:
            capital: Available capital for sizing.
            price: Current asset price.
            confidence: Confidence multiplier for the position.

        Returns:
            Position size as a float.
        """
        if price <= 0:
            self.logger.warning(
                "calculate_position_size received non-positive price: %s",
                price,
            )
            return 0.0

        size = (capital * self.max_position_pct * confidence) / price
        return float(max(0.0, size))

    def kelly_size(
        self,
        win_prob: float,
        win_return: float,
        loss_return: float,
        capital: float,
        price: float,
    ) -> float:
        """Compute position size using a half-Kelly criterion.

        Args:
            win_prob: Estimated probability of winning.
            win_return: Expected return on win (as a multiple).
            loss_return: Expected return on loss (as a multiple).
            capital: Available capital for sizing.
            price: Current asset price.

        Returns:
            Position size as a float.
        """
        if win_return <= 0 or price <= 0:
            self.logger.warning(
                "kelly_size received invalid win_return or price: win_return=%s price=%s",
                win_return,
                price,
            )
            return 0.0

        kelly = (win_prob * win_return - (1 - win_prob) * loss_return) / win_return
        half_kelly = max(0.0, min(1.0, kelly * 0.5))
        size = (capital * half_kelly) / price
        return float(max(0.0, size))

    def check_circuit_breaker(
        self,
        current_value: float,
        day_start_value: float,
    ) -> bool:
        """Determine if trading should halt due to daily drawdown.

        Args:
            current_value: Current portfolio value.
            day_start_value: Portfolio value at start of day.

        Returns:
            True if trading should halt, otherwise False.
        """
        if day_start_value <= 0:
            self.logger.warning(
                "check_circuit_breaker received non-positive day_start_value: %s",
                day_start_value,
            )
            return False

        drawdown = (day_start_value - current_value) / day_start_value
        if drawdown > self.max_daily_drawdown:
            self.logger.critical(
                "Circuit breaker triggered: drawdown=%.2f%% threshold=%.2f%%",
                drawdown * 100,
                self.max_daily_drawdown * 100,
            )
            return True
        return False

    def should_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        stop_pct: float = 0.02,
    ) -> bool:
        """Evaluate whether a stop-loss condition has been met.

        Args:
            entry_price: Entry price for the position.
            current_price: Current market price.
            side: Position side (long or short).
            stop_pct: Stop-loss percentage threshold.

        Returns:
            True if stop-loss condition is met, otherwise False.
        """
        if side == "long":
            return current_price < entry_price * (1 - stop_pct)
        if side == "short":
            return current_price > entry_price * (1 + stop_pct)

        self.logger.warning("should_stop_loss received unrecognized side: %s", side)
        return False
