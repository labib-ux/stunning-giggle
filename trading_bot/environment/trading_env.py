"""Gymnasium-compatible trading environment for reinforcement learning."""

import gymnasium as gym
import numpy as np
import pandas as pd
import logging

from data.features import FEATURE_COLUMNS

COMMISSION_RATE = 0.001
DRAWDOWN_THRESHOLD = 0.10
DRAWDOWN_MULTIPLIER = 0.1
SHARPE_WINDOW = 50
RUIN_THRESHOLD = 0.10


class TradingEnvironment(gym.Env):
    """Simulated trading environment with portfolio-based rewards."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        lookback_window: int = 30,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the trading environment.

        Args:
            df: Input DataFrame containing market data and engineered features.
            initial_capital: Starting capital for the simulated portfolio.
            lookback_window: Number of timesteps to include in each observation.
            render_mode: Optional render mode for Gymnasium compatibility.

        Raises:
            ValueError: If any feature columns contain NaN or infinite values.
        """
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.render_mode = render_mode
        self.feature_columns = FEATURE_COLUMNS
        n_features = len(FEATURE_COLUMNS)

        feature_data = self.df[self.feature_columns]
        bad_columns: list[str] = []
        for column in self.feature_columns:
            series = feature_data[column]
            if series.isna().any():
                bad_columns.append(column)
                continue
            values = series.to_numpy(dtype=float, copy=False)
            if np.isinf(values).any():
                bad_columns.append(column)
        if bad_columns:
            raise ValueError(
                "Feature columns contain NaN or infinite values: "
                f"{bad_columns}"
            )

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window * n_features + 3,),
            dtype=np.float32,
        )

        self.logger = logging.getLogger(__name__)

        self.current_capital: float = initial_capital
        self.position_size: float = 0.0
        self.entry_price: float = 0.0
        self.trade_count: int = 0
        self.current_step: int = lookback_window
        self.peak_value: float = initial_capital
        self.portfolio_value: float = initial_capital

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment state.

        Args:
            seed: Optional random seed for environment reproducibility.
            options: Optional reset options for Gymnasium compatibility.

        Returns:
            Tuple of the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        self.current_capital = self.initial_capital
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.current_step = self.lookback_window
        self.peak_value = self.initial_capital
        self.portfolio_value = self.initial_capital

        observation = self._get_observation()
        return observation.astype(np.float32), {"portfolio_value": self.portfolio_value}

    def _get_observation(self) -> np.ndarray:
        """Construct the current observation vector.

        Returns:
            Numpy array containing the flattened feature window and account state.

        Raises:
            AssertionError: If the observation contains NaN or infinite values.
        """
        safe_step = min(self.current_step, len(self.df) - 1)
        window = self.df[self.feature_columns].iloc[
            max(0, safe_step - self.lookback_window) : safe_step
        ].values.flatten()

        capital_ratio = self.current_capital / self.initial_capital
        position_held = 1.0 if self.position_size > 0 else 0.0
        if self.position_size > 0:
            current_price = float(self.df["close"].iloc[safe_step])
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0

        obs = np.concatenate(
            [window, [capital_ratio, position_held, unrealized_pnl_pct]]
        )

        nan_mask = np.isnan(obs)
        inf_mask = np.isinf(obs)
        if nan_mask.any() or inf_mask.any():
            nan_indices = np.where(nan_mask)[0].tolist()
            inf_indices = np.where(inf_mask)[0].tolist()
            raise AssertionError(
                "Observation contains invalid values. "
                f"NaN indices: {nan_indices} Inf indices: {inf_indices}"
            )

        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply an action and advance the environment by one timestep.

        Args:
            action: Integer representing the trading action.

        Returns:
            Tuple containing observation, reward, terminated, truncated, and info.
        """
        current_price = float(self.df["close"].iloc[self.current_step])

        if action == 1 and self.position_size == 0:
            self.position_size = (
                self.current_capital * (1 - COMMISSION_RATE)
            ) / current_price
            self.entry_price = current_price
            self.current_capital = 0.0
            step_reward = -COMMISSION_RATE
        elif action == 2 and self.position_size > 0:
            sale_value = self.position_size * current_price * (1 - COMMISSION_RATE)
            realized_pnl = sale_value - (self.position_size * self.entry_price)
            self.current_capital = sale_value
            step_reward = realized_pnl / self.initial_capital
            self.position_size = 0.0
            self.entry_price = 0.0
            self.trade_count += 1
        else:
            step_reward = 0.0

        self.portfolio_value = self.current_capital + (
            self.position_size * current_price
        )
        self.peak_value = max(self.peak_value, self.portfolio_value)

        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if drawdown > DRAWDOWN_THRESHOLD:
            step_reward -= drawdown * DRAWDOWN_MULTIPLIER

        reward = float(np.clip(step_reward, -1.0, 1.0))

        terminated = bool(
            self.current_step >= len(self.df) - 1
            or self.portfolio_value < self.initial_capital * RUIN_THRESHOLD
        )

        self.current_step += 1

        info = {
            "portfolio_value": self.portfolio_value,
            "position_size": self.position_size,
            "trade_count": self.trade_count,
            "drawdown": drawdown,
            "current_capital": self.current_capital,
        }

        return self._get_observation(), reward, terminated, False, info

    def render(self) -> None:
        """Render the environment state to the configured logger."""
        if self.render_mode == "human":
            self.logger.info(
                "Step %d | Portfolio: $%.2f | Shares: %.4f | "
                "Capital: $%.2f | Trades: %d",
                self.current_step,
                self.portfolio_value,
                self.position_size,
                self.current_capital,
                self.trade_count,
            )

    def close(self) -> None:
        """Close the environment and release resources."""
        self.logger.info("Trading environment is closing.")
