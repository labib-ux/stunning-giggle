"""Training and inference utilities for the reinforcement learning agent."""

import os
import logging
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from agent.policy import CustomFeatureExtractor
from config import MODEL_SAVE_PATH

logger = logging.getLogger(__name__)


class TradingAgent:
    """Wrapper for PPO training, loading, and inference."""

    def __init__(
        self,
        env: object | None = None,
        tensorboard_log: str = "./tensorboard_logs/",
    ) -> None:
        """Initialize the trading agent and underlying PPO model.

        Args:
            env: Optional Gymnasium environment instance.
            tensorboard_log: TensorBoard logging directory.

        Raises:
            Exception: Re-raises any model initialization failures.
        """
        self.env = env
        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "cpu"
            self.logger.warning(
                "Apple MPS detected but defaulting to CPU. "
                "MPS has known instability with stable-baselines3 PPO "
                "advantage calculation. Pass device='mps' manually to override."
            )
        else:
            self.device = "cpu"
        self.logger.info("Training device selected: %s", self.device)

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )

        try:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.isfile(MODEL_SAVE_PATH):
                self.model = PPO.load(
                    MODEL_SAVE_PATH,
                    env=self.env,
                    device=self.device,
                    tensorboard_log=tensorboard_log,
                )
                self.logger.info("Loaded existing model from: %s", MODEL_SAVE_PATH)
            else:
                if os.path.exists(MODEL_SAVE_PATH) and not os.path.isfile(MODEL_SAVE_PATH):
                    self.logger.warning(
                        "MODEL_SAVE_PATH '%s' exists but is a directory, not a file. "
                        "Initializing new model instead.",
                        MODEL_SAVE_PATH,
                    )
                self.model = PPO(
                    "MlpPolicy",
                    env=self.env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    ent_coef=0.01,
                    clip_range=0.2,
                    verbose=0,
                    device=self.device,
                    tensorboard_log=tensorboard_log,
                )
                self.logger.info("Initialized new PPO model.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Model initialization failed: %s", exc, exc_info=True)
            raise

    def train(
        self,
        total_timesteps: int = 100000,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train the PPO model on the provided environment.

        reset_num_timesteps explanation:
            True  = fresh training run, timestep counter resets to 0,
                    learning rate schedule starts from beginning.
                    Use for initial training.
            False = continued training, timestep counter accumulates,
                    TensorBoard graphs continue from previous run.
                    Use explicitly when resuming — do not default to False
                    or learning rate will be incorrectly decayed from step 1.

        Args:
            total_timesteps: Total number of training timesteps.
            reset_num_timesteps: Whether to reset timestep counters.

        Raises:
            ValueError: If no environment is attached to the agent.
            Exception: Re-raises any training failures.
        """
        if self.env is None:
            raise ValueError(
                "Cannot train: no environment provided to TradingAgent."
            )

        self.logger.info(
            "Starting training: total_timesteps=%s reset_num_timesteps=%s",
            total_timesteps,
            reset_num_timesteps,
        )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True,
            )
            self.save()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Training failed at step: %s", exc, exc_info=True)
            raise

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict the next action for a given observation.

        Args:
            observation: Environment observation array.
            deterministic: Whether to use deterministic policy output.

        Returns:
            Integer action identifier.
        """
        try:
            action_array, _state = self.model.predict(
                observation,
                deterministic=deterministic,
            )
            return int(action_array.flat[0])
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "predict() failed: %s — returning HOLD (0) as safe fallback",
                exc,
                exc_info=True,
            )
            return 0

    def save(self) -> None:
        """Persist the model to disk."""
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            self.model.save(MODEL_SAVE_PATH)
            self.logger.info("Model saved successfully to: %s", MODEL_SAVE_PATH)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Model save failed: %s", exc, exc_info=True)
