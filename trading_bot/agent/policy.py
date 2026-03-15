"""Policy feature extractor definitions for reinforcement learning agents."""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor with LayerNorm safeguards."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
    ) -> None:
        """Initialize the feature extractor network.

        Args:
            observation_space: Observation space describing input shape.
            features_dim: Output feature dimension for downstream policy heads.
        """
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor.

        Args:
            observations: Input observation tensor.

        Returns:
            Extracted feature tensor.
        """
        return self.network(observations)
