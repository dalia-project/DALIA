# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod
from dalia import xp

from dalia.configs.priorhyperparameters_config import PriorHyperparametersConfig


class PriorHyperparameters(ABC):
    """Abstract core class for prior hyperparameters."""

    def __init__(
        self,
        config: PriorHyperparametersConfig,
    ) -> None:
        """Initializes the prior hyperparameters."""

        self.config: PriorHyperparametersConfig = config

    @abstractmethod
    def rescale_hyperparameters_to_internal(self, theta: float, direction: str) -> float:
        """Rescale hyperparameters to and from internal scale.

        Args:
            theta: Hyperparameter
            direction: "forward", "backward", "forward_jacobian", "backward_jacobian"
        Returns:
            Rescaled hyperparameter.

        """
        
        if direction == "forward" or direction == "backward":
            return theta
        elif direction == "forward_jacobian" or direction == "backward_jacobian":
            return xp.ones_like(theta)
        
    @abstractmethod
    def evaluate_log_prior(self, theta: float) -> float:
        """Evaluate the log prior hyperparameters."""
        pass
