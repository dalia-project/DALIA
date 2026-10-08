# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import xp
from dalia.configs.priorhyperparameters_config import (
    PenalizedComplexityPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class PenalizedComplexityPriorHyperparameters(PriorHyperparameters):
    """Penalized Complexity prior hyperparameters."""

    def __init__(
        self,
        config: PenalizedComplexityPriorHyperparametersConfig,
        **kwargs,
    ) -> None:
        """Initializes the Penalized Complexity prior hyperparameters."""
        super().__init__(config)

        self.hyperparameter_type: str = kwargs.get("hyperparameter_type")

        self.alpha: float = config.alpha
        self.u: float = config.u

        self.lambda_theta: float = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

            self.lambda_theta = -xp.log(self.alpha) * pow(
                self.u,
                0.5 * spatial_dim,
            )
        elif self.hyperparameter_type == "r_t":
            self.lambda_theta = -xp.log(self.alpha) * pow(self.u, 0.5)
        elif (
            self.hyperparameter_type == "sigma_st"
            or self.hyperparameter_type == "sigma_e"
        ):
            self.lambda_theta = -xp.log(self.alpha) / self.u
        elif self.hyperparameter_type == "prec_o":
            self.lambda_theta = -xp.log(self.alpha) / self.u

        # print("lambda_theta: ", self.lambda_theta)

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """Rescale hyperparameters to and from internal scale.
        
        TODO: Implement the re-scaling.
        """
        return super().rescale_hyperparameters_to_internal(theta, direction)

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the prior hyperparameters."""
        log_prior: float = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

            if spatial_dim == 2:
                log_prior = (
                    xp.log(self.lambda_theta)
                    - self.lambda_theta * xp.exp(-theta)
                    - theta
                )
            else:
                raise ValueError("Not implemented for other than 2D spatial domains")
            # print("log prior r s: ", log_prior)
        elif self.hyperparameter_type == "r_t":
            log_prior = (
                xp.log(self.lambda_theta)
                - self.lambda_theta * xp.exp(-0.5 * theta)
                + xp.log(0.5)
                - 0.5 * theta
            )
            # print("log prior r t: ", log_prior)
        elif (
            self.hyperparameter_type == "sigma_st"
            or self.hyperparameter_type == "sigma_e"
        ):
            log_prior = (
                xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) + theta
            )
            # print("log prior sigma e: ", log_prior)
        elif self.hyperparameter_type == "prec_o":
            log_prior = (
                xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) + theta
            )
            # print("log prior prec o: ", log_prior)

        return log_prior
