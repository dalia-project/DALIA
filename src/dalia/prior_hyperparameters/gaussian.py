# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from scipy.sparse import spmatrix
from dalia import sp, xp

from dalia import NDArray
from dalia.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GaussianPriorHyperparameters(PriorHyperparameters):
    """
    Univariate Gaussian prior hyperparameters.

    This class implements prior hyperparameters following a univariate normal
    (Gaussian) distribution with specified mean and precision (inverse variance).

    Parameters
    ----------
    config : GaussianPriorHyperparametersConfig
        Configuration object containing mean and precision parameters.

    Attributes
    ----------
    mean : float
        Mean of the Gaussian distribution.
    precision : float
        Precision (inverse variance) of the Gaussian distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GaussianPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gaussian prior hyperparameters.

        Parameters
        ----------
        config : GaussianPriorHyperparametersConfig
            Configuration containing mean and precision parameters.

        Raises
        ------
        ValueError
            If the precision is not positive.
        """
        super().__init__(config)

        self.mean: float = config.mean
        self.precision: float = config.precision

        # Validate precision is positive
        if self.precision <= 0:
            raise ValueError(f"Precision must be positive, got {self.precision}")

        self.normalizing_constant = -0.5 * xp.log(2 * xp.pi) + 0.5 * xp.log(
            self.precision
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Rescale hyperparameters between internal and external representations.

        Parameters
        ----------
        theta : NDArray or float
            Hyperparameter values to rescale.
        direction : str
            Direction of rescaling ('forward' or 'backward', i.e. from interpretable/user/external to internal or vice versa).

        Returns
        -------
        NDArray or float
            Rescaled hyperparameter values. In this case it is the identity, therefore unchanged.

        Notes
        -----
        For Gaussian priors, the rescaling is the identity function since the internal and external representations are the same.
        """
        return super().rescale_hyperparameters_to_internal(theta, direction)

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the univariate normal
        distribution at the given theta value.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Notes
        -----
        The computation follows:
            log p(θ) = C - 0.5 * τ * (θ - μ)²
        where C is the normalizing constant, τ is precision, and μ is the mean.
        """
        return (
            self.normalizing_constant - 0.5 * self.precision * (theta - self.mean) ** 2
        )
