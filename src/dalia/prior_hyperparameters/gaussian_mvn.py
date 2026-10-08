# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from scipy.sparse import spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.priorhyperparameters_config import (
    GaussianMVNPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GaussianMVNPriorHyperparameters(PriorHyperparameters):
    """
    Gaussian multivariate normal (MVN) prior hyperparameters.

    This class implements prior hyperparameters following a multivariate normal
    distribution with specified mean and precision matrix.

    Parameters
    ----------
    config : GaussianMVNPriorHyperparametersConfig
        Configuration object containing mean and precision matrix.

    Attributes
    ----------
    mean : NDArray
        Mean vector of the multivariate normal distribution.
    precision : spmatrix
        Precision matrix (inverse covariance) of the distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GaussianMVNPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gaussian MVN prior hyperparameters.

        Parameters
        ----------
        config : GaussianMVNPriorHyperparametersConfig
            Configuration containing mean vector and precision matrix.

        Raises
        ------
        ValueError
            If the precision matrix is not positive definite.
        """
        super().__init__(config)

        self.mean: NDArray = config.mean
        self.precision: spmatrix = config.precision

        if xp == np:
            self.mean: NDArray = self.mean
            self.precision: spmatrix = self.precision
        else:
            self.mean: NDArray = xp.asarray(self.mean)
            self.precision: sp.sparse.spmatrix = sp.sparse.csc_matrix(self.precision)

        sign, logabsdet = np.linalg.slogdet(self.precision.toarray())
        if sign != 1:
            raise ValueError("Precision matrix must be positive definite.")

        self.normalizing_constant = (
            -0.5 * self.mean.shape[0] * xp.log(2 * xp.pi) + 0.5 * logabsdet
        )
        print("Normalizing constant: ", self.normalizing_constant)

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Rescale hyperparameters between internal and external/user representations.

        Parameters
        ----------
        theta : NDArray
            Hyperparameter values to rescale.
        direction : str
            Direction of rescaling ('forward' or 'backward').

        Returns
        -------
        NDArray
            Rescaled hyperparameter values, which is the identity in this case, therefore unchanged.
        """
        return super().rescale_hyperparameters_to_internal(theta, direction)

    def evaluate_log_prior(self, theta: NDArray, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the multivariate normal
        distribution at the given theta values.

        Parameters
        ----------
        theta : NDArray
            Parameter values at which to evaluate the log prior.
            Must have the same shape as the mean vector.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Raises
        ------
        ValueError
            If theta and mean have incompatible shapes.
        """

        # TODO: add check in config or somewhere else that dim(theta) and dim(mean) match
        if self.mean.shape != theta.shape:
            raise ValueError(
                f"Shape of theta ({theta.shape}) and mean ({self.mean.shape}) do not match."
            )

        if isinstance(self.mean, float):
            return (
                self.normalizing_constant
                - 0.5 * self.precision * (theta - self.mean) ** 2
            )
        else:
            return self.normalizing_constant - 0.5 * (
                theta - self.mean
            ).T @ self.precision @ (theta - self.mean)
