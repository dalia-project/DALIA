# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.likelihood_config import GaussianLikelihoodConfig
from dalia.core.likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: GaussianLikelihoodConfig,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(n_observations, config)

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> float:
        """Evaluate a Gaussian likelihood.

        Notes
        -----

        Evaluate Gaussian log-likelihood for a given set of observations, latent parameters, and design matrix, where
        the observations are assumed to be identically and independently distributed given eta (=A*x). Leading to:
        log (p(y|eta)) = -0.5 * n * log(2 * pi) - 0.5 * n * log(theta) - 0.5 * theta * (y - eta)^T * (y - eta)
        where the constant in front of the likelihood is omitted.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                precision parameter for the likelihood calculation. theta > 0.

        Returns
        -------
        likelihood : float
            Likelihood.
        """

        theta: NDArray = kwargs.get("theta", None)
        if theta is None:
            raise ValueError("theta must be provided to evaluate gaussian likelihood.")

        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

        yEta = eta - y
        # print("xp.exp(theta) in lh:", xp.exp(theta))

        likelihood: float = (
            -0.5 * self.n_observations * xp.log(2 * xp.pi)
            + 0.5 * xp.log(theta) * self.n_observations
            - 0.5 * theta * yEta.T @ yEta
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                precision parameter for the likelihood calculation. theta > 0.

        Returns
        -------
        gradient_likelihood : NDArray
            Gradient of the likelihood.
        """

        theta: NDArray = kwargs.get("theta", None)
        if theta is None:
            raise ValueError(
                "theta must be provided to evaluate gradient of gaussian likelihood."
            )

        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

        gradient_likelihood: NDArray = -theta * (eta - y)

        return gradient_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        theta: float = kwargs.get("theta")
        if theta is None:
            raise ValueError(
                "theta must be provided to evaluate gradient of gaussian likelihood."
            )
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

        # print("hessian lh: xp.exp(theta)", xp.exp(theta))

        hessian_likelihood: ArrayLike = -theta * sp.sparse.eye(self.n_observations)

        return hessian_likelihood
