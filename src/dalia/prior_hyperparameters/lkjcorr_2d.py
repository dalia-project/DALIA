# Copyright 2024-2026 DALIA authors. All rights reserved.

import numpy as np
from scipy import special as scipy_special

from dalia import sp, xp

from dalia.configs.priorhyperparameters_config import (
    LKJCorrPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters

## \Sigma = ([\sigma_alpha^2, \rho \sigma_alpha \sigma_beta], [\rho \sigma_alpha \sigma_beta, \sigma_beta^2])
# deconstruct into
# \Sigma = ([\sigma_alpha^2, 0], [0, \sigma_beta^2]) * ([1, \rho], [\rho, 1]) * ([\sigma_alpha^2, 0], [0, \sigma_beta^2])

# P(\sigma_alpha, \sigma_beta, \rho) = P(\sigma_alpha) * P(\sigma_beta) * P(\rho)

## put half-normal priors on \sigma_alpha and \sigma_beta
# R = [1, \rho], [\rho, 1], det(R) = 1 - \rho^2 ## LKJ prior here
# P(R | \eta) \propto det(R)^{\eta - 1} = (1 - \rho^2)^{\eta - 1}

## so then in log form: (\eta - 1) * log det(R) = (\eta - 1) * log(1 - \rho^2)
# need to manually pick eta

## need rescaling for optimization to unconstrained space, so we use theta = log((1 + \rho) / (1 - \rho)) = 2 * atanh(\rho)
## reverse is rho = 2 * logistic(theta) - 1 = tanh(theta / 2), where theta \in (-\infty, \infty),
## need Jacobian for this transformation
# d rho / d theta = (1 - \rho^2) / 2
# log p(theta | \eta) = log p(rho | \eta) + log |d rho / d theta| = C + \eta * log(1 - \rho^2) - log(2)


class LKJCorrPriorHyperparameters(PriorHyperparameters):
    r"""LKJ prior hyperparameters.

    p(theta) \propto (1 - theta^2)^{eta - 1} for theta in (-1, 1), where eta > 0 is the shape parameter.

    and in log scale:
    log p(theta) = (eta - 1) * log(1 - theta^2)

    where theta is a parameter on (-1, 1), commonly used for correlation coefficients.

    Parameters
    ----------
    config : LKJCorrPriorHyperparametersConfig
        Configuration object containing eta parameter.

    Attributes
    ----------
    eta : float. eta > 0
        Shape parameter of the LKJ distribution.
    """

    def __init__(
        self,
        config: LKJCorrPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the LKJ prior hyperparameters.

        Parameters
        ----------
        config : LKJCorrPriorHyperparametersConfig
            Configuration containing eta parameter.

        Note
        ----
        The LKJ prior is defined for correlation matrices, meaning the marginal variances are fixed to 1 and need to
        be handled separately, e.g. using half-normal priors for the standard deviations. The LKJ prior is only applied to the correlation coefficients.
        """
        super().__init__(config)

        self.eta: float = config.eta

        # Validate eta is positive
        if self.eta <= 0:
            raise ValueError(f"Eta must be positive, got {self.eta}")

        # int_{-1}^{1} (1 - rho^2)^(eta - 1) d rho = 2^(2 eta - 1) * B(eta, eta)
        self.log_normalizing_constant: float = -(
            (2 * self.eta - 1) * float(np.log(2.0))
            + float(scipy_special.betaln(self.eta, self.eta))
        )

    def evaluate_prior(self, rho):
        """
        Evaluate the LKJ prior for a given correlation coefficient rho.

        Parameters
        ----------
        rho : float
            Correlation coefficient (must be in [-1, 1]).

        Returns
        -------
        float
            Prior probability density of the correlation coefficient.
        """
        if xp.any(xp.abs(xp.asarray(rho)) > 1):
            raise ValueError("Correlation coefficient rho must be in [-1, 1].")

        # For a 2x2 correlation matrix, the determinant is (1 - rho^2)
        det_R = 1 - rho**2

        # The LKJ prior density is proportional to det(R)^(eta - 1)
        prior_density = float(np.exp(self.log_normalizing_constant)) * det_R ** (
            self.eta - 1
        )

        return prior_density

    def evaluate_log_prior(self, rho):
        """
        Log prior for the LKJ distribution for a 2x2 correlation matrix.

        Parameters:
        -----------
        eta : float
            Shape parameter (eta > 0). eta=1 gives uniform correlation matrices,
            larger values concentrate near identity matrix.
        rho : float
            Correlation coefficient (must be in [-1, 1]).

        Returns:
        --------
        log_prior : float
            Log prior probability of the correlation matrix.
        """
        if xp.any(xp.abs(xp.asarray(rho)) > 1):
            raise ValueError("Correlation coefficient rho must be in [-1, 1].")

        # For a 2x2 correlation matrix, the determinant is (1 - rho^2)
        det_R = 1 - rho**2

        # The LKJ prior density is proportional to det(R)^(eta - 1)
        log_prior = self.log_normalizing_constant + (self.eta - 1) * xp.log(det_R)

        return log_prior

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The LKJ 2d distribution has only one parameter between (-1,1), but optimization
        works better in unconstrained space. This function transforms
        between constrained theta (in (-1,1)) and transformed theta (unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta_internal = log((1 + theta) / (1 - theta)) (external to internal)
            - "backward": theta_external = 2 * logit^{-1}(theta_internal) -1 (internal to external)
            - "forward_jacobian": derivative of forward transformation
            - "backward_log_jacobian": log of the derivative of backward transformation

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not "forward" or "backward", "forward_jacobian" or "backward_log_jacobian".
        """

        if direction == "forward":  # input is theta (external)
            theta_scaled = xp.log((1 + theta) / (1 - theta))
        elif direction == "backward":  # input is theta (internal)
            theta_scaled = 2 * (1 / (1 + xp.exp(-theta))) - 1
        elif direction == "forward_jacobian":  # input is theta (external)
            theta_scaled = 2 / (1 - theta**2)
        elif direction == "backward_log_jacobian":  # input is theta (internal)
            theta_scaled = xp.log(2) - theta - 2 * xp.log1p(xp.exp(-theta))
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_internal_log_prior(self, theta):
        """
        Log prior for the LKJ distribution for a 2x2 correlation matrix in internal parameterization.

        Parameters:
        -----------
        eta : float
            Shape parameter (eta > 0). eta=1 gives uniform correlation matrices,
            larger values concentrate near identity matrix.
        theta : float
            Internal parameter (unconstrained, real-valued).

        Returns:
        --------
        log_prior : float
            Log prior probability of the correlation matrix.
        """
        # Transform theta to rho
        theta_external = self.rescale_hyperparameters_to_internal(
            theta, direction="backward"
        )

        # Compute the log prior using the external parameterization
        log_prior = self.evaluate_log_prior(theta_external)

        # Add the Jacobian adjustment for the transformation from theta to rho
        log_jacobian = self.rescale_hyperparameters_to_internal(
            theta, direction="backward_log_jacobian"
        )

        return log_prior + log_jacobian


## currently only for 2x2 correlation matrices
# TODO: extend to higher dimensions

import numpy as np

np.random.seed(41)

## factorize prior into P(\sigma1, \sigma2, \rho) = P(\sigma1) * P(\sigma2) * P(\rho)
## we will only consider the LKJ prior for \rho, and assume \sigma1 and \sigma2 are fixed for this test

if __name__ == "__main__":

    # test that LKJ rescaling function works correctly
    theta_external_list = np.linspace(-0.99, 0.99, 20)
    eta = 2.0  # example eta value for testing
    config = LKJCorrPriorHyperparametersConfig(eta=eta)
    lkj_corr_prior = LKJCorrPriorHyperparameters(config=config)

    for theta_external in theta_external_list:
        theta_internal = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_external, direction="forward"
        )
        theta_external_back = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_internal, direction="backward"
        )
        assert np.isclose(
            theta_external, theta_external_back
        ), f"Rescaling failed for theta_external={theta_external}"
    print("LKJ forward-backward rescaling function is consistent.")

    # verify jacobians using finite differences
    epsilon = 1e-6
    for theta_external in theta_external_list:
        jac_forward_analytical = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_external, direction="forward_jacobian"
        )

        f_plus = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_external + epsilon, direction="forward"
        )
        f_minus = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_external - epsilon, direction="forward"
        )
        jac_forward_numerical = (f_plus - f_minus) / (2 * epsilon)

        assert np.isclose(
            jac_forward_analytical, jac_forward_numerical, rtol=1e-5
        ), f"Jacobian check failed for theta_external={theta_external}"
    print("LKJ forward jacobian check passed.")

    theta_internal_list = np.linspace(-5, 5, 20)

    for theta_internal in theta_internal_list:
        log_back_jacobian_analytical = (
            lkj_corr_prior.rescale_hyperparameters_to_internal(
                theta_internal, direction="backward_log_jacobian"
            )
        )

        f_plus = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_internal + epsilon, direction="backward"
        )
        f_minus = lkj_corr_prior.rescale_hyperparameters_to_internal(
            theta_internal - epsilon, direction="backward"
        )
        log_jacobian_numerical = np.log((f_plus - f_minus) / (2 * epsilon))

        assert np.isclose(
            log_back_jacobian_analytical, log_jacobian_numerical, rtol=1e-5
        ), f"Log Jacobian check failed for theta_internal={theta_internal}"
    print("LKJ backward log jacobian check passed.")

    # ========================================================================
    # Validation Test 2: Forward Transformation with Jacobian Correction
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION TESTS: LKJ Prior - Forward Transformation")
    print("=" * 70)

    from scipy.stats import norm
    import matplotlib.pyplot as plt

    N = 1000000

    mean_values = [-1.2, 0.0, 3.0]
    sd_values = [0.2, 1.0, 3.5]

    for mean, sd in zip(mean_values, sd_values):
        internal_samples = np.random.normal(loc=mean, scale=sd, size=N)

        # Transform to external space (rho in [-1, 1])
        external_samples = lkj_corr_prior.rescale_hyperparameters_to_internal(
            internal_samples, direction="backward"
        )

        xmin, xmax = (
            external_samples.min(),
            external_samples.max(),
        )
        counts, bins = np.histogram(
            external_samples, bins=500, range=(xmin, xmax), density=True
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2

        external_grid = np.linspace(xmin, xmax, 1000)

        # Compute theoretical density with Jacobian correction
        # P(rho) = P(theta) * |d_theta/d_rho|
        # where theta = forward(rho) and |d_theta/d_rho| = |d_forward/d_rho|

        forward_transformed = lkj_corr_prior.rescale_hyperparameters_to_internal(
            external_grid, direction="forward"
        )
        jacobian_forward = lkj_corr_prior.rescale_hyperparameters_to_internal(
            external_grid, direction="forward_jacobian"
        )

        # Theoretical density in external space
        theoretical_density = (
            norm.pdf(
                forward_transformed,
                loc=mean,
                scale=sd,
            )
            * jacobian_forward
        )

        # Plot validation
        plt.figure(figsize=(9, 6))
        plt.scatter(
            bin_centers,
            counts,
            color="royalblue",
            s=15,
            alpha=0.8,
            label="Empirical Densities (from samples)",
        )
        plt.plot(
            external_grid,
            theoretical_density,
            color="crimson",
            lw=2.5,
            label="Theoretical Density with Jacobian Correction",
        )
        plt.title(
            f"LKJ Prior: Forward Transformation Validation (μ={mean}, σ={sd})",
            fontsize=14,
        )
        plt.xlabel("ρ (External Space)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        plt.show()

    print("\n" + "=" * 70)
