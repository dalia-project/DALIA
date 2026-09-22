# Copyright 2024-2026 DALIA authors. All rights reserved.
from dalia import xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    HalfCauchyPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class HalfCauchyPriorHyperparameters(PriorHyperparameters):
    """Half-Cauchy prior hyperparameters.

    p(sigma) = 2 / (pi * scale * (1 + (sigma / scale)^2)), for sigma > 0

    and in log scale:
    log p(sigma) = log(2) - log(pi) - log(scale) - log(1 + (sigma / scale)^2)

    where sigma is a positive scale parameter, typically the standard deviation.

    We use the internal parameterization:
    theta = log(1 / sigma^2) = -2 * log(sigma)
    and evaluate the prior in internal space through change of variables:
    log p(theta) = log p(sigma(theta)) + log(|d sigma / d theta|)

    Parameters
    ----------
    config : HalfCauchyPriorHyperparametersConfig
        Configuration object containing the scale parameter.

    Attributes
    ----------
    scale : float. scale > 0. Default is 25.0.
        Scale parameter of the Half-Cauchy distribution.
    log_normalizing_constant : float
        Precomputed log of the normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: HalfCauchyPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Half-Cauchy prior hyperparameters.

        Parameters
        ----------
        config : HalfCauchyPriorHyperparametersConfig
            Configuration containing the scale parameter.

        Raises
        ------
        ValueError
            If scale is not positive.
        """
        super().__init__(config)

        self.scale: float = config.scale

        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

        self.log_normalizing_constant: float = (
            xp.log(2.0) - xp.log(np.pi) - xp.log(self.scale)
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Half-Cauchy distribution is defined on sigma > 0, but optimization often
        works better in unconstrained space. This method transforms between sigma
        (external, positive) and log precision theta = log(1/sigma^2) = -2*log(sigma)
        (internal, unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": sigma -> log(1/sigma^2) (external to internal)
            - "backward": log(1/sigma^2) -> sigma (internal to external)
            - "forward_jacobian": Jacobian of forward transformation
            - "backward_log_jacobian": Log-Jacobian of backward transformation

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not recognized.
        """
        if direction == "forward":  # input is sigma (external)
            theta_scaled = -2.0 * xp.log(theta)
        elif direction == "backward":  # input is theta (internal)
            theta_scaled = xp.exp(-0.5 * theta)
        elif (
            direction == "forward_jacobian"
        ):  # input is sigma (external), should be positive anyway
            theta_scaled = 2.0 / xp.abs(theta)
        elif direction == "backward_log_jacobian":  # input is theta (internal)
            theta_scaled = -xp.log(2.0) - 0.5 * theta
            # log |d(exp(-0.5*theta))/d(theta)| = log|-0.5*exp(-0.5*theta)| = -log(2) - 0.5*theta
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the Half-Cauchy distribution
        at the given sigma value in its external/user representation.

        Is mainly used for plotting to understand the shape of the prior.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the prior.
            Must be positive (external/user representation, sigma).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Prior probability density at sigma.

        Notes
        -----
        The computation follows:
            p(σ) = 2 / (π * scale * (1 + (σ / scale)²))
        """
        if xp.min(theta) <= 0:
            raise ValueError(f"Half-Cauchy sigma must be positive. Got theta={theta}")

        prior = 2.0 / (np.pi * self.scale * (1.0 + (theta / self.scale) ** 2))

        return prior

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Computes the log probability density of the Half-Cauchy distribution
        at the given sigma value in its external/user representation.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
            Must be positive (external/user representation, sigma).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at sigma.

        Notes
        -----
        The computation follows:
            log p(σ) = log(2) - log(π) - log(scale) - log(1 + (σ / scale)²)
        """
        if xp.min(theta) <= 0:
            raise ValueError(f"Half-Cauchy sigma must be positive. Got theta={theta}")

        log_prior = self.log_normalizing_constant - xp.log(
            1.0 + (theta / self.scale) ** 2
        )

        return log_prior

    def evaluate_internal_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density transformed to internal space.

        Computes the log prior in unconstrained (internal/log-precision) space by
        applying a log-Jacobian correction for the change of variables from
        constrained external space (σ > 0) to unconstrained internal space
        (θ_internal ∈ ℝ, where θ_internal = -2*log(σ)).

        Parameters
        ----------
        theta : float
            Parameter value in INTERNAL representation (must be positive, sigma).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space:
            log p(θ_internal) = log p_external(g^{-1}(θ_internal)) + log|dσ/dθ_internal|

        Notes
        -----
        The transformation uses the change of variables formula:
            log p(θ_internal) = log p(σ) + log|dσ/dθ_internal|
        where dσ/dθ_internal = -0.5*exp(-0.5*θ_internal), so
        log|dσ/dθ_internal| = -log(2) - 0.5*θ_internal

        This ensures the log prior is correctly normalized in internal space.
        """

        theta_external = self.rescale_hyperparameters_to_internal(theta, "backward")

        transformed_log_prior = self.evaluate_log_prior(
            theta_external
        ) + self.rescale_hyperparameters_to_internal(theta, "backward_log_jacobian")

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Half-Cauchy prior hyperparameters with scipy validation.

    Validates:
    1. Prior evaluation against scipy.stats.halfcauchy
    2. Log-prior evaluation against scipy.stats.halfcauchy.logpdf
    3. Internal log-prior (transformed to log-precision space) via empirical sampling
    """

    print("=" * 80)
    print("Testing Half-Cauchy Prior Hyperparameters")
    print("=" * 80)

    from scipy.stats import halfcauchy

    # Test configurations
    scale_values = [0.5, 1.0, 2.0]

    for scale in scale_values:
        print(f"\nTesting scale={scale}")
        config = HalfCauchyPriorHyperparametersConfig(scale=scale)
        half_cauchy_prior = HalfCauchyPriorHyperparameters(config=config)

        # Test values: sigma must be positive
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        print("Comparing prior evaluations with scipy.stats.halfcauchy:")
        for val in test_values:
            p_dalia = half_cauchy_prior.evaluate_prior(val)
            p_scipy = halfcauchy.pdf(val, loc=0, scale=scale)
            print(
                f"  σ = {val:4.1f}: DALIA p = {p_dalia:.6f}, "
                f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
            )
            if abs(p_dalia - p_scipy) > 1e-6:
                raise ValueError(
                    "Prior evaluation does not match scipy implementation."
                )

        print("Comparing log prior evaluations with scipy.stats.halfcauchy:")
        for val in test_values:
            log_p_dalia = half_cauchy_prior.evaluate_log_prior(val)
            log_p_scipy = halfcauchy.logpdf(val, loc=0, scale=scale)
            print(
                f"  σ = {val:4.1f}: DALIA log p = {log_p_dalia:.6f}, "
                f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
            )
            if abs(log_p_dalia - log_p_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

        # Test internal log prior via sampling
        N = 1000000
        sigma_external_samples = halfcauchy.rvs(loc=0, scale=scale, size=N)
        theta_internal_samples = half_cauchy_prior.rescale_hyperparameters_to_internal(
            sigma_external_samples, "forward"
        )
        theta_min, theta_max = (
            theta_internal_samples.min(),
            theta_internal_samples.max(),
        )
        counts, bins = np.histogram(
            theta_internal_samples, bins=500, range=(theta_min, theta_max), density=True
        )

        # Create grid over the observed internal space range (avoid extreme boundaries)
        theta_grid = np.linspace(theta_min, theta_max, 1000)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter to core region (exclude low-count tail bins)
        min_count = 0
        valid_bins = counts > min_count
        empirical_log_density = np.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        # Restrict theoretical evaluation to empirical range
        empirical_theta_min = filtered_bin_centers.min()
        empirical_theta_max = filtered_bin_centers.max()

        # Evaluate theoretical log density in internal space
        theoretical_log_density = half_cauchy_prior.evaluate_internal_log_prior(
            theta_grid
        )

        from matplotlib import pyplot as plt

        # Plot comparison
        plt.figure(figsize=(9, 6))
        plt.scatter(
            filtered_bin_centers,
            empirical_log_density,
            color="royalblue",
            s=15,
            alpha=0.8,
            label="Empirical Log-Densities (from samples)",
        )
        plt.plot(
            theta_grid,
            theoretical_log_density,
            color="crimson",
            lw=2.5,
            label="Half-Cauchy Log-Density (internal space)",
        )
        plt.title(
            f"Half-Cauchy Prior Internal Log-Space Validation (scale={scale})",
            fontsize=14,
        )
        plt.xlabel("θ (log precision)", fontsize=12)
        plt.ylabel("Log-Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    ######### Check forward transformation
    # Suppose I have now fitted a Gaussian in internal space and want to check that the
    # forward transformation to external space gives the correct density shape.
    # Do this by sampling from the Gaussian in internal space, transforming to external space using the half-Cauchy prior's rescaling function,
    # and comparing the empirical density of the transformed samples to the theoretical density computed using the half-Cauchy prior's evaluate_prior() method.

    # Create a inverse gamma prior configuration
    mean_values = [-1.0, 3.0, 5.0]
    sd_values = [0.5, 1.0, 2.0]

    for mean, sd in zip(mean_values, sd_values):
        N = 1000000
        internal_samples = np.random.normal(loc=mean, scale=sd, size=N)
        external_samples = half_cauchy_prior.rescale_hyperparameters_to_internal(
            internal_samples, "backward"
        )

        xmin, xmax = external_samples.min(), external_samples.max()
        counts, bins = np.histogram(
            external_samples, bins=500, range=(xmin, xmax), density=True
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2

        valid_bins = counts > 0
        empirical_log_density = np.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        external_grid = np.linspace(xmin, xmax, 1000)

        from scipy.stats import norm

        theoretical_density = norm.pdf(
            half_cauchy_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward"
            ),
            loc=mean,
            scale=sd,
        ) * np.abs(
            half_cauchy_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward_jacobian"
            )
        )

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
            "Validation of Forward Projection with Jacobian Correction", fontsize=14
        )
        plt.xlabel("x (External Space)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    print("\nAll value direct value comparisons passed. Check sampling plots.")
