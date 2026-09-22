# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import sp, xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.utils.link_functions import scaled_logit


class BetaPriorHyperparameters(PriorHyperparameters):
    """Beta prior hyperparameters.

    p(theta) = (theta^(alpha - 1) * (1 - theta)^(beta - 1)) / B(alpha, beta)

    and in log scale:
    log p(theta) = (alpha - 1) * log(theta) + (beta - 1) * log(1 - theta) - log B(alpha, beta)

    where theta is a parameter on (0, 1), commonly used for probabilities or rates.

    Parameters
    ----------
    config : GaussianPriorHyperparametersConfig
        Configuration object containing alpha and beta parameters.

    Attributes
    ----------
    alpha : float. alpha > 0
        Shape parameter of the Beta distribution.
    beta : float. beta > 0
        Shape parameter of the Beta distribution.
    log_normalizing_constant : float
        Precomputed log of the normalizing constant (log B(alpha, beta)).
    """

    def __init__(
        self,
        config: BetaPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Beta prior hyperparameters.

        Parameters
        ----------
        config : BetaPriorHyperparametersConfig
            Configuration containing alpha and beta parameters.

        Raises
        ------
        ValueError
            If alpha or beta are not positive.
        """
        super().__init__(config)

        self.alpha: float = config.alpha
        self.beta: float = config.beta

        # Validate alpha and beta are positive
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"Beta must be positive, got {self.beta}")

        # Support (lower, upper) of the scaled beta distribution:
        # u = (theta - lower) / (upper - lower) ~ Beta(alpha, beta)
        self.lower, self.upper = (float(v) for v in config.support)
        self.width: float = self.upper - self.lower

        # includes -log(upper - lower), the Jacobian of the affine map to (0, 1)
        self.log_normalizing_constant: float = float(
            sp.special.gammaln(self.alpha + self.beta)
            - sp.special.gammaln(self.alpha)
            - sp.special.gammaln(self.beta)
            - np.log(self.width)
        )

    def _to_unit(self, theta):
        """Map theta from (lower, upper) to (0, 1)."""
        return (theta - self.lower) / self.width

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Beta distribution is defined on (0, 1), but optimization often works
        better in unconstrained space. This method transforms between theta (on (0, 1))
        and logit(theta) (unconstrained, on ℝ).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta -> logit(theta) (external to internal)
            - "backward": logit(theta) -> theta (internal to external)
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
        ### TODO: on longer term make scaled_logit default but let it be configurable in config
        ## beta prior is defined on (lower, upper), while BFGS works on (-inf, inf)
        if direction == "forward":
            theta_scaled = scaled_logit(self._to_unit(theta), direction="forward")
        elif direction == "backward":
            theta_scaled = self.lower + self.width * scaled_logit(
                theta, direction="backward"
            )
        elif direction == "forward_jacobian":
            # d internal / d theta = d internal / d u * d u / d theta
            theta_scaled = (
                scaled_logit(self._to_unit(theta), direction="forward_jacobian")
                / self.width
            )
        elif direction == "backward_log_jacobian":
            # log |d theta / d internal| = log(width) + log |d u / d internal|
            theta_scaled = np.log(self.width) + scaled_logit(
                theta, direction="backward_log_jacobian"
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the Beta distribution
        at the given theta value in its external/user representation.

        Is mainly used for plotting to understand the shape of the prior.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the prior.
            Must be in (0, 1) (external/user representation).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Prior probability density at theta.

        Notes
        -----
        The computation follows:
            p(θ) = C * θ^(α - 1) * (1 - θ)^(β - 1)
        where C is the normalizing constant, α and β are the shape parameters.

        Raises
        ------
        ValueError
            If theta is not in (0, 1).
        """

        u = self._to_unit(theta)
        prior = (
            xp.exp(self.log_normalizing_constant)
            * u ** (self.alpha - 1)
            * (1 - u) ** (self.beta - 1)
        )

        return prior

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Computes the log probability density of the Beta distribution
        at the given theta value in its external/user representation.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
            Must be in (0, 1) (external/user representation).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Notes
        -----
        The computation follows:
            log p(θ) = (α - 1) * log(θ) + (β - 1) * log(1 - θ) - log B(α, β)
        where log B(α, β) is the log normalizing constant.

        Raises
        ------
        ValueError
            If theta is not in (0, 1).
        """

        u = self._to_unit(theta)
        log_prior = (
            (self.alpha - 1) * xp.log(u)
            + (self.beta - 1) * xp.log(1 - u)
            + self.log_normalizing_constant
        )

        return log_prior

    def evaluate_internal_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density transformed to internal space.

        Computes the log prior in unconstrained (internal/logit) space by applying
        a log-Jacobian correction for the change of variables from constrained
        external space (θ ∈ (0, 1)) to unconstrained internal space (θ_internal ∈ ℝ).

        Parameters
        ----------
        theta : float
            Parameter value in internal representation (R).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space:
            log p(θ_internal) = log p_external(g^(-1)(θ_internal)) + log|dθ/dθ_internal g^(-1)(θ_internal)|

        Notes
        -----
        The transformation uses the change of variables computed using
        rescale_hyperparameters_to_internal() with the "backward_log_jacobian"
        direction to account for the Jacobian of the logit transformation.

        Raises
        ------
        ValueError
            If theta is not in (0, 1).
        """

        theta_external = self.rescale_hyperparameters_to_internal(theta, "backward")

        transformed_log_prior = self.evaluate_log_prior(
            theta_external
        ) + self.rescale_hyperparameters_to_internal(theta, "backward_log_jacobian")

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Beta prior hyperparameters with scipy validation.

    Validates:
    1. Prior evaluation against scipy.stats.beta
    2. Log-prior evaluation against scipy.stats.beta.logpdf
    3. Internal log-prior (transformed to logit space) via empirical sampling
    """

    print("=" * 80)
    print("Testing Beta Prior Hyperparameters")
    print("=" * 80)

    from scipy.stats import beta as scipy_beta

    # Test configurations
    alpha_values = [0.5, 1.0, 2.0]
    beta_values = [0.5, 1.0, 2.0]

    for alpha, beta in zip(alpha_values, beta_values):
        print(f"\nTesting alpha={alpha}, beta={beta}")
        config = BetaPriorHyperparametersConfig(alpha=alpha, beta=beta)
        beta_prior = BetaPriorHyperparameters(config=config)

        # Test values in (0, 1)
        test_values = [0.1, 0.25, 0.5, 0.75, 0.9]

        print("Comparing prior evaluations with scipy.stats.beta:")
        for val in test_values:
            p_dalia = beta_prior.evaluate_prior(val)
            p_scipy = scipy_beta.pdf(val, a=alpha, b=beta)
            print(
                f"  θ = {val:4.2f}: DALIA p = {p_dalia:.6f}, "
                f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
            )
            if abs(p_dalia - p_scipy) > 1e-6:
                raise ValueError(
                    "Prior evaluation does not match scipy implementation."
                )

        print("Comparing log prior evaluations with scipy.stats.beta:")
        for val in test_values:
            log_p_dalia = beta_prior.evaluate_log_prior(val)
            log_p_scipy = scipy_beta.logpdf(val, a=alpha, b=beta)
            print(
                f"  θ = {val:4.2f}: DALIA log p = {log_p_dalia:.6f}, "
                f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
            )
            if abs(log_p_dalia - log_p_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

        # Test internal log prior via sampling
        N = 1000000
        theta_external_samples = np.random.beta(a=alpha, b=beta, size=N)
        theta_internal_samples = beta_prior.rescale_hyperparameters_to_internal(
            theta_external_samples, "forward"
        )

        ymin, ymax = theta_internal_samples.min(), theta_internal_samples.max()
        counts, bins = np.histogram(
            theta_internal_samples, bins=500, range=(ymin, ymax), density=True
        )

        y_grid = np.linspace(ymin, ymax, 1000)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out bins with 0 counts
        valid_bins = counts > 0
        empirical_log_density = xp.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        # Compute theoretical log-density
        theoretical_log_density = beta_prior.evaluate_internal_log_prior(y_grid)

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
            y_grid,
            theoretical_log_density,
            color="crimson",
            lw=2.5,
            label="Beta Prior Log-Density (internal space)",
        )
        plt.title(
            f"Beta Prior Internal Log-Space Validation (α={alpha}, β={beta})",
            fontsize=14,
        )
        plt.xlabel("θ_internal (logit space)", fontsize=12)
        plt.ylabel("Log-Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    ######### Check forward transformation
    # Suppose I have now fitted a Gaussian in internal space and want to check that the
    # forward transformation to external space gives the correct density shape.
    # Do this by sampling from the Gaussian in internal space, transforming to external space using the gamma prior's rescaling function,
    # and comparing the empirical density of the transformed samples to the theoretical density computed using the gamma prior's evaluate_prior() method.

    # Create a inverse gamma prior configuration
    mean_values = [-1.0, 3.0, 5.0]
    sd_values = [0.5, 1.0, 2.0]

    for mean, sd in zip(mean_values, sd_values):
        N = 1000000
        internal_samples = np.random.normal(loc=mean, scale=sd, size=N)
        external_samples = beta_prior.rescale_hyperparameters_to_internal(
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
            beta_prior.rescale_hyperparameters_to_internal(external_grid, "forward"),
            loc=mean,
            scale=sd,
        ) * np.abs(
            beta_prior.rescale_hyperparameters_to_internal(
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
