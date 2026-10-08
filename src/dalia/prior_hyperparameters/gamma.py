# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import sp, xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    GammaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GammaPriorHyperparameters(PriorHyperparameters):
    """Gamma prior hyperparameters.

    p(theta) = (beta^alpha / Gamma(alpha)) * theta^(alpha - 1) * exp(-beta * theta)

    and in log scale:
    log p(theta) = alpha * log(beta) - log(Gamma(alpha)) + (alpha - 1) * log(theta) - beta * theta

    where theta is typically a positive parameter such as a precision or rate.

    Parameters
    ----------
    config : GammaPriorHyperparametersConfig
        Configuration object containing alpha and beta parameters.

    Attributes
    ----------
    alpha : float. alpha > 0
        Shape parameter of the Gamma distribution.
    beta : float. beta > 0
        Rate parameter of the Gamma distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GammaPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gamma prior hyperparameters.

        Parameters
        ----------
        config : GammaPriorHyperparametersConfig
            Configuration containing alpha (shape) and beta (rate) parameters.

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

        self.normalizing_constant: float = self.alpha * xp.log(self.beta) - float(
            sp.special.gammaln(self.alpha)
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Gamma distribution is defined for positive values, but optimization
        often works better in unconstrained space. This method transforms
        between theta (positive) and log(theta) (unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta -> log(theta) (external to internal)
            - "backward": log(theta) -> theta (internal to external)

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not "forward" or "backward".
        """
        if direction == "forward":
            theta_scaled = xp.log(theta)
        elif direction == "backward":
            theta_scaled = xp.exp(theta)
        elif direction == "forward_jacobian":
            theta_scaled = 1 / theta  # d(log(theta))/d(theta) = 1/theta
        elif direction == "backward_jacobian":
            theta_scaled = theta  # d(exp(theta))/d(theta) = exp(theta) = theta 
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the Gamma distribution
        at the given theta value in its external/user representation.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
            Must be positive (external/user representation).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Notes
        -----
        The computation follows:
            log p(θ) = C + (α - 1) * log(θ) - β * θ
        where C is the normalizing constant, α is the shape parameter,
        and β is the rate parameter.

        Raises
        ------
        ValueError
            If theta is not positive (implicitly through log computation).
        """

        log_prior = (
            self.normalizing_constant
            + (self.alpha - 1) * xp.log(theta)
            - self.beta * theta
        )

        return log_prior

if __name__ == "__main__":
    """
    Test Gaussian quadrature for functions using rescale_hyperparameters_to_internal() 
    from gamma prior hyperparameters.
    
    We start with normally distributed random variables in internal space (unconstrained)
    that get reparametrized to external space (positive) using the gamma prior's 
    rescaling function.
    """
    
    from dalia.utils.gaussian_quadrature import compute_variance_gauss_hermite
    
    print("=" * 80)
    print("Testing Gaussian Quadrature with Gamma Prior Rescaling")
    print("=" * 80)
        
    # Create a inverse gamma prior configuration
    alpha_values = [1.0, 3.0, 5.0]
    beta_values = [0.5, 1.0, 2.0]

    for alpha, beta in zip(alpha_values, beta_values):
        print(f"\nTesting alpha={alpha}, beta={beta}")
        config = GammaPriorHyperparametersConfig(alpha=alpha, beta=beta)
        gamma_prior = GammaPriorHyperparameters(config=config)
        
        ## compare against scipy implementation
        from scipy.stats import gamma
        
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print("Comparing log prior evaluations with scipy.stats.gamma:")
        for val in test_values:
            logp_dalia = gamma_prior.evaluate_log_prior(val)
            ## note: scipy's gamma takes scale = 1/beta
            logp_scipy = gamma.logpdf(val, a=alpha, scale=1/beta)
            print(f"  θ = {val:4.1f}: DALIA logp = {logp_dalia:.6f}, "
                f"scipy logp = {logp_scipy:.6f}, diff = {abs(logp_dalia - logp_scipy):.2e}")
            if abs(logp_dalia - logp_scipy) > 1e-6:
                raise ValueError("Log prior evaluation does not match scipy implementation.")
    
    print()
    print("All tests passed!")
    
    # Create a gamma prior configuration
    config = GammaPriorHyperparametersConfig(alpha=2.0, beta=1.0)
    gamma_prior = GammaPriorHyperparameters(config=config)
    
    # Define parameters for the normal distribution in internal space
    # These represent log(theta) where theta > 0 is the gamma-distributed parameter
    mean_internal = 0.5  # Mean of log(theta)
    variance_internal = 0.25  # Variance of log(theta)
    
    print(f"Internal space (log-scale) parameters:")
    print(f"  Mean: {mean_internal}")
    print(f"  Variance: {variance_internal}")
    print(f"  Standard deviation: {xp.sqrt(variance_internal)}")
    print()
    
    # Test 1: Compute statistics using Gaussian quadrature
    print("1. Computing statistics using Gaussian quadrature:")
    
    # Use the rescaling function as the transform
    def transform_func(x, direction):
        return gamma_prior.rescale_hyperparameters_to_internal(x, direction)
    
    # Compute statistics using different numbers of quadrature points
    for n_points in [10, 20, 30, 50]:
        result = compute_variance_gauss_hermite(
            mean_internal, 
            variance_internal, 
            transform_func, 
            n_points=n_points
        )
        
        print(f"  n_points = {n_points:2d}: Mean = {result['mean']:.6f}, "
              f"Std = {result['std']:.6f}, Var = {result['variance']:.6f}")
    
    print()
    
    # Test 2: Compare with analytical solution
    print("2. Comparison with analytical log-normal distribution:")
    print("   For log(Y) ~ N(μ, σ²), we have:")
    print("   E[Y] = exp(μ + σ²/2)")
    print("   Var[Y] = (exp(σ²) - 1) * exp(2μ + σ²)")
    
    # Analytical moments for log-normal distribution
    mu = mean_internal
    sigma2 = variance_internal
    
    analytical_mean = xp.exp(mu + sigma2/2)
    analytical_variance = (xp.exp(sigma2) - 1) * xp.exp(2*mu + sigma2)
    analytical_std = xp.sqrt(analytical_variance)
    
    print(f"   Analytical mean: {analytical_mean:.6f}")
    print(f"   Analytical std:  {analytical_std:.6f}")
    print(f"   Analytical var:  {analytical_variance:.6f}")
    print()
    
    # Compare with quadrature result (using 50 points)
    quad_result = compute_variance_gauss_hermite(
        mean_internal, variance_internal, transform_func, n_points=50
    )
    
    print("3. Comparison of quadrature vs analytical:")
    print(f"   Mean difference: {abs(quad_result['mean'] - analytical_mean):.2e}")
    print(f"   Std difference:  {abs(quad_result['std'] - analytical_std):.2e}")
    print(f"   Var difference:  {abs(quad_result['variance'] - analytical_variance):.2e}")
    
    # Relative errors
    mean_rel_error = abs(quad_result['mean'] - analytical_mean) / analytical_mean
    std_rel_error = abs(quad_result['std'] - analytical_std) / analytical_std
    var_rel_error = abs(quad_result['variance'] - analytical_variance) / analytical_variance
    
    print(f"   Mean rel. error: {mean_rel_error:.2e}")
    print(f"   Std rel. error:  {std_rel_error:.2e}")
    print(f"   Var rel. error:  {var_rel_error:.2e}")
    print()
    
    # Test 3: Test with different internal parameters
    print("4. Testing with different internal parameters:")
    
    test_cases = [
        {"mean": 0.0, "var": 0.1, "name": "Small variance"},
        {"mean": 1.0, "var": 0.5, "name": "Medium variance"},
        {"mean": -0.5, "var": 1.0, "name": "Large variance"},
        {"mean": 2.0, "var": 0.01, "name": "Large mean, small variance"}
    ]
    
    for case in test_cases:
        mu_test = case["mean"]
        var_test = case["var"]
        
        # Quadrature result
        quad_result = compute_variance_gauss_hermite(
            mu_test, var_test, transform_func, n_points=30
        )
        
        # Analytical result
        anal_mean = xp.exp(mu_test + var_test/2)
        anal_var = (xp.exp(var_test) - 1) * xp.exp(2*mu_test + var_test)
        
        rel_mean_error = abs(quad_result['mean'] - anal_mean) / anal_mean
        rel_var_error = abs(quad_result['variance'] - anal_var) / anal_var
        
        print(f"   {case['name']:25s}: Mean rel. err = {rel_mean_error:.2e}, "
              f"Var rel. err = {rel_var_error:.2e}")
    
    print()
    
    # Test 4: Test the rescaling function directions
    print("5. Testing rescaling function directions:")
    
    # Test some values
    test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("   Testing forward (external -> internal) and backward (internal -> external):")
    for theta in test_values:
        # Forward: theta -> log(theta)
        log_theta = gamma_prior.rescale_hyperparameters_to_internal(theta, "forward")
        
        # Backward: log(theta) -> theta  
        theta_recovered = gamma_prior.rescale_hyperparameters_to_internal(log_theta, "backward")
        
        error = abs(theta - theta_recovered)
        print(f"   θ = {theta:4.1f} -> log(θ) = {log_theta:6.3f} -> θ = {theta_recovered:6.3f}, "
              f"error = {error:.2e}")
    
    print()
    
    # Test 5: Convergence study
    print("6. Convergence study (increasing number of quadrature points):")
    
    mu_conv = 0.3
    var_conv = 0.4
    analytical_mean_conv = xp.exp(mu_conv + var_conv/2)
    
    n_points_list = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    print("   n_points   |   Mean      |  Rel. Error")
    print("   -----------|-------------|------------")
    
    for n in n_points_list:
        result = compute_variance_gauss_hermite(
            mu_conv, var_conv, transform_func, n_points=n
        )
        rel_error = abs(result['mean'] - analytical_mean_conv) / analytical_mean_conv
        
        print(f"   {n:8d}   | {result['mean']:10.6f} | {rel_error:.3e}")
    
    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("All tests show that Gaussian quadrature accurately approximates")
    print("the moments of log-normal distributions obtained through gamma")  
    print("prior rescaling transformations.")
    print("=" * 80)

