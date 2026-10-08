# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import sp, xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    InverseGammaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class InverseGammaPriorHyperparameters(PriorHyperparameters):
    """Inverse Gamma prior hyperparameters.

    p(theta) = (beta^alpha / Gamma(alpha)) * (1/theta)^(alpha + 1) * exp(-beta / theta)

    and in log scale:
    log p(theta) = alpha * log(beta) - log(Gamma(alpha)) - (alpha + 1) * log(theta) - beta / theta

    where theta is typically a positive parameter such as a variance.

    Parameters
    ----------
    config : InverseGammaPriorHyperparametersConfig
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
        config: InverseGammaPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Inverse Gamma prior hyperparameters.

        Parameters
        ----------
        config : InverseGammaPriorHyperparametersConfig
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

        The Inverse Gamma distribution is defined for positive values, but the optimization
        happens in an unconstrained space. This method transforms
        between theta (positive) and log(theta) (unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta -> log(theta) (external to internal)
            - "backward": log(theta) -> theta (internal to external)
            - "forward jacobian": derivative of forward transformation
            - "backward jacobian": 1 / derivative of forward transformation

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
            theta_scaled = theta  
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the Inverse Gamma distribution
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
            log p(θ) = C - (α + 1) * log(θ) - β / θ
        where C is the normalizing constant, α is the shape parameter,
        and β is the rate parameter.

        Raises
        ------
        ValueError
            If theta is not positive (implicitly through log computation).
        """

        log_prior = (
            self.normalizing_constant
            - (self.alpha + 1) * xp.log(theta)
            - self.beta / theta
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
    print("Testing Gaussian Quadrature with Inverse Gamma Prior Rescaling")
    print("=" * 80)

    # Create a inverse gamma prior configuration
    alpha_values = [1.0, 3.0, 5.0]
    beta_values = [0.5, 1.0, 2.0]

    for alpha, beta in zip(alpha_values, beta_values):
        print(f"\nTesting alpha={alpha}, beta={beta}")
        config = InverseGammaPriorHyperparametersConfig(alpha=alpha, beta=beta)
        inverse_gamma_prior = InverseGammaPriorHyperparameters(config=config)
        
        ## compare against scipy implementation
        from scipy.stats import invgamma
        
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print("Comparing log prior evaluations with scipy.stats.invgamma:")
        for val in test_values:
            logp_dalia = inverse_gamma_prior.evaluate_log_prior(val)
            logp_scipy = invgamma.logpdf(val, a=alpha, scale=beta)
            print(f"  θ = {val:4.1f}: DALIA logp = {logp_dalia:.6f}, "
                f"scipy logp = {logp_scipy:.6f}, diff = {abs(logp_dalia - logp_scipy):.2e}")
            if abs(logp_dalia - logp_scipy) > 1e-6:
                raise ValueError("Log prior evaluation does not match scipy implementation.")
    
    print()
    print("All tests passed!")
    
