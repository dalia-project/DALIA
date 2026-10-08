# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import sp, xp
from dalia.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.utils.link_functions import scaled_logit


class BetaPriorHyperparameters(PriorHyperparameters):
    """
    Scaled beta prior hyperparameters with support (lower, upper).

    The hyperparameter theta lives on the support (lower, upper) and

        u = (theta - lower) / (upper - lower) ~ Beta(alpha, beta).

    With the default support (0, 1) this is the standard beta distribution.
    The log density in theta is the beta log density in u minus
    log(upper - lower), the Jacobian of the affine map.

    The internal (unconstrained) scale is the scaled logit of u.
    """

    def __init__(
        self,
        config: BetaPriorHyperparametersConfig,
    ) -> None:
        """Initializes the beta prior hyperparameters."""
        super().__init__(config)

        self.alpha: float = config.alpha
        self.beta: float = config.beta

        self.lower, self.upper = (float(v) for v in config.support)
        self.width: float = self.upper - self.lower

        self.log_beta: float = float(
            sp.special.gammaln(self.alpha)
            + sp.special.gammaln(self.beta)
            - sp.special.gammaln(self.alpha + self.beta)
        )

    def _to_unit(self, theta):
        """Map theta from (lower, upper) to (0, 1)."""
        return (theta - self.lower) / self.width

    def rescale_hyperparameters_to_internal(self, theta, direction):

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
        elif direction == "backward_jacobian":
            # d theta / d internal = d theta / d u * d u / d internal
            theta_scaled = self.width * scaled_logit(
                self._to_unit(theta), direction="backward_jacobian"
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        u = self._to_unit(theta)

        if u <= 0 or u >= 1:
            raise ValueError(
                f"Beta prior is defined on ({self.lower}, {self.upper}), got theta: {theta}"
            )

        log_prior = (
            (self.alpha - 1) * xp.log(u)
            + (self.beta - 1) * xp.log(1 - u)
            - self.log_beta
            - xp.log(self.width)
        )

        return log_prior
