# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.sparse import spmatrix
from typing_extensions import Annotated

from dalia.__init__ import NDArray


# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: Literal[
        "gaussian", "penalized_complexity", "beta", "gaussian_mvn", "gamma", "inverse_gamma"
    ] = None


class GaussianPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: float = 0.0
    precision: Annotated[float, Field(strict=True, gt=0)] = 0.5


class GaussianMVNPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: NDArray = None
    precision: spmatrix = None


class PenalizedComplexityPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    u: float = None

    # Generalized formula:
    # lambda = - log(alpha) * pow(u, c_l)
    #
    # log_prior = a + b + c
    # a = log(lambda)
    # b = -lambda * exp(c_b * r)
    # c = c_c * r


class BetaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    beta: float = None
    # Support (lower, upper) of the scaled beta distribution. The standard
    # beta distribution on (0, 1) is the default; e.g. (-1, 1) for correlation
    # type hyperparameters such as the partial autocorrelations of an AR(p).
    support: tuple[float, float] = (0.0, 1.0)

    @field_validator("support")
    @classmethod
    def _check_support(cls, value):
        lower, upper = value
        if not lower < upper:
            raise ValueError(
                f"Beta prior support must satisfy lower < upper, got {value}"
            )
        return value


class GammaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    beta: float = None
    
class InverseGammaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    beta: float = None


def parse_config(config: dict) -> PriorHyperparametersConfig:
    prior_type = config.get("type")
    if prior_type == "gaussian":
        return GaussianPriorHyperparametersConfig(**config)
    if prior_type == "gaussian_mvn":
        return GaussianMVNPriorHyperparametersConfig(**config)
    if prior_type == "penalized_complexity":
        return PenalizedComplexityPriorHyperparametersConfig(**config)
    if prior_type == "beta":
        return BetaPriorHyperparametersConfig(**config)
    if prior_type == "gamma":
        return GammaPriorHyperparametersConfig(**config)
    if prior_type == "inverse_gamma":
        return InverseGammaPriorHyperparametersConfig(**config)
    raise ValueError(f"Unknown prior hyperparameters config type: {prior_type}")
