# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from typing_extensions import Annotated

from dalia.__init__ import ArrayLike, xp
from dalia.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
    GaussianMVNPriorHyperparametersConfig,
    PriorHyperparametersConfig,
)
from dalia.configs.priorhyperparameters_config import (
    parse_config as parse_priorhyperparameters_config,
)

class SubModelConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Input folder for this specific submodel
    input_dir: str = None
    type: Literal["spatio_temporal", "spatial", "regression", "brainiac", "ar"] = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class RegressionSubModelConfig(SubModelConfig):
    n_fixed_effects: Annotated[int, Field(strict=True, ge=1)] = 1
    fixed_effects_prior_precision: float = 0.001

    def read_hyperparameters(self):
        return xp.array([]), []


class ARSubModelConfig(SubModelConfig):

    ## Stationary AR(p) process of arbitrary order p >= 1, parametrized through
    ## its partial autocorrelations pacf[0], ..., pacf[p-1], each in (-1, 1),
    ## which guarantees stationarity. The AR coefficients follow from the
    ## Durbin-Levinson recursion.
    ## Use beta priors with support (-1, 1) to cover the full range; the
    ## default beta support (0, 1) restricts a pacf to positive values.
    order: Annotated[int, Field(strict=True, ge=1)]
    pacf: list[float]  # partial autocorrelations, one per lag
    ph_pacf: list[PriorHyperparametersConfig]  # one prior per pacf

    ## marginal precision of the process, Var(x_t) = 1 / tau
    ## (not the innovation precision)
    tau: float  # Precision
    ph_tau: PriorHyperparametersConfig

    @model_validator(mode="after")
    def _check_hyperparameters(self):
        if len(self.pacf) != self.order:
            raise ValueError(
                f"AR({self.order}) requires {self.order} partial autocorrelations, "
                f"got {len(self.pacf)}."
            )
        if len(self.ph_pacf) != self.order:
            raise ValueError(
                f"AR({self.order}) requires one prior per partial autocorrelation "
                f"({self.order}), got {len(self.ph_pacf)}."
            )
        for k, (pacf, ph_pacf) in enumerate(zip(self.pacf, self.ph_pacf), start=1):
            # also rejects nan and +/- inf
            if not -1.0 < pacf < 1.0:
                raise ValueError(f"pacf{k} must be in (-1, 1), got {pacf}.")
            if isinstance(ph_pacf, BetaPriorHyperparametersConfig):
                lower, upper = ph_pacf.support
                if not lower < pacf < upper:
                    raise ValueError(
                        f"pacf{k} = {pacf} is outside the support "
                        f"({lower}, {upper}) of its beta prior."
                    )
        if not 0.0 < self.tau < float("inf"):
            raise ValueError(f"tau must be finite and positive, got {self.tau}.")
        return self

    def read_hyperparameters(self):

        theta = xp.array([*self.pacf, self.tau])
        theta_keys = [f"pacf{k}" for k in range(1, self.order + 1)] + ["tau"]

        return theta, theta_keys


class SpatioTemporalSubModelConfig(SubModelConfig):
    spatial_domain_dimension: PositiveInt = 2

    # --- Model hyperparameters in the interpretable scale ---
    r_s: float = None  # Spatial range
    r_t: float = None  # Temporal range
    sigma_st: float = None  # Spatio-temporal variation

    ph_s: PriorHyperparametersConfig = None
    ph_t: PriorHyperparametersConfig = None
    ph_st: PriorHyperparametersConfig = None

    manifold: Literal["plane", "sphere"] = "plane"

    def read_hyperparameters(self):
        theta = xp.array([self.r_s, self.r_t, self.sigma_st])
        theta_keys = ["r_s", "r_t", "sigma_st"]

        return theta, theta_keys


class SpatialSubModelConfig(SubModelConfig):
    spatial_domain_dimension: PositiveInt = 2

    # --- Model hyperparameters in the interpretable scale ---
    r_s: float = None  # Spatial range
    sigma_e: float = None  # Spatial variation

    ph_s: PriorHyperparametersConfig = None
    ph_e: PriorHyperparametersConfig = None

    def read_hyperparameters(self):
        theta = xp.array([self.r_s, self.sigma_e])
        theta_keys = ["r_s", "sigma_e"]

        return theta, theta_keys


class TemporalSubModelConfig(SubModelConfig): ...


class BrainiacSubModelConfig(SubModelConfig):
    # --- Hyperparameters ---
    h2: float = None
    h2_scaled: float = None
    alpha: list[float] = None

    # --- Prior hyperparameters ---
    ph_h2: BetaPriorHyperparametersConfig = None
    ph_alpha: GaussianMVNPriorHyperparametersConfig = None

    def read_hyperparameters(self):
        theta = xp.concatenate([xp.array([self.h2]), xp.array(self.alpha)])
        theta_keys = ["h2"] + [f"alpha_{i}" for i in range(len(self.alpha))]

        return theta, theta_keys



def parse_config(config: dict | str) -> SubModelConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)
    model_type = config.get("type")
    if model_type == "spatio_temporal":
        config["ph_s"] = parse_priorhyperparameters_config(config["ph_s"])
        config["ph_t"] = parse_priorhyperparameters_config(config["ph_t"])
        config["ph_st"] = parse_priorhyperparameters_config(config["ph_st"])
        return SpatioTemporalSubModelConfig(**config)
    if model_type == "spatial":
        config["ph_s"] = parse_priorhyperparameters_config(config["ph_s"])
        config["ph_e"] = parse_priorhyperparameters_config(config["ph_e"])
        return SpatialSubModelConfig(**config)
    if model_type == "regression":
        return RegressionSubModelConfig(**config)
    if model_type == "brainiac":
        config["ph_h2"] = parse_priorhyperparameters_config(config["ph_h2"])
        config["ph_alpha"] = parse_priorhyperparameters_config(config["ph_alpha"])
        return BrainiacSubModelConfig(**config)
    if model_type == "ar":
        config["ph_tau"] = parse_priorhyperparameters_config(config["ph_tau"])
        ph_pacf = config["ph_pacf"]
        if not isinstance(ph_pacf, (list, tuple)):
            raise ValueError(
                "ph_pacf must be a list with one prior per partial autocorrelation."
            )
        config["ph_pacf"] = [parse_priorhyperparameters_config(ph) for ph in ph_pacf]
        return ARSubModelConfig(**config)
    raise ValueError(f"Unknown submodel type: {model_type}")
