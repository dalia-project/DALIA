# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
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
    type: Literal["spatio_temporal", "spatial", "regression", "brainiac", "ar1", "ar2"] = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class RegressionSubModelConfig(SubModelConfig):
    n_fixed_effects: Annotated[int, Field(strict=True, ge=1)] = 1
    fixed_effects_prior_precision: float = 0.001

    def read_hyperparameters(self):
        return xp.array([]), []


class AR1SubModelConfig(SubModelConfig):

    ## prior on phi
    phi: float = None  # AR(1) coefficient
    phi_scaled: float = None
    ph_phi: PriorHyperparametersConfig = None
    ## check that phi is between -1 and 1 (use pc prior)
    # check inla.doc("pc.cor1")

    ## either define tau or sigma2
    tau: float = None  # Precision
    # sigma2: float = None  # Marginal variance
    
    
    ph_tau: PriorHyperparametersConfig = None
    # ph_sigma2: PriorHyperparametersConfig = None

    def read_hyperparameters(self):

        # input of phi is in (0,1), rescale to -/+ INF
        #self.phi_scaled = scaled_logit(self.phi, direction="forward")
        theta = xp.array([self.phi, self.tau])
        #theta_internal = xp.array([self.phi, self.tau])
        theta_keys = ["phi", "tau"]

        return theta, theta_keys


class AR2SubModelConfig(SubModelConfig):

    ## The AR(2) process is parametrized through its partial autocorrelations
    ## (pacf1, pacf2), each in (-1, 1), which guarantees stationarity.
    ## The AR coefficients follow as phi2 = pacf2 and phi1 = pacf1 * (1 - pacf2).
    ## Use a beta prior with support (-1, 1) to cover the full range; the
    ## default beta support (0, 1) restricts the pacf to positive values.
    pacf1: float = None  # first partial autocorrelation (= lag-1 autocorrelation)
    pacf2: float = None  # second partial autocorrelation (= phi2)
    ph_pacf1: PriorHyperparametersConfig = None
    ph_pacf2: PriorHyperparametersConfig = None

    ## marginal precision of the process
    tau: float = None  # Precision
    ph_tau: PriorHyperparametersConfig = None

    def read_hyperparameters(self):

        theta = xp.array([self.pacf1, self.pacf2, self.tau])
        theta_keys = ["pacf1", "pacf2", "tau"]

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
    if model_type == "ar1":
        config["ph_tau"] = parse_priorhyperparameters_config(config["ph_tau"])
        config["ph_phi"] = parse_priorhyperparameters_config(config["ph_phi"])
        return AR1SubModelConfig(**config)
    if model_type == "ar2":
        config["ph_tau"] = parse_priorhyperparameters_config(config["ph_tau"])
        config["ph_pacf1"] = parse_priorhyperparameters_config(config["ph_pacf1"])
        config["ph_pacf2"] = parse_priorhyperparameters_config(config["ph_pacf2"])
        return AR2SubModelConfig(**config)
    raise ValueError(f"Unknown submodel type: {model_type}")
