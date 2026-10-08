# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dalia.__init__ import ArrayLike, xp
from dalia.configs.priorhyperparameters_config import (
    PriorHyperparametersConfig,
)
from dalia.configs.priorhyperparameters_config import (
    parse_config as parse_prior_hyperparameters_config,
)


# --- LIKELIHOODS --------------------------------------------------------------
class LikelihoodConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "poisson", "binomial"] = None

    # TODO: cleaner way to let user fix hyperparameters
    fix_hyperparameters: bool = False
    prior_hyperparameters: PriorHyperparametersConfig = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class GaussianLikelihoodConfig(LikelihoodConfig):
    prec_o: float = None  # Observation precision

    def read_hyperparameters(self):
        if self.fix_hyperparameters:
            return xp.array([]), []
        else:
            theta = xp.array([self.prec_o])
            theta_keys = ["prec_o"]
            return theta, theta_keys


class PoissonLikelihoodConfig(LikelihoodConfig):
    input_dir: str = None

    def read_hyperparameters(self):
        return xp.array([]), []


class BinomialLikelihoodConfig(LikelihoodConfig):
    input_dir: str = None
    link_function: Literal["sigmoid"] = "sigmoid"

    def read_hyperparameters(self):
        return xp.array([]), []


def parse_config(config: dict | str) -> LikelihoodConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    likelihood_type = config.get("type")
    if "prior_hyperparameters" in config:
        config["prior_hyperparameters"] = parse_prior_hyperparameters_config(
            config["prior_hyperparameters"]
        )

    if likelihood_type == "gaussian":
        return GaussianLikelihoodConfig(**config)
    elif likelihood_type == "poisson":
        return PoissonLikelihoodConfig(**config)
    elif likelihood_type == "binomial":
        return BinomialLikelihoodConfig(**config)
    else:
        raise ValueError(f"Unknown likelihood config type: {likelihood_type}")
