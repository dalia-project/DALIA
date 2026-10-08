# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from typing_extensions import Annotated

from dalia.__init__ import ArrayLike, xp
from dalia.configs.priorhyperparameters_config import (
    PriorHyperparametersConfig,
)
from dalia.configs.priorhyperparameters_config import (
    parse_config as parse_priorhyperparameters_config,
)


class ModelConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    # Input folder for this specific submodel
    type: Literal["coregional"] = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class CoregionalModelConfig(ModelConfig):
    n_models: PositiveInt = None

    sigmas: list[float] = None  # Sigmas
    lambdas: list[float] = None  # Lambdas

    ph_sigmas: list[PriorHyperparametersConfig] = None
    ph_lambdas: list[PriorHyperparametersConfig] = None

    @model_validator(mode="after")
    def check_n_models(self):
        assert self.n_models == 2 or self.n_models == 3, "n_models must be 2 or 3"
        return self

    @model_validator(mode="after")
    def check_hyperparameters_length(self):
        if self.n_models is not None:
            if self.sigmas is not None and len(self.sigmas) != self.n_models:
                raise ValueError(
                    f"Length of sigmas ({len(self.sigmas)}) does not match n_models ({self.n_models})"
                )
            n_lambdas = self.n_models * (self.n_models - 1) // 2
            if self.lambdas is not None and len(self.lambdas) != n_lambdas:
                raise ValueError(
                    f"Length of lambdas ({len(self.lambdas)}) does not match the required number of lambdas ({n_lambdas})"
                )
        return self

    @model_validator(mode="after")
    def check_prior_hyperparameters_length(self):
        if self.n_models is not None:
            if self.sigmas is not None and len(self.ph_sigmas) != len(self.sigmas):
                raise ValueError(
                    f"Length of sigmas prior hyperparameters ({len(self.ph_sigmas)}) does not match number of sigmas ({len(self.sigmas)})"
                )
            if self.lambdas is not None and len(self.ph_lambdas) != len(self.lambdas):
                raise ValueError(
                    f"Length of lambdas prior hyperparameters ({len(self.ph_lambdas)}) does not match number of lambdas ({len(self.lambdas)})"
                )
        return self

    def read_hyperparameters(self):
        theta = xp.array(self.sigmas + self.lambdas)
        theta_keys: list = []
        for i in range(self.n_models):
            theta_keys.append(f"sigma_{i}")
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                theta_keys.append(f"lambda_{i}_{j}")

        return theta, theta_keys


def parse_config(config: dict | str) -> ModelConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    type = config.get("type")
    if type == "coregional":
        for i in range(len(config["ph_sigmas"])):
            config["ph_sigmas"][i] = parse_priorhyperparameters_config(
                config["ph_sigmas"][i]
            )
        for i in range(len(config["ph_lambdas"])):
            config["ph_lambdas"][i] = parse_priorhyperparameters_config(
                config["ph_lambdas"][i]
            )
        return CoregionalModelConfig(**config)
    # Add more elif branches for other model types
    else:
        raise ValueError(f"Invalid submodel type: {type}")
