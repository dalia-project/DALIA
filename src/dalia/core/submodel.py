# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.submodels_config import SubModelConfig


class SubModel(ABC):
    """Abstract core class for statistical models."""

    def __init__(
        self,
        config: SubModelConfig,
    ) -> None:
        """Initializes the model."""
        self.config = config
        self.input_path = Path(config.input_dir)
        self.submodel_type = config.type

        # --- by default, not intrinsic 
        # handle how intrinsic models with constraints get handled later
        self.intrinsic: bool = False
        # --- model constraints come from config
        self.has_constraints: bool = config.has_constraints
        
        self.constraints_D = None
        self.constraints_e = None

        # --- Load design matrix
        try:
            a: spmatrix = load_npz(self.input_path.joinpath("a.npz"))
            self.a = sp.sparse.csc_matrix(a)
        except FileNotFoundError:
            # check if dense a matrix exists
            try:
                a: NDArray = np.load(self.input_path.joinpath("a.npy"))
                if xp == np:
                    self.a: NDArray = a
                else:
                    self.a: NDArray = xp.array(a)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No design matrix found. Please provide a valid design matrix."
                )

        self.n_latent_parameters: int = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            x_initial: NDArray = np.load(self.input_path.joinpath("x.npy"))
            if xp == np:
                self.x_initial: NDArray = x_initial
            else:
                self.x_initial: NDArray = xp.array(x_initial)
        except FileNotFoundError:
            self.x_initial: NDArray = xp.zeros((self.a.shape[1]), dtype=float)

        # --- Initialize constraints if they exist
        # TODO: fix cupy compatibility later
        if self.has_constraints:
            try:
                # first check if constraints passed through config
                if self.config.constraints_D is not None and self.config.constraints_e is not None:
                    self.constraints_D = self.config.constraints_D
                    self.constraints_e = self.config.constraints_e
                else: 
                    self.constraints_D: NDArray = np.load(self.input_path.joinpath("constraints_D.npy"))
                    self.constraints_e: NDArray = np.load(self.input_path.joinpath("constraints_e.npy"))
                                        
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Constraints specified in config but constraint files not found."
                )
                
            # check dimensions match 
            if self.constraints_D.shape[1] != self.n_latent_parameters:
                raise ValueError(
                    f"Number of columns in constraints_D ({self.constraints_D.shape[1]}) "
                    f"does not match number of latent parameters ({self.n_latent_parameters})."
                )
            if self.constraints_D.shape[0] != self.constraints_e.shape[0]:
                raise ValueError(
                    f"Number of rows in constraints_D ({self.constraints_D.shape[0]}) "
                    f"does not match number of rows in constraints_e ({self.constraints_e.shape[0]})."
                )



    @abstractmethod
    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""
        ...

    def load_a_predict(self) -> sp.sparse.csc_matrix:
        """Load the design matrix for prediction."""
        self.a_predict: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
            load_npz(self.input_path.joinpath("apr.npz"))
        )

        # check that number of columns is the same as in a
        if self.a_predict.shape[1] != self.a.shape[1]:
            raise ValueError(
                f"Number of columns in a_predict ({self.a_predict.shape[1]}) "
                f"does not match number of columns in a ({self.a.shape[1]})."
            )

        return self.a_predict
