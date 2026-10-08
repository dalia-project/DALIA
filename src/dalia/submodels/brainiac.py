# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from tabulate import tabulate

from dalia import NDArray, sp, xp
from dalia.configs.submodels_config import BrainiacSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header, scaled_logit


class BrainiacSubModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        config: BrainiacSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        print("Calling BrainiacSubModel.__init__")

        # Load covariates matrix "z"
        z: NDArray = np.load(self.input_path.joinpath("z.npy"))

        """ 
        -> We already load the "design" matrix that is called "a" in the submodel.py
        -> Is it the same? I assumed yes for now and so it gets loaded there.
        Also:
        If it's a numpy array -> it is stored as .npy
        If it's a sparse matrix -> it gets stored as .npz

        # Load projection matrix "a"
        a: NDArray = np.load(self.input_path.joinpath("a.npy")) 
        """

        if xp == np:
            self.z: NDArray = z
            # self.a: NDArray = a
        else:
            self.z: NDArray = xp.asarray(z)
            # self.a: NDArray = xp.asarray(a)

        self._check_dimensions_matrices()

    def _check_dimensions_matrices(self) -> None:
        """Check the dimensions of the model."""

        assert (
            self.z.shape[0] == self.a.shape[1]
        ), f"Numbers rows in z ({self.z.shape[0]}) must match number of columns in a ({self.a.shape[1]})."


    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""
        # Extract all alpha_x values and put them into an array
        alpha_keys = sorted([key for key in kwargs if key.startswith("alpha_")])
        alpha = xp.array([kwargs[key] for key in alpha_keys])
        h2 = kwargs.get("h2")

        # \Phi = 1 / \sum_k=1^B exp(Z^k \alpha) * diag(exp(Z_1 \alpha), exp(Z_2 \alpha), ... )
        exp_Z_alpha = xp.exp(self.z @ alpha)
        # print(exp_Z_alpha)
        sum_exp_Z_alpha = xp.sum(exp_Z_alpha)
        # print(sum_exp_Z_alpha)

        normalized_exp_Z_alpha = exp_Z_alpha / sum_exp_Z_alpha
        # print(normalized_exp_Z_alpha)

        h2_phi = h2 * normalized_exp_Z_alpha.flatten()
        Q_prior: sp.sparse.spmatrix = sp.sparse.diags(1 / h2_phi)

        return Q_prior.tocoo()

    def evaluate_likelihood(self, eta: NDArray, y: NDArray, **kwargs) -> float:
        n_observations = y.shape[0]

        h2 = kwargs.get("h2")
        if h2 == 1:
            raise ValueError("h2 is 1. Will lead to division by zero.")

        yEta = y - eta
        likelihood: float = (
            0.5 * -np.log(1 - h2) * n_observations - 0.5 / (1 - h2) * yEta.T @ yEta
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self, eta: NDArray, y: NDArray, **kwargs
    ) -> NDArray:
        h2 = kwargs.get("h2")

        if h2 == 1:
            raise ValueError("h2 is 1. Will lead to division by zero.")

        gradient = -1 / (1 - h2) * (eta - y)

        return gradient

    def evaluate_d_matrix(self, **kwargs) -> NDArray:
        h2 = kwargs.get("h2")

        if h2 == 1:
            raise ValueError("h2 is 1. Will lead to division by zero.")
        d_matrix = -1 / (1 - h2) * sp.sparse.eye(self.a.shape[0])

        return d_matrix

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["h2", f"{self.config.h2:.3f}"],
            ["alpha", [f"{a:.3f}" for a in self.config.alpha]],
        ]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )

        # Add the header title
        submodel_table = add_str_header(
            title=self.submodel_type.replace("_", " ").title(),
            table=submodel_table,
        )
        str_representation += submodel_table

        return str_representation
