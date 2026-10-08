# Copyright 2024-2025 DALIA authors. All rights reserved.

import math

import numpy as np
from scipy.sparse import csc_matrix, load_npz, spmatrix
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.submodels_config import SpatialSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class SpatialSubModel(SubModel):
    """Fit a spatial model."""

    def __init__(
        self,
        config: SpatialSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.sigma_e = self.config.sigma_e

        # Load spatial_matrices
        c0: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("c0.npz")))
        g1: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("g1.npz")))
        g2: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("g2.npz")))

        if xp == np:
            self.c0: spmatrix = c0
            self.g1: spmatrix = g1
            self.g2: spmatrix = g2
        else:
            self.c0: sp.sparse.spmatrix = sp.sparse.csc_matrix(c0)
            self.g1: sp.sparse.spmatrix = sp.sparse.csc_matrix(g1)
            self.g2: sp.sparse.spmatrix = sp.sparse.csc_matrix(g2)

        self._check_dimensions_spatial_matrices()

        self._check_matrix_symmetry()

        self.ns: int = self.c0.shape[0]  # Number of spatial nodes in the mesh

        # Check that design_matrix shape match spatio-temporal fields
        assert (
            self.n_latent_parameters == self.ns
        ), f"Design matrix has incorrect number of columns. \n    n_latent_parameters: {self.n_latent_parameters}\n    ns:{self.ns}"

    def _check_dimensions_spatial_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Spatial matrices checks
        assert self.c0.shape[0] == self.c0.shape[1], "Spatial matrix c0 is not square."
        assert self.g1.shape[0] == self.g1.shape[1], "Spatial matrix g1 is not square."
        assert self.g2.shape[0] == self.g2.shape[1], "Spatial matrix g2 is not square."
        assert (
            self.c0.shape == self.g1.shape == self.g2.shape  # == self.g3.shape
        ), "Dimensions of spatial matrices do not match."

    def _check_matrix_symmetry(self) -> None:
        """Check the symmetry of the matrices."""
        diff = self.c0 - self.c0.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.g1 - self.g1.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.g2 - self.g2.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        gamma_s, gamma_e = self._interpretable2compute(
            r_s=kwargs.get("r_s"),
            sigma_e=kwargs.get("sigma_e", self.sigma_e),
            dim_spatial_domain=2,
        )

        exp_gamma_s = xp.exp(gamma_s)
        exp_gamma_e = xp.exp(gamma_e)

        q2s = (
            pow(exp_gamma_s, 4) * self.c0 + 2 * pow(exp_gamma_s, 2) * self.g1 + self.g2
        )
        # Leave this here for now to be able to do higher-order later
        # q3s = (
        #     pow(exp_gamma_s, 6) * self.c0
        #     + 3 * pow(exp_gamma_s, 4) * self.g1
        #     + 3 * pow(exp_gamma_s, 2) * self.g2
        #     + self.g3
        # )

        Q_prior: sp.sparse.spmatrix = sp.sparse.csc_matrix(pow(exp_gamma_e, 2) * q2s)

        return Q_prior.tocoo()

    def _interpretable2compute(
        self, r_s: float, sigma_e: float, dim_spatial_domain: int = 2
    ) -> tuple:
        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        alpha = 2
        nu_s = alpha - dim_spatial_domain / 2
        gamma_s = 0.5 * np.log(8 * nu_s) - r_s
        gamma_e = 0.5 * (
            sp.special.gamma(nu_s)
            - (
                sp.special.gamma(alpha)
                + 0.5 * dim_spatial_domain * np.log(4 * np.pi)
                + 2 * nu_s * gamma_s
                + 2 * sigma_e
            )
        )

        return gamma_s, gamma_e

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Number of Spatial Nodes", self.ns],
            ["Spatial Range (r_s)", f"{self.config.r_s:.3f}"],
            ["Spatial Variation (sigma_e)", f"{self.sigma_e:.3f}"],
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
