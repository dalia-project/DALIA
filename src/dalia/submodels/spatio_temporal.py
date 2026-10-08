# Copyright 2024-2025 DALIA authors. All rights reserved.

import math

import numpy as np
from scipy.sparse import csc_matrix, load_npz, spmatrix
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.submodels_config import SpatioTemporalSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class SpatioTemporalSubModel(SubModel):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        config: SpatioTemporalSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.sigma_st: float = config.sigma_st

        # Load spatial_matrices
        c0: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("c0.npz")))
        g1: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("g1.npz")))
        g2: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("g2.npz")))
        g3: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("g3.npz")))

        # Load temporal_matrices
        m0: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("m0.npz")))
        m1: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("m1.npz")))
        m2: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("m2.npz")))

        if xp == np:
            self.c0: spmatrix = c0
            self.g1: spmatrix = g1
            self.g2: spmatrix = g2
            self.g3: spmatrix = g3

            self.m0: spmatrix = m0
            self.m1: spmatrix = m1
            self.m2: spmatrix = m2
        else:
            self.c0: sp.sparse.spmatrix = sp.sparse.csc_matrix(c0)
            self.g1: sp.sparse.spmatrix = sp.sparse.csc_matrix(g1)
            self.g2: sp.sparse.spmatrix = sp.sparse.csc_matrix(g2)
            self.g3: sp.sparse.spmatrix = sp.sparse.csc_matrix(g3)

            self.m0: sp.sparse.spmatrix = sp.sparse.csc_matrix(m0)
            self.m1: sp.sparse.spmatrix = sp.sparse.csc_matrix(m1)
            self.m2: sp.sparse.spmatrix = sp.sparse.csc_matrix(m2)

        self._check_dimensions_spatial_matrices()
        self._check_dimensions_temporal_matrices()
        self._check_matrix_symmetry()

        self.ns: int = self.c0.shape[0]  # Number of spatial nodes in the mesh
        self.nt: int = self.m0.shape[0]  # Number of temporal nodes in the mesh

        # Check that design_matrix shape match spatio-temporal fields
        assert (
            self.n_latent_parameters == self.ns * self.nt
        ), f"Design matrix has incorrect number of columns. \n    n_latent_parameters: {self.n_latent_parameters}\n    ns:{self.ns} x nt:{self.nt} = {self.ns * self.nt}"

        self.manifold: str = config.manifold

    def _check_dimensions_spatial_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Spatial matrices checks
        assert self.c0.shape[0] == self.c0.shape[1], "Spatial matrix c0 is not square."
        assert self.g1.shape[0] == self.g1.shape[1], "Spatial matrix g1 is not square."
        assert self.g2.shape[0] == self.g2.shape[1], "Spatial matrix g2 is not square."
        assert self.g3.shape[0] == self.g3.shape[1], "Spatial matrix g3 is not square."
        assert (
            self.c0.shape == self.g1.shape == self.g2.shape == self.g3.shape
        ), "Dimensions of spatial matrices do not match."

    def _check_dimensions_temporal_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Temporal matrices checks
        assert self.m0.shape[0] == self.m0.shape[1], "Temporal matrix m0 is not square."
        assert self.m1.shape[0] == self.m1.shape[1], "Temporal matrix m1 is not square."
        assert self.m2.shape[0] == self.m2.shape[1], "Temporal matrix m2 is not square."
        assert (
            self.m0.shape == self.m1.shape == self.m2.shape
        ), "Dimensions of temporal matrices do not match."

    def _check_matrix_symmetry(self) -> None:
        """Check the symmetry of the matrices."""
        diff = self.c0 - self.c0.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.g1 - self.g1.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.g2 - self.g2.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.g3 - self.g3.T
        assert np.all(np.abs(diff.data) < 1e-10), "Spatial matrix c0 is not symmetric."

        diff = self.m0 - self.m0.T
        assert np.all(np.abs(diff.data) < 1e-10), "Temporal matrix m0 is not symmetric."

        diff = self.m1 - self.m1.T
        assert np.all(np.abs(diff.data) < 1e-10), "Temporal matrix m1 is not symmetric."

        diff = self.m2 - self.m2.T
        assert np.all(np.abs(diff.data) < 1e-10), "Temporal matrix m2 is not symmetric."

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        gamma_s, gamma_t, gamma_st = self._interpretable2compute(
            r_s=kwargs.get("r_s"),
            r_t=kwargs.get("r_t"),
            sigma_st=kwargs.get("sigma_st", self.sigma_st),
            dim_spatial_domain=2,
        )

        exp_gamma_s = xp.exp(gamma_s)
        exp_gamma_t = xp.exp(gamma_t)
        exp_gamma_st = xp.exp(gamma_st)

        q1s = pow(exp_gamma_s, 2) * self.c0 + self.g1
        q2s = (
            pow(exp_gamma_s, 4) * self.c0 + 2 * pow(exp_gamma_s, 2) * self.g1 + self.g2
        )
        q3s = (
            pow(exp_gamma_s, 6) * self.c0
            + 3 * pow(exp_gamma_s, 4) * self.g1
            + 3 * pow(exp_gamma_s, 2) * self.g2
            + self.g3
        )

        # withsparseKroneckerProduct", color_id=0):
        Q_prior: sp.sparse.spmatrix = sp.sparse.csc_matrix(
            pow(exp_gamma_st, 2)
            * (
                sp.sparse.kron(self.m0, q3s)
                + exp_gamma_t * sp.sparse.kron(self.m1, q2s)
                + pow(exp_gamma_t, 2) * sp.sparse.kron(self.m2, q1s)
            )
        )
        # TODO: csc()
        return Q_prior.tocoo()

    def _interpretable2compute(
        self, r_s: float, r_t: float, sigma_st: float, dim_spatial_domain: int = 2
    ) -> tuple:
        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        # Assumes alphas as fixed for now
        alpha_s = 2
        alpha_t = 1
        alpha_e = 1

        # Implicit assumption that spatial domain is 2D
        alpha = alpha_e + alpha_s * (alpha_t - 0.5)

        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        gamma_s = 0.5 * xp.log(8 * nu_s) - r_s
        gamma_t = r_t - 0.5 * xp.log(8 * nu_t) + alpha_s * gamma_s

        if self.manifold == "sphere":
            cR_t = sp.special.gamma(nu_t) / (
                sp.special.gamma(alpha_t) * pow(4 * math.pi, 0.5)
            )
            c_s = 0.0
            for k in range(50):
                c_s += (2.0 * k + 1.0) / (
                    4.0 * math.pi * pow(pow(xp.exp(gamma_s), 2) + k * (k + 1), alpha)
                )
            gamma_st = 0.5 * xp.log(cR_t) + 0.5 * xp.log(c_s) - 0.5 * gamma_t - sigma_st

        elif self.manifold == "plane":
            c1_scaling_constant = pow(4 * math.pi, 1.5)
            c1 = (
                sp.special.gamma(nu_t)
                * sp.special.gamma(nu_s)
                / (
                    sp.special.gamma(alpha_t)
                    * sp.special.gamma(alpha)
                    * c1_scaling_constant
                )
            )
            gamma_st = 0.5 * xp.log(c1) - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
        else:
            raise ValueError("Manifold not supported: ", self.manifold)

        return gamma_s, gamma_t, gamma_st

    def convert_theta_from_model2interpret(
        self,
        gamma_s: float,
        gamma_t: float,
        gamma_st: float,
        dim_spatial_domain: int = 2,
    ) -> tuple:
        """Convert theta from model scale to interpretable scale."""

        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        alpha_t = 1
        alpha_s = 2
        alpha_e = 1

        alpha = alpha_e + alpha_s * (alpha_t - 0.5)
        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        exp_gamma_s = xp.exp(gamma_s)
        exp_gamma_t = xp.exp(gamma_t)
        exp_gamma_st = xp.exp(gamma_st)

        r_s = xp.log(xp.sqrt(8 * nu_s) / exp_gamma_s)
        r_t = xp.log(exp_gamma_t * xp.sqrt(8 * nu_t) / (pow(exp_gamma_s, alpha_s)))

        if self.manifold == "sphere":
            c_r_t = sp.special.gamma(alpha_t - 0.5) / (
                sp.special.gamma(alpha_t) * pow(4 * math.pi, 0.5)
            )
            c_s = 0.0
            for k in range(50):
                c_s += (2.0 * k + 1) / (
                    4 * math.pi * pow(pow(exp_gamma_s, 2) + k * (k + 1), alpha)
                )
            sigma_st = xp.log(
                xp.sqrt(c_r_t * c_s) / (exp_gamma_st * xp.sqrt(exp_gamma_t))
            )

        elif self.manifold == "plane":
            c1_scaling_const = pow(4 * math.pi, dim_spatial_domain / 2.0) * pow(
                4 * math.pi, 0.5
            )
            c1 = (
                sp.special.gamma(nu_t)
                * sp.special.gamma(nu_s)
                / (
                    sp.special.gamma(alpha_t)
                    * sp.special.gamma(alpha)
                    * c1_scaling_const
                )
            )
            sigma_st = xp.log(
                xp.sqrt(c1)
                / (
                    (exp_gamma_st * xp.sqrt(exp_gamma_t))
                    * pow(exp_gamma_s, alpha - dim_spatial_domain / 2)
                )
            )

        else:
            raise ValueError("Manifold not supported: ", self.manifold)

        return r_s, r_t, sigma_st

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Number of Spatial Nodes", self.ns],
            ["Number of Temporal Nodes", self.nt],
            ["Manifold", self.manifold.capitalize()],
            ["Spatial Range (r_s)", f"{self.config.r_s:.3f}"],
            ["Temporal Range (r_t)", f"{self.config.r_t:.3f}"],
            ["Spatio-temporal Variation (sigma_st)", f"{self.sigma_st:.3f}"],
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
