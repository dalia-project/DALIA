# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.constraints_config import (
    LinearConstraintConfig,
    SumToZeroConstraintConfig,
)
from dalia.configs.submodels_config import RW2SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RW2SubModel(SubModel):
    """Fit a random walk of order 2 on the locations s_1 < ... < s_n, the
    Galerkin approximation of the integrated Wiener process d^2 x / dt^2 = dW / dt
    (Lindgren & Rue 2008, "On the second-order random walk model for irregular
    locations", Section 2).

    Its precision matrix is Q = tau * R, with structure matrix R = H^T A^{-1} H:
    H is tridiagonal with interior rows (1 / d_{i-1}, -(1 / d_{i-1} + 1 / d_i),
    1 / d_i) and zero first and last rows, d_i = s_{i+1} - s_i the gaps, and A
    the diagonal matrix of trapezoid weights (d_{i-1} + d_i) / 2. On a regular
    grid, d_i = 1, R = D^T D with D the matrix of second differences
    (Rue & Held 2005, Section 3.4.1). The model is intrinsic: R has rank n - 2,
    the constant and the locations s are in its null space.

    The locations are read from `locations.npy` in the input directory, one
    per latent parameter, strictly increasing. Without the file the grid is
    regular, s = 0, ..., n - 1. R-INLA: f(..., model = "rw2", values = s).
    """

    intrinsic: bool = True

    def __init__(
        self,
        config: RW2SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        n = self.n_latent_parameters
        if n < 3:
            raise ValueError(
                f"RW2 submodel requires at least 3 latent parameters, got {n}."
            )

        self.load_locations()

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix Q = tau * R."""

        # kwargs expects hyperparameters in external scale
        tau = kwargs.get("tau")

        # R = H^T A^{-1} H, pentadiagonal, from the gaps d_i = s_{i+1} - s_i
        d = xp.diff(self.locations)

        # H: first and last rows are zero, interior row i holds
        # H[i, i-1] = 1 / d_{i-1}, H[i, i] = -(1 / d_{i-1} + 1 / d_i), H[i, i+1] = 1 / d_i
        lower = xp.concatenate([1.0 / d[:-1], xp.zeros(1)])
        main = xp.concatenate([xp.zeros(1), -(1.0 / d[:-1] + 1.0 / d[1:]), xp.zeros(1)])
        upper = xp.concatenate([xp.zeros(1), 1.0 / d[1:]])
        H = sp.sparse.diags([lower, main, upper], [-1, 0, 1])

        # A: diagonal of trapezoid weights, A_11 = d_1 / 2, A_ii = (d_{i-1} + d_i) / 2,
        # A_nn = d_{n-1} / 2
        weights = xp.concatenate([d[:1] / 2.0, (d[:-1] + d[1:]) / 2.0, d[-1:] / 2.0])
        A_inv = sp.sparse.diags(1.0 / weights)

        Q_prior = tau * (H.T @ A_inv @ H)

        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

        return Q_prior.tocoo()

    def _construct_identifiability_constraints(self) -> list:
        """The null space of R is spanned by the constant and the locations s,
        so the prior determines neither the level nor the slope of the field:
        sum(x) = 0 and sum(s x) = 0."""
        return [
            SumToZeroConstraintConfig(),
            LinearConstraintConfig(C=[self.locations.tolist()]),
        ]

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["tau", f"{self.config.tau:.3f}"],
            ["Diagonal", f"{self.config.diagonal:.1e}"],
            ["Locations", "regular grid" if self.regular_grid else "irregular"],
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
