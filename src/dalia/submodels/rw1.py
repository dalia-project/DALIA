# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.constraints_config import SumToZeroConstraintConfig
from dalia.configs.submodels_config import RW1SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RW1SubModel(SubModel):
    """Fit a random walk of order 1 on the locations s_1 < ... < s_n, the
    Wiener process observed at the locations: the increments are independent,
    x_{i+1} - x_i ~ N(0, d_i / tau) with d_i = s_{i+1} - s_i the gaps
    (Rue & Held 2005, "Gaussian Markov Random Fields: Theory and Applications",
    Section 3.3.1, "The first-order random walk model for irregular locations",
    pp. 97-99: gaps (3.25), density (3.26)).

    Its precision matrix is Q = tau * R, with structure matrix
    R = D^T diag(1 / d) D and D the matrix of first differences: tridiagonal
    with R_ii = 1 / d_{i-1} + 1 / d_i, R_{i,i+1} = -1 / d_i, and 1 / d_1,
    1 / d_{n-1} in the first and last diagonal entries (the explicit Q_ij of
    p. 99, boundary terms from a diffuse prior on W(0)). On a regular grid,
    d_i = 1, R = D^T D (equation (3.24)). The model is intrinsic: R has rank
    n - 1, the constant vector is in its null space.

    The locations are read from `locations.npy` in the input directory, one
    per latent parameter, strictly increasing. Without the file the grid is
    regular, s = 0, ..., n - 1. R-INLA: f(..., model = "rw1", values = s).
    """

    intrinsic: bool = True

    def __init__(
        self,
        config: RW1SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        n = self.n_latent_parameters
        if n < 2:
            raise ValueError(
                f"RW1 submodel requires at least 2 latent parameters, got {n}."
            )

        self.load_locations()

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix Q = tau * R."""

        # kwargs expects hyperparameters in external scale
        tau = kwargs.get("tau")

        # R = D^T diag(1 / d) D, tridiagonal, from the gaps d_i = s_{i+1} - s_i
        inv_d = 1.0 / xp.diff(self.locations)

        # R_ii = 1 / d_{i-1} + 1 / d_i, with R_11 = 1 / d_1 and R_nn = 1 / d_{n-1}
        diag = xp.concatenate([inv_d[:1], inv_d[:-1] + inv_d[1:], inv_d[-1:]])
        off_diag = -inv_d

        Q_prior = tau * sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1])

        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

        return Q_prior.tocoo()

    def _construct_identifiability_constraints(self) -> list:
        """The null space of R is the constant vector, so the prior does not
        determine the level of the field: sum(x) = 0."""
        return [SumToZeroConstraintConfig()]

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
