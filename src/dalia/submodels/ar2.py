# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

import numpy as np

from dalia import sp, xp
from dalia.configs.submodels_config import AR2SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class AR2SubModel(SubModel):
    """Fit an AR(2) model.

    The process x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + eps_t is parametrized
    through its partial autocorrelations (pacf1, pacf2), each in (-1, 1), and
    its marginal precision tau. The AR coefficients follow as

    phi1 = pacf1 * (1 - pacf2),
    phi2 = pacf2,

    which guarantees a stationary process for any admissible (pacf1, pacf2).
    """

    def __init__(
        self,
        config: AR2SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        # check that dimensions match
        if self.n_latent_parameters < 4:
            raise ValueError(
                "AR(2) submodel requires at least 4 latent parameters, "
                f"got {self.n_latent_parameters}."
            )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """
        Construct the prior precision matrix.

        The precision of the stationary AR(2) is obtained from the conditional
        densities p(x_t | x_{t-1}, x_{t-2}) for t >= 3 combined with the exact
        stationary distribution of (x_1, x_2).
        """

        # kwargs expects hyperparameters in external scale
        pacf1 = kwargs.get("pacf1")
        pacf2 = kwargs.get("pacf2")
        tau = kwargs.get("tau")

        # partial autocorrelations -> AR coefficients
        phi2 = pacf2
        phi1 = pacf1 * (1 - pacf2)

        # marginal variance -> innovation variance
        s2 = 1 / tau
        denom = s2 * (1 + phi2) * ((1 - phi2) ** 2 - phi1**2) / (1 - phi2)

        n = self.n_latent_parameters

        diag = [(1 + phi1**2 + phi2**2) / denom] * n
        diag[0] = diag[-1] = 1 / denom
        diag[1] = diag[-2] = (1 + phi1**2) / denom

        off_diag_1 = [-phi1 * (1 - phi2) / denom] * (n - 1)
        off_diag_1[0] = off_diag_1[-1] = -phi1 / denom

        off_diag_2 = [-phi2 / denom] * (n - 2)

        Q_prior = sp.sparse.diags(
            [off_diag_2, off_diag_1, diag, off_diag_1, off_diag_2],
            [-2, -1, 0, 1, 2],
        )

        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

        return Q_prior.tocoo()

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["pacf1", f"{self.config.pacf1:.3f}"],
            ["pacf2", f"{self.config.pacf2:.3f}"],
            ["tau", f"{self.config.tau:.3f}"],
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
