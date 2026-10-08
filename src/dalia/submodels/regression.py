# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp
from dalia.configs.submodels_config import RegressionSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class RegressionSubModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        config: RegressionSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.n_fixed_effects: int = config.n_fixed_effects
        self.fixed_effects_prior_precision: float = config.fixed_effects_prior_precision

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), f"Design matrix has {self.n_latent_parameters} columns, but expected {self.n_fixed_effects} columns."

        # --- Construct the prior precision matrix
        self.Q_prior: sp.sparse.coo_matrix = sp.sparse.coo_matrix(
            self.fixed_effects_prior_precision * sp.sparse.eye(self.n_fixed_effects)
        )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        return self.Q_prior

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Number of Fixed Effects", self.n_fixed_effects],
            ["Prior Precision of Fixed Effects", self.fixed_effects_prior_precision],
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
