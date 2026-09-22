# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate


from dalia import sp, xp
from dalia.configs.submodels_config import LKJSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class LKJSubModel(SubModel):
    """2D latent Gaussian with LKJ prior on correlation and independent variances."""

    def __init__(
        self,
        config: LKJSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        # Enforce 2D constraint
        assert (
            self.n_latent_parameters_core == 2
        ), f"LKJSubModel only supports 2D latent variables, got {self.n_latent_parameters}"

    def _construct_Q_prior_core(self, **kwargs) -> sp.sparse.coo_matrix:
        """
        Construct the 2x2 prior precision matrix Q from covariance structure.

        Covariance: Σ = [σ₁²      ρσ₁σ₂]
                        [ρσ₁σ₂     σ₂² ]

        Precision: Q = Σ⁻¹ = 1/(σ₁²σ₂²(1-ρ²)) [σ₂²      -ρσ₁σ₂]
                                                [-ρσ₁σ₂    σ₁²  ]
        """
        # kwargs expects hyperparameters in external scale
        sigma1 = kwargs.get("sigma1")
        sigma2 = kwargs.get("sigma2")
        rho = kwargs.get("rho")

        # Compute denominator: σ₁²σ₂²(1-ρ²)
        denom = sigma1**2 * sigma2**2 * (1 - rho**2)

        # Q matrix elements
        q00 = sigma2**2 / denom
        q11 = sigma1**2 / denom
        q01 = -rho * sigma1 * sigma2 / denom

        # Build sparse matrix
        # explicit 1-D xp arrays: cupyx coo_matrix rejects lists of 0-d arrays
        row = xp.array([0, 0, 1, 1])
        col = xp.array([0, 1, 0, 1])
        data = xp.stack(
            [xp.asarray(q, dtype=xp.float64) for q in (q00, q01, q01, q11)]
        )

        Q_prior = sp.sparse.coo_matrix((data, (row, col)), shape=(2, 2))

        return Q_prior
    
        # additional method to return LKJ covariance matrix given theta
    def lkj_covariance_matrix(self, theta_external):
        """
        Construct the LKJ covariance matrix for a 2x2 correlation matrix given the external parameter theta.

        Returns
        -------
        np.ndarray
            2x2 covariance matrix corresponding to the LKJ prior.
        """
        
        cov_matrix = xp.array([[theta_external[0]**2, theta_external[2] * theta_external[0] * theta_external[1]],
                               [theta_external[2] * theta_external[0] * theta_external[1], theta_external[1]**2]])

        return cov_matrix

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["Sigma1", f"{self.config.sigma1:.3f}"],
            ["Sigma2", f"{self.config.sigma2:.3f}"],
            ["Rho", f"{self.config.rho:.3f}"],
        ]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )

        submodel_table = add_str_header(
            title=self.submodel_type.replace("_", " ").title(),
            table=submodel_table,
        )
        str_representation += submodel_table

        return str_representation
