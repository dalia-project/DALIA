# Copyright 2024-2025 DALIA authors. All rights reserved.

import re

import numpy as np
from tabulate import tabulate

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.models_config import CoregionalModelConfig
from dalia.core.model import Model
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.prior_hyperparameters import (
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)
from dalia.submodels import RegressionSubModel, SpatialSubModel, SpatioTemporalSubModel
from dalia.utils import (
    add_str_header,
    align_tables_side_by_side,
    bdiag_tiling,
    boxify,
    free_unused_gpu_memory,
)
from dalia.utils.scalar_ndarray import ensure_scalar


class CoregionalModel(Model):
    """Core class for statistical models."""

    def __init__(
        self,
        models: list[Model],
        coregional_model_config: CoregionalModelConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        self.models: list[Model] = models

        # Check the coregionalization type (Spacial or SpatioTemporal)
        self.coregionalization_type: str
        self.n_models: int = coregional_model_config.n_models
        assert self.n_models == len(
            self.models
        ), "Number of models does not match the number of models in the CoregionalModelConfig"
        self.n_spatial_nodes: int = None
        self.n_temporal_nodes: int = 1
        self.n_fixed_effects_per_model: int = 0
        for i, model in enumerate(self.models):
            if i == 0:
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    self.coregionalization_type = "spatio_temporal"
                    self.n_spatial_nodes = model.submodels[0].ns
                    self.n_temporal_nodes = model.submodels[0].nt
                elif isinstance(model.submodels[0], SpatialSubModel):
                    self.coregionalization_type = "spatial"
                    self.n_spatial_nodes = model.submodels[0].ns
                else:
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        self.n_fixed_effects_per_model = model.submodels[
                            1
                        ].n_fixed_effects
            else:
                # Check that all models are the same
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    if self.coregionalization_type != "spatio_temporal":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatioTemporalModel)"
                        )
                    # Check that the size of the SpatioTemporal fields are the same
                    if (
                        model.submodels[0].ns != self.n_spatial_nodes
                        or model.submodels[0].nt != self.n_temporal_nodes
                    ):
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatioTemporalModel)"
                        )
                elif isinstance(model.submodels[0], SpatialSubModel):
                    if self.coregionalization_type != "spatial":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatialModel)"
                        )
                    # Check that the size of the Spatial fields are the same
                    if model.submodels[0].ns != self.n_spatial_nodes:
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatialModel)"
                        )
                else:
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        if (
                            model.submodels[1].n_fixed_effects
                            != self.n_fixed_effects_per_model
                        ):
                            raise ValueError(
                                f"Model {model} has a different number of fixed effects than the first model"
                            )

        self._theta_external: ArrayLike = []
        self._theta_internal: ArrayLike = []

        # For each model
        # Get Models() hyperparameters
        theta_external: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for model in self.models:
            theta_model = model.theta_external
            theta_keys_model = model.theta_keys

            # remove the theta that correspond to the "sigma_xx" where x can be whatever
            sigma_indices = [
                i
                for i, key in enumerate(theta_keys_model)
                if re.match(r"sigma_\w+", key)
            ]
            theta_model = [
                theta for i, theta in enumerate(theta_model) if i not in sigma_indices
            ]
            theta_keys_model = [
                key for i, key in enumerate(theta_keys_model) if i not in sigma_indices
            ]

            theta_external.append(xp.array(theta_model))
            theta_keys += theta_keys_model

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_model)
            )

            # Get the prior hyperparameters of the model
            self.prior_hyperparameters += [
                prior_hyperparameters
                for i, prior_hyperparameters in enumerate(model.prior_hyperparameters)
                if i not in sigma_indices
            ]

        # Initialize the Coregional Hyperparameters:
        (
            theta_coregional_model,
            theta_keys_coregional_model,
        ) = coregional_model_config.read_hyperparameters()
        theta_external.append(xp.array(theta_coregional_model))
        theta_keys += theta_keys_coregional_model

        self.hyperparameters_idx.append(
            self.hyperparameters_idx[-1] + len(theta_coregional_model)
        )

        # Finalize the hyperparameters
        self.theta_external: NDArray = xp.concatenate(theta_external)
        self.n_hyperparameters = self.theta_external.size
        self.theta_keys: NDArray = theta_keys

        # Initialize the Coregional Prior Hyperparameters
        for ph in coregional_model_config.ph_sigmas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")

        for ph in coregional_model_config.ph_lambdas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")

        # Construct Coregional Model data from its Models()
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: list[int] = [0]
        self.n_observations: int = 0
        self.n_observations_idx: list[int] = [0]
        for model in self.models:
            self.n_latent_parameters += model.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)
            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)

        # Assemble the latent parameters and observations from the Models()
        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        self.y: NDArray = xp.zeros(self.n_observations)
        for i, model in enumerate(self.models):
            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = model.x

            self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]] = (
                model.y
            )

        self.a: sp.sparse.spmatrix = bdiag_tiling(
            [model.a for model in self.models]
        ).tocsc()

        for model in self.models:
            model.a = None
            model.y = None
            model.x = None

        free_unused_gpu_memory()

        if self.coregionalization_type == "spatio_temporal":
            self.permutation_Qst = self._generate_permutation_indices(
                self.n_temporal_nodes, self.n_spatial_nodes, self.n_models
            )

            self.permutation_latent_variables = (
                self._generate_permutation_indices_for_a(
                    self.n_temporal_nodes,
                    self.n_spatial_nodes,
                    self.n_models,
                    self.n_fixed_effects_per_model,
                )
            )

            self.a = self.a[:, self.permutation_latent_variables]
            self.x = self.x[self.permutation_latent_variables]

        elif self.coregionalization_type == "spatial":
            n_re = self.n_spatial_nodes * self.n_temporal_nodes

            # don't permute when nt is 1
            self.permutation_Qst = xp.arange(0, self.n_models * n_re, 1)

            # permute fixed effects to the end
            self.permutation_latent_variables = (
                self._generate_permutation_indices_spatial(
                    self.n_spatial_nodes, self.n_fixed_effects_per_model, self.n_models
                )
            )

            self.a = self.a[:, self.permutation_latent_variables]
            self.x = self.x[self.permutation_latent_variables]

        # self.inverse_permutation_latent_variables = xp.argsort(self.permutation_latent_variables)
        # self.perm2 = self._generate_permutation_indices_for_a_new(
        #     self.n_temporal_nodes,
        #     self.n_spatial_nodes,
        #     self.n_models,
        #     self.n_fixed_effects_per_model,
        # )
        # self.inverse_permutation_latent_variables = xp.argsort(self.perm2)

        # --- Recurrent variables
        self.Q_prior_data_mapping = [0]

        self.rows_Qprior_re = None
        self.columns_Qprior_re = None
        self.data_Qprior_re = None

        self.permutation_vector_Q_prior = None
        self.permutation_indices_Q_prior = None
        self.permutation_indptr_Q_prior = None

        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

        self.Q_prior: sp.sparse.spmatrix = (
            None  # need this otherwise the construct will fail
        )

        self.construct_Q_prior()

    def construct_Q_prior(self) -> sp.sparse.spmatrix:
        # number of random effects per model
        n_re = self.n_spatial_nodes * self.n_temporal_nodes

        Qu_list: list = [None] * self.n_models
        Q_r: list = [None] * self.n_models

        for i, model in enumerate(self.models):
            submodel_st = model.submodels[0]
            # Get the spatio-temporal submodel idx
            kwargs_st = {}
            for hp_idx in range(
                self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
            ):
                kwargs_st[self.theta_keys[hp_idx]] = float(self.theta_external[hp_idx])

            Qu_list[i] = submodel_st.construct_Q_prior(**kwargs_st).tocsc()

            if len(model.submodels) > 1:
                # Create the regression tip
                submodel_r = model.submodels[1]
                # Get the spatio-temporal submodel idx
                kwargs_r = {}
                Q_r[i] = submodel_r.construct_Q_prior(**kwargs_r).tocsc()

        sigma_0 = xp.exp(self.theta_external[self.theta_keys.index("sigma_0")])
        sigma_1 = xp.exp(self.theta_external[self.theta_keys.index("sigma_1")])

        lambda_0_1 = self.theta_external[self.theta_keys.index("lambda_0_1")]

        if self.n_models == 2:
            q11 = sp.sparse.coo_matrix(
                (1 / sigma_0**2) * Qu_list[0]
                + (lambda_0_1**2 / sigma_1**2) * Qu_list[1]
            )
            if not q11.has_canonical_format:
                q11.sum_duplicates()

            Qu_list[0] = None

            q21 = sp.sparse.coo_matrix((-lambda_0_1 / sigma_1**2) * Qu_list[1])
            if not q21.has_canonical_format:
                q21.sum_duplicates()

            q12 = sp.sparse.coo_matrix((-lambda_0_1 / sigma_1**2) * Qu_list[1])
            if not q12.has_canonical_format:
                q12.sum_duplicates()

            q22 = sp.sparse.coo_matrix((1 / sigma_1**2) * Qu_list[1])
            if not q22.has_canonical_format:
                q22.sum_duplicates()

            Qu_list[1] = None

            if self.data_Qprior_re is None:
                q11_rows = q11.row
                q11_columns = q11.col

                q21_rows = q21.row + n_re
                q21_columns = q21.col

                q12_rows = q12.row
                q12_columns = q12.col + n_re

                q22_rows = q22.row + n_re
                q22_columns = q22.col + n_re

                self.rows_Qprior_re = xp.concatenate(
                    [q11_rows, q12_rows, q21_rows, q22_rows]
                )
                self.columns_Qprior_re = xp.concatenate(
                    [q11_columns, q12_columns, q21_columns, q22_columns]
                )

            self.data_Qprior_re = xp.concatenate(
                [q11.data, q12.data, q21.data, q22.data]
            )

            # Qprior_st = sp.sparse.bmat([[q11, q12], [q21, q22]]).tocsc()

        elif self.n_models == 3:
            sigma_2 = xp.exp(self.theta_external[self.theta_keys.index("sigma_2")])

            lambda_0_2 = self.theta_external[self.theta_keys.index("lambda_0_2")]
            lambda_1_2 = self.theta_external[self.theta_keys.index("lambda_1_2")]

            q11 = sp.sparse.coo_matrix(
                (1 / sigma_0**2) * Qu_list[0]
                + (lambda_0_1**2 / sigma_1**2) * Qu_list[1]
                + (lambda_1_2**2 / sigma_2**2) * Qu_list[2]
            )
            Qu_list[0] = None
            if not q11.has_canonical_format:
                q11.sum_duplicates()

            q21 = sp.sparse.coo_matrix(
                (-lambda_0_1 / sigma_1**2) * Qu_list[1]
                + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qu_list[2]
            )
            if not q21.has_canonical_format:
                q21.sum_duplicates()

            q31 = sp.sparse.coo_matrix(-lambda_1_2 / sigma_2**2 * Qu_list[2])
            if not q31.has_canonical_format:
                q31.sum_duplicates()

            q22 = sp.sparse.coo_matrix(
                (1 / sigma_1**2) * Qu_list[1]
                + (lambda_0_2**2 / sigma_2**2) * Qu_list[2]
            )
            if not q22.has_canonical_format:
                q22.sum_duplicates()
            Qu_list[1] = None

            q32 = sp.sparse.coo_matrix(-lambda_0_2 / sigma_2**2 * Qu_list[2])
            if not q32.has_canonical_format:
                q32.sum_duplicates()

            q33 = sp.sparse.coo_matrix((1 / sigma_2**2) * Qu_list[2])
            if not q33.has_canonical_format:
                q33.sum_duplicates()
            Qu_list[2] = None

            # not the most elegant way but im afraid that without the copy it might break in some cases
            q12 = (q21.copy()).T
            if not q12.has_canonical_format:
                q12.sum_duplicates()

            q13 = (q31.copy()).T
            if not q13.has_canonical_format:
                q13.sum_duplicates()

            q23 = (q32.copy()).T
            if not q23.has_canonical_format:
                q23.sum_duplicates()

            free_unused_gpu_memory()

            # we only need these indices once in the beginning
            # then they can be none again and we can only collect data array
            if self.data_Qprior_re is None:
                q11_rows = q11.row
                q11_columns = q11.col

                q21_rows = q21.row + n_re
                q21_columns = q21.col

                q31_rows = q31.row + 2 * n_re
                q31_columns = q31.col

                q22_rows = q22.row + n_re
                q22_columns = q22.col + n_re

                q32_rows = q32.row + 2 * n_re
                q32_columns = q32.col + n_re

                q33_rows = q33.row + 2 * n_re
                q33_columns = q33.col + 2 * n_re

                ## CAREFUL IF THIS IS NOT A "TRUE" COPY ...
                q12_rows = q12.row
                q12_columns = q12.col + n_re

                q13_rows = q13.row
                q13_columns = q13.col + 2 * n_re

                q23_rows = q23.row + n_re
                q23_columns = q23.col + 2 * n_re

                self.rows_Qprior_re = xp.concatenate(
                    [
                        q11_rows,
                        q12_rows,
                        q13_rows,
                        q21_rows,
                        q22_rows,
                        q23_rows,
                        q31_rows,
                        q32_rows,
                        q33_rows,
                    ]
                )
                self.columns_Qprior_re = xp.concatenate(
                    [
                        q11_columns,
                        q12_columns,
                        q13_columns,
                        q21_columns,
                        q22_columns,
                        q23_columns,
                        q31_columns,
                        q32_columns,
                        q33_columns,
                    ]
                )

            # this changes every time -> need to keep
            self.data_Qprior_re = xp.concatenate(
                [
                    q11.data,
                    q12.data,
                    q13.data,
                    q21.data,
                    q22.data,
                    q23.data,
                    q31.data,
                    q32.data,
                    q33.data,
                ]
            )

            free_unused_gpu_memory()

            # Qprior_st = sp.sparse.bmat(
            #     [[q11, q12, q13], [q21, q22, q23], [q31, q32, q33]]
            # ).tocsc()

            # Qprior_re = sp.sparse.coo_matrix((self.data_Qprior_re, (self.rows_Qprior_re, self.columns_Qprior_re)), shape=( self.n_models * n_re,  self.n_models * n_re))

        # Permute matrix
        # Qprior_st_perm = Qprior_st[self.permutation_Qst, :][:, self.permutation_Qst]

        if self.permutation_vector_Q_prior is None:
            self.Qprior_re_perm = sp.sparse.csc_matrix(
                (self.n_models * n_re, self.n_models * n_re),
                dtype=self.data_Qprior_re.dtype,
            )

            # self.permutation_Qst is the identity for spatial models
            self.set_data_array_permutation_indices(
                self.permutation_Qst,
                self.rows_Qprior_re,
                self.columns_Qprior_re,
                self.n_models * n_re,
            )

            # we only need to set these once
            self.Qprior_re_perm.indices = self.permutation_indices_Q_prior
            self.Qprior_re_perm.indptr = self.permutation_indptr_Q_prior

            # dont need these anymore
            self.rows_Qprior_re = None
            self.columns_Qprior_re = None

        free_unused_gpu_memory()

        self.data_Qprior_re = self.data_Qprior_re[self.permutation_vector_Q_prior]

        if Q_r != []:
            if self.Q_prior is None:
                self.Qprior_re_perm.data = self.data_Qprior_re

                Qprior_reg = bdiag_tiling(Q_r).tocsc()
                self.Q_prior = bdiag_tiling([self.Qprior_re_perm, Qprior_reg]).tocsc()
                self.nnz_Qprior_re_perm = self.Qprior_re_perm.nnz

                # free all memory not needed anymore
                self.Qprior_re_perm = None
                self.permutation_indices_Q_prior = None
                self.permutation_indptr_Q_prior = None
            else:
                free_unused_gpu_memory()

                self.Q_prior.tocsc()
                self.Q_prior.sort_indices()
                self.Q_prior.data[: self.nnz_Qprior_re_perm] = self.data_Qprior_re
        else:
            self.Q_prior = self.Qprior_re_perm

        free_unused_gpu_memory()

        return self.Q_prior

    def spgemm(self, A, B, rows: int = 5408):
        free_unused_gpu_memory()

        C = None
        for i in range(0, A.shape[0], rows):
            A_block = A[i : min(A.shape[0], i + rows)]
            C_block = A_block @ B
            if C is None:
                C = C_block
            else:
                C = sp.sparse.vstack([C, C_block], format="csr")

        free_unused_gpu_memory()

        return C.tocsc()

    def custom_Q_ATDA(
        self, Q: sp.sparse.csc_matrix, A: sp.sparse.csc_matrix, D_diag: xp.ndarray
    ) -> sp.sparse.csr_matrix:
        """
        Computes A^T * D * A with minimal memory and maximum speed.
        - Uses sparse diagonal multiplication.
        - No temporary dense matrices.
        """

        DA = A.multiply(D_diag[:, xp.newaxis]).T.tocsr()

        mem_used_bytes = free_unused_gpu_memory()

        # use batched spgemm if mempool is full
        if mem_used_bytes > 80 * 1024**3:
            batch_size = int(xp.ceil(A.shape[0] / 2))
        else:
            batch_size = A.shape[0]

        ATDA = self.spgemm(DA, A, rows=batch_size)
        free_unused_gpu_memory()

        self.Qconditional = Q - ATDA
        free_unused_gpu_memory()

        return self.Qconditional

    def construct_Q_conditional(
        self,
        eta: NDArray,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """
        d_vec = xp.zeros(self.n_observations)

        for i, model in enumerate(self.models):
            if model.likelihood_config.type == "gaussian":
                kwargs = {
                    "eta": eta[
                        self.n_observations_idx[i] : self.n_observations_idx[i + 1]
                    ],
                    "theta": float(self.theta_external[self.hyperparameters_idx[i + 1] - 1]),
                }
            else:
                kwargs = {
                    "eta": eta[
                        self.n_observations_idx[i] : self.n_observations_idx[i + 1]
                    ],
                }

            # d_list[i] = model.likelihood.evaluate_hessian_likelihood(**kwargs)
            d_vec[self.n_observations_idx[i] : self.n_observations_idx[i + 1]] = (
                model.likelihood.evaluate_hessian_likelihood(**kwargs).diagonal()
            )

        self.Qconditional = self.custom_Q_ATDA(
            Q=self.Q_prior,
            A=self.a,
            D_diag=d_vec,
        )
        self.Q_conditional = self.Qconditional.tocsc()
        free_unused_gpu_memory()

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        gradient_vector_list = []
        for i, model in enumerate(self.models):
            gradient_likelihood = model.likelihood.evaluate_gradient_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta_external[self.hyperparameters_idx[i + 1] - 1]),
            )

            gradient_vector_list.append(gradient_likelihood)

        gradient_likelihood = xp.concatenate(gradient_vector_list)

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def is_likelihood_gaussian(self) -> bool:
        """Check if the likelihood is Gaussian.
        
        Returns
        -------
        is_gaussian : bool
            True if the likelihood is Gaussian, False otherwise.
        """
        for model in self.models:
            if not model.is_likelihood_gaussian():
                return False
        return True

    def evaluate_likelihood(
        self,
        eta: NDArray,
    ) -> float:
        """Evaluate the likelihood.
        
        Parameters
        ----------
        eta : NDArray
            Linear predictor.
        kwargs : dict
            Additional arguments for the likelihood evaluation. These parameters are model dependent.

        Returns
        -------
        likelihood : float
            The evaluated likelihood.

        Implementation Notes:
        ---------------------
        - The likelihood is evaluated for each model and then summed up to get the total likelihood of the CoregionalModel.
        - Returned as a scalar for consistency, even if the likelihood is computed as a sum of multiple likelihoods from different models.
        """
        likelihood: float = 0.0
        for i, model in enumerate(self.models):
            likelihood += model.likelihood.evaluate_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta_external[self.hyperparameters_idx[i + 1] - 1]),
            )
   
        return ensure_scalar(likelihood)

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        theta_interpret = self.theta_external

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def __str__(self) -> str:
        """String representation of the model."""
        str_representation = ""

        # --- Make the Coregional Model() table ---
        headers = [
            "Number of Hyperparameters",
            "Number of Latent Parameters",
            "Number of Observations",
        ]
        values = [self.n_hyperparameters, self.n_latent_parameters, self.n_observations]

        model_table = tabulate(
            [headers, values],
            tablefmt="fancy_grid",
            colalign=("center", "center", "center"),
        )

        # Add the header title
        model_table = add_str_header(
            f"Coregional Model ({self.n_models} variates)", model_table
        )

        # --- Add the model information ---
        # Create headers and values for the model table
        models_str_representation = []
        for model in self.models:
            models_str_representation.append(str(model))

        # Create the model table
        model_jointed_representation = align_tables_side_by_side(
            models_str_representation
        )

        # Add the model header title
        model_jointed_representation = add_str_header(
            "Models", model_jointed_representation
        )

        # Combine the model and model tables
        str_representation = model_table + "\n" + boxify(model_jointed_representation)

        return str_representation

    def _generate_permutation_indices_spatial(
        self, n_spatial_nodes: int, n_fixed_effects_per_model: int, n_models: int
    ):
        perm_vec = np.zeros(
            n_models * (n_spatial_nodes + n_fixed_effects_per_model), dtype=int
        )
        for i in range(n_models):
            perm_vec[i * n_spatial_nodes : (i + 1) * n_spatial_nodes] = range(
                i * (n_spatial_nodes + n_fixed_effects_per_model),
                i * (n_spatial_nodes + n_fixed_effects_per_model) + n_spatial_nodes,
            )
            perm_vec[
                n_models * n_spatial_nodes
                + i * n_fixed_effects_per_model : n_models * n_spatial_nodes
                + (i + 1) * n_fixed_effects_per_model
            ] = range(
                i * (n_spatial_nodes + n_fixed_effects_per_model) + n_spatial_nodes,
                (i + 1) * (n_spatial_nodes + n_fixed_effects_per_model),
            )

        return perm_vec

    def _generate_permutation_indices(
        self, n_temporal_nodes: int, n_spatial_nodes: int, n_models: int
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes

        if n_models == 2:
            perm_vectorized = np.hstack((first_idx, second_idx)).flatten()
        if n_models == 3:
            third_idx = second_idx + n_temporal_nodes * n_spatial_nodes
            perm_vectorized = np.hstack((first_idx, second_idx, third_idx)).flatten()

        return perm_vectorized

    def _generate_permutation_indices_for_a_new(
        self,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        n_models: int,
        n_fixed_effects_per_model: int,
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)
        # indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes
        indices_fixed_effects_1 = 2 * n_temporal_nodes * n_spatial_nodes + np.arange(
            n_fixed_effects_per_model
        )
        indices_fixed_effects_2 = n_fixed_effects_per_model + indices_fixed_effects_1

        if n_models == 2:
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                ]
            )
        elif n_models == 3:
            third_idx = (
                second_idx
                + n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
            )
            indices_fixed_effects_3 = (
                3 * n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
                + np.arange(n_fixed_effects_per_model)
            )
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx, third_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                    indices_fixed_effects_3,
                ]
            )

        return perm_vectorized

    def _generate_permutation_indices_for_a(
        self,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        n_models: int,
        n_fixed_effects_per_model: int,
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)
        indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = (
            first_idx + n_fixed_effects_per_model + n_temporal_nodes * n_spatial_nodes
        )
        indices_fixed_effects_2 = (
            2 * n_temporal_nodes * n_spatial_nodes
            + n_fixed_effects_per_model
            + np.arange(n_fixed_effects_per_model)
        )

        if n_models == 2:
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                ]
            )
        elif n_models == 3:
            third_idx = (
                second_idx
                + n_temporal_nodes * n_spatial_nodes
                + n_fixed_effects_per_model
            )
            indices_fixed_effects_3 = (
                3 * n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
                + np.arange(n_fixed_effects_per_model)
            )
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx, third_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                    indices_fixed_effects_3,
                ]
            )

        return perm_vectorized

    def set_data_array_permutation_indices(
        self, permutation, a_rows: NDArray, a_cols: NDArray, n: int
    ) -> None:
        a_data_placeholder = xp.arange(0, len(a_rows), 1)
        a = sp.sparse.csc_matrix(
            sp.sparse.coo_matrix(
                (a_data_placeholder, (a_rows, a_cols)), shape=(n, n), dtype=xp.float64
            )
        )

        a_perm = a[permutation, :][:, permutation]
        a_perm.sort_indices()  ## new

        self.permutation_vector_Q_prior = a_perm.data.astype(xp.int32)
        self.permutation_indices_Q_prior = a_perm.indices
        self.permutation_indptr_Q_prior = a_perm.indptr

    def construct_a_predict(self) -> sp.sparse.spmatrix:
        # iterate through the models to load their respective a_predict
        for i, model in enumerate(self.models):
            model.construct_a_predict()

        self.a_predict: sp.sparse.spmatrix = bdiag_tiling(
            [model.a_predict for model in self.models]
        ).tocsc()

        # Reorder a_predict in the same way as a
        self.a_predict = self.a_predict[:, self.permutation_latent_variables]

        return self.a_predict

    def compare_matrices(
        self, a1_data_vec, a1_indices, a1_indptr, a2_data_vec, a2_indices, a2_indptr
    ):
        """
        Compare two sparse matrices represented by their data vectors, indices, and indptr arrays.
        """
        # Check if the shapes of the matrices are the same
        # if len(a1_data_vec) != len(a2_data_vec):
        #     return False

        # Check if the indices arrays are equal
        if not xp.array_equal(a1_indices, a2_indices):
            print("indices arrays are not equal")
            for i, (idx1, idx2) in enumerate(zip(a1_indices, a2_indices)):
                if idx1 != idx2:
                    print(f"Indices differ at index {i}: {idx1} != {idx2}")
            return False

        # Check if the indptr arrays are equal
        if not xp.array_equal(a1_indptr, a2_indptr):
            print("indptr arrays are not equal")
            for i, (ptr1, ptr2) in enumerate(zip(a1_indptr, a2_indptr)):
                if ptr1 != ptr2:
                    print(f"Indptr arrays differ at index {i}: {ptr1} != {ptr2}")
            return False

        if not xp.array_equal(a1_data_vec, a2_data_vec):
            print("data vectors are not equal")
            print("theta: ", self.theta)
            for i, (val1, val2) in enumerate(zip(a1_data_vec, a2_data_vec)):
                if val1 != val2:
                    print(f"Data vectors differ at index {i}: {val1} != {val2}")
            return False

        return True

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = self.n_models * self.n_spatial_nodes
        n_diag_blocks = self.n_temporal_nodes
        arrowhead_blocksize = self.n_fixed_effects_per_model * self.n_models

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param

    def total_number_fixed_effects(self) -> int:
        """Get the number of fixed effects."""
        return self.n_fixed_effects_per_model * self.n_models
