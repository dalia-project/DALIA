# Copyright 2024-2025 DALIA authors. All rights reserved.

import os
from abc import ABC
from pathlib import Path

import numpy as np
from tabulate import tabulate

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.likelihood_config import LikelihoodConfig
from dalia.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
    GaussianMVNPriorHyperparametersConfig,
    GaussianPriorHyperparametersConfig,
    PenalizedComplexityPriorHyperparametersConfig,
    GammaPriorHyperparametersConfig,
)
from dalia.core.likelihood import Likelihood
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.core.submodel import SubModel
from dalia.likelihoods import BinomialLikelihood, GaussianLikelihood, PoissonLikelihood
from dalia.prior_hyperparameters import (
    BetaPriorHyperparameters,
    GaussianMVNPriorHyperparameters,
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
    GammaPriorHyperparameters,
)
from dalia.submodels import (
    BrainiacSubModel,
    RegressionSubModel,
    SpatialSubModel,
    SpatioTemporalSubModel,
    AR1SubModel,
    AR2SubModel,
)
from dalia.utils import add_str_header, boxify, scaled_logit
from dalia.utils.scalar_ndarray import ensure_scalar


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        submodels: list[SubModel],
        likelihood_config: LikelihoodConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        self.modeltype = kwargs.get("modeltype", "Default Model")

        # Check the order of the submodels, we want the SpatioTemporalSubModel first
        # as this will decide the sparsity pattern of the precision matrix.
        for i, submodel in enumerate(submodels):
            if isinstance(submodel, SpatioTemporalSubModel):
                submodels.insert(0, submodels.pop(i))

        self.submodels: list[SubModel] = submodels

        self.n_fixed_effects: int = 0

        self._theta_external: ArrayLike = []
        self._theta_internal: ArrayLike = []

        # For each submodel...
        theta_external: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for submodel in self.submodels:
            # ...initialize their prior hyperparameters matching their hyperparameters
            if isinstance(submodel, SpatioTemporalSubModel):
                # Spatial hyperparameters
                if isinstance(submodel.config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_s,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_s,
                            hyperparameter_type="r_s",
                        )
                    )

                # Temporal hyperparameters
                if isinstance(submodel.config.ph_t, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_t,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_t, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_t,
                            hyperparameter_type="r_t",
                        )
                    )

                # Sigma spatio-temporal hyperparameters
                if isinstance(
                    submodel.config.ph_st, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_st,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_st, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_st,
                            hyperparameter_type="sigma_st",
                        )
                    )
            elif isinstance(submodel, SpatialSubModel):
                # spatial range
                if isinstance(submodel.config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_s,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_s,
                            hyperparameter_type="r_s",
                        )
                    )

                # spatial variation
                if isinstance(submodel.config.ph_e, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_e,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_e, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_e,
                            hyperparameter_type="sigma_e",
                        )
                    )
            elif isinstance(submodel, RegressionSubModel):
                self.n_fixed_effects += submodel.n_fixed_effects

            elif isinstance(submodel, AR1SubModel):

                if isinstance(submodel.config.ph_phi, BetaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        BetaPriorHyperparameters(
                            config=submodel.config.ph_phi,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_phi,
                    PenalizedComplexityPriorHyperparametersConfig,
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_phi,
                            hyperparameter_type="phi",
                        )
                    )

                if isinstance(
                    submodel.config.ph_tau, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_tau,
                        )
                    )
                if isinstance(submodel.config.ph_tau, GammaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GammaPriorHyperparameters(
                            config=submodel.config.ph_tau,
                        )
                    )
                else:
                    raise ValueError("Unknown prior hyperparameter type for ph_tau")

            elif isinstance(submodel, AR2SubModel):

                for ph_pacf, hp_type in [
                    (submodel.config.ph_pacf1, "pacf1"),
                    (submodel.config.ph_pacf2, "pacf2"),
                ]:
                    if isinstance(ph_pacf, BetaPriorHyperparametersConfig):
                        self.prior_hyperparameters.append(
                            BetaPriorHyperparameters(
                                config=ph_pacf,
                            )
                        )
                    elif isinstance(
                        ph_pacf,
                        PenalizedComplexityPriorHyperparametersConfig,
                    ):
                        self.prior_hyperparameters.append(
                            PenalizedComplexityPriorHyperparameters(
                                config=ph_pacf,
                                hyperparameter_type=hp_type,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown prior hyperparameter type for ph_{hp_type}"
                        )

                if isinstance(
                    submodel.config.ph_tau, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_tau,
                        )
                    )
                elif isinstance(submodel.config.ph_tau, GammaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GammaPriorHyperparameters(
                            config=submodel.config.ph_tau,
                        )
                    )
                else:
                    raise ValueError("Unknown prior hyperparameter type for ph_tau")

            elif isinstance(submodel, BrainiacSubModel):
                # h2 hyperparameters
                if isinstance(submodel.config.ph_h2, BetaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        BetaPriorHyperparameters(
                            config=submodel.config.ph_h2,
                        )
                    )

                # alpha hyperparameters
                if isinstance(
                    submodel.config.ph_alpha, GaussianMVNPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianMVNPriorHyperparameters(
                            config=submodel.config.ph_alpha,
                        )
                    )
                if isinstance(
                    submodel.config.ph_alpha,
                    PenalizedComplexityPriorHyperparametersConfig,
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_alpha,
                            hyperparameter_type="alpha",
                        )
                    )
            else:
                raise ValueError("Unknown submodel type")

            # ...and read their hyperparameters
            theta_submodel, theta_keys_submodel = submodel.config.read_hyperparameters()

            theta_external.append(theta_submodel)
            theta_keys += theta_keys_submodel

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_submodel)
            )

        # --- Initialize the latent parameters and the design matrix
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: list[int] = [0]

        for submodel in self.submodels:
            self.n_latent_parameters += submodel.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.x: NDArray = xp.zeros(self.n_latent_parameters)

        # check if all a are sparse -> if not construct dense a
        if all(sp.sparse.issparse(submodel.a) for submodel in self.submodels):
            data = []
            rows = []
            cols = []
            for i, submodel in enumerate(self.submodels):
                # Convert csc_matrix to coo_matrix to allow slicing
                coo_submodel_a = submodel.a.tocoo()
                data.append(coo_submodel_a.data)
                rows.append(coo_submodel_a.row)
                cols.append(
                    coo_submodel_a.col
                    + self.latent_parameters_idx[i]
                    * xp.ones(coo_submodel_a.col.size, dtype=int)
                )

                self.x[
                    self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
                ] = submodel.x_initial

            self.a: sp.sparse.spmatrix = sp.sparse.coo_matrix(
                (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
                shape=(submodel.a.shape[0], self.n_latent_parameters),
            )
        else:
            data = []
            for i, submodel in enumerate(self.submodels):
                if sp.sparse.issparse(submodel.a):
                    data.append(submodel.a.toarray())
                else:
                    data.append(submodel.a)

                self.x[
                    self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
                ] = submodel.x_initial

            self.a: NDArray = xp.concatenate(data, axis=1)

        self.permutation_latent_variables = xp.arange(0, self.n_latent_parameters, 1)
        self.inverse_permutation_latent_variables = xp.arange(
            0, self.n_latent_parameters, 1
        )

        # if data is gaussian compute t(A)*A once
        if likelihood_config.type == "gaussian":
            self.aTa = self.a.T @ self.a
        else:
            self.aTa = None

        # --- Load observation vector
        input_dir = Path(
            kwargs.get("input_dir", os.path.dirname(submodels[0].config.input_dir))
        )

        y: NDArray = np.load(input_dir / "y.npy")
        if xp == np:
            self.y: NDArray = y
        else:
            self.y: NDArray = xp.asarray(y)

        self.y = self.y.flatten()

        self.n_observations: int = self.y.shape[0]

        # --- Initialize likelihood
        if likelihood_config.type == "gaussian":
            self.likelihood: Likelihood = GaussianLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

            if self.submodels[0] == BrainiacSubModel:
                # skip setting prior as it's already set in the submodel
                print(
                    "Brainiac model detected. Skipping setting prior hyperparameters as already set."
                )
            # Instantiate the prior hyperparameters for the likelihood
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                GaussianPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                    )
                )
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                PenalizedComplexityPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                        hyperparameter_type="prec_o",
                    )
                )
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                BetaPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    BetaPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                    )
                )
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                GammaPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    GammaPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                    )
                )
        elif likelihood_config.type == "poisson":
            self.likelihood: Likelihood = PoissonLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )
        elif likelihood_config.type == "binomial":
            self.likelihood: Likelihood = BinomialLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

        self.likelihood_config: LikelihoodConfig = likelihood_config

        # Add the likelihood hyperparameters
        (
            lh_hyperparameters,
            lh_hyperparameters_keys,
        ) = likelihood_config.read_hyperparameters()

        theta_external.append(lh_hyperparameters)
        self.theta_external = xp.concatenate(theta_external)

        print("Initial hyperparameters (external scale): ", self.theta_external)
        print("Initial hyperparameters (internal scale): ", self.theta_internal)

        theta_keys += lh_hyperparameters_keys
        self.theta_keys: NDArray = theta_keys

        self.n_hyperparameters = self.theta_external.size

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    ########################################################################
    @property
    def theta_external(self):
        """External/user/interpretable scale theta."""
        # the copy is important to make sure that in place operations still trigger updating
        return self._theta_external.copy()

    @theta_external.setter
    def theta_external(self, value):
        """Set external theta and automatically update internal.
        
        Notes
        -----
        The re-scaling is implemented for all prios but PenalizedComplexity (identity but already in the correct "log" scale).
        """
        self._theta_external = xp.array(value)
        self._theta_internal = self.rescale_hyperparameters_to_internal(
            self._theta_external, direction="forward"
        )

    @property
    def theta_internal(self):
        """Internal/BFGS scale theta."""
        return self._theta_internal.copy()

    @theta_internal.setter
    def theta_internal(self, value):
        """Set internal theta and automatically update external."""
        self._theta_internal = xp.array(value)
        self._theta_external = self.rescale_hyperparameters_to_internal(
            self._theta_internal, direction="backward"
        )

    ########################################################################

    def construct_Q_prior(self) -> sp.sparse.spmatrix:
        kwargs = {}

        if self.Q_prior is None:
            # During the first construction of Q_prior, we allocate the memory for
            # the data and the mapping of each submodel's to the Q prior matrix.
            rows = []
            cols = []
            data = []

            ## TODO: improve the if / elif statements
            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, SpatialSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, AR1SubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, AR2SubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, RegressionSubModel):
                    ...

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                rows.append(
                    submodel_Q_prior.row
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.row))
                )
                cols.append(
                    submodel_Q_prior.col
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.col))
                )
                data.append(submodel_Q_prior.data)

                self.Q_prior_data_mapping.append(
                    self.Q_prior_data_mapping[i] + len(submodel_Q_prior.data)
                )

            self.Q_prior: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
                (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
                shape=(self.n_latent_parameters, self.n_latent_parameters),
            )

        else:
            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, RegressionSubModel):
                    ...
                elif isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, SpatialSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, AR1SubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])
                elif isinstance(submodel, AR2SubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(
                            self.theta_external[hp_idx]
                        )
                        # kwargs[self.theta_keys[hp_idx]] = float(theta_interpret[hp_idx])

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                self.Q_prior.data[
                    self.Q_prior_data_mapping[i] : self.Q_prior_data_mapping[i + 1]
                ] = submodel_Q_prior.data

        return self.Q_prior

    def construct_Q_conditional(
        self,
        eta: NDArray,
    ):
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """

        if self.likelihood_config.type == "gaussian":
            kwargs = {
                "eta": eta,
                "theta": float(self.theta_external[-1]),
            }
        else:
            kwargs = {
                "eta": eta,
            }

        if isinstance(self.submodels[0], BrainiacSubModel):
            # Brainiac specific rule
            kwargs["h2"] = float(self.theta_external[0])
            d_matrix = self.submodels[0].evaluate_d_matrix(**kwargs)
        else:
            # General rules
            d_matrix = self.likelihood.evaluate_hessian_likelihood(**kwargs)

        # if self.a is sparse -> Q_conditional should be sparse, else dense
        if sp.sparse.issparse(self.a):
            if self.aTa is not None:
                self.Q_conditional = self.Q_prior - d_matrix.diagonal()[0] * self.aTa
            else:
                self.Q_conditional = self.Q_prior - self.a.T @ d_matrix @ self.a
            # self.Q_conditional = self.Q_prior - self.a.T @ d_matrix @ self.a
        else:
            if self.aTa is not None:
                self.Q_conditional = (
                    self.Q_prior.toarray() - d_matrix.diagonal()[0] * self.aTa
                )
            else:
                self.Q_conditional = (
                    self.Q_prior.toarray() - self.a.T @ d_matrix @ self.a
                )
            # self.Q_conditional = self.Q_prior.toarray() - self.a.T @ d_matrix @ self.a

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            kwargs = {"h2": float(self.theta_external[0])}
            gradient_likelihood = self.submodels[0].evaluate_gradient_likelihood(
                eta=eta, y=self.y, **kwargs
            )

        else:
            gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
                eta=eta,
                y=self.y,
                theta=self.theta_external[self.hyperparameters_idx[-1] :],
            )

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def is_likelihood_gaussian(self) -> bool:
        """Check if the likelihood is Gaussian."""
        return self.likelihood_config.type == "gaussian"

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            if isinstance(prior_hyperparameter, GaussianMVNPriorHyperparameters):
                # for MVN prior hyperparameters, we need to pass the full vector
                log_prior += prior_hyperparameter.evaluate_log_prior(
                    self.theta_external[i : i + prior_hyperparameter.mean.shape[0]]
                )
            else:
                log_prior += prior_hyperparameter.evaluate_log_prior(
                    self.theta_external[i]
                )

        return log_prior

    def get_theta_likelihood(self) -> NDArray:
        """Return the likelihood hyperparameters."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            theta_likelihood = 1 - self.theta_external[0]
        else:
            theta_likelihood = self.theta_external[self.hyperparameters_idx[-1] :]

        return theta_likelihood

    def rescale_hyperparameters_to_internal(self, theta, direction):

        # need to iterate over theta and its prior hyperparameters
        theta_internal = xp.copy(theta)

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            ## how to handle priors that have multiple hyperparameters?
            if isinstance(prior_hyperparameter, GaussianMVNPriorHyperparameters):
                pass  # no rescaling implemented
            else:
                theta_internal[i] = (
                    prior_hyperparameter.rescale_hyperparameters_to_internal(
                        theta[i], direction=direction
                    )
                )

        return theta_internal

    def evaluate_likelihood(self, eta: NDArray, **kwargs) -> float:
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
        """

        if isinstance(self.submodels[0], BrainiacSubModel):
            # kwargs["h2"] = float(self.theta[0])
            kwargs["h2"] = float(self.theta_external[0])
            likelihood = self.submodels[0].evaluate_likelihood(eta, self.y, **kwargs)
        else:
            likelihood = self.likelihood.evaluate_likelihood(
                eta, self.y, theta=self.theta_external[self.hyperparameters_idx[-1] :]
            )

        return ensure_scalar(likelihood)

        

    def __str__(self) -> str:
        """String representation of the model."""
        str_representation = ""

        # --- Make the Model() table ---
        headers = [
            "Number of Hyperparameters",
            "Number of Latent Parameters",
            "Number of Observations",
            "Type of Likelihood",
        ]
        values = [
            self.n_hyperparameters,
            self.n_latent_parameters,
            self.n_observations,
            self.likelihood_config.type.capitalize(),
        ]

        model_table = tabulate(
            [headers, values],
            tablefmt="fancy_grid",
            colalign=("center", "center", "center", "center"),
        )

        # Add the header title
        model_table = add_str_header("Default Model", model_table)

        # --- Add the submodel information ---
        # Create headers and values for the submodel table
        submodels_str_representation = []
        for submodel in self.submodels:
            submodels_str_representation.append(str(submodel))

        lines_list = [s.splitlines() for s in submodels_str_representation]
        max_len = max(len(lines) for lines in lines_list)

        # Pad each list of lines to the same length
        for lines in lines_list:
            lines += [""] * (max_len - len(lines))

        # Concatenate corresponding lines
        result_lines = ["  ".join(parts) for parts in zip(*lines_list)]
        submodel_jointed_representation = "\n".join(result_lines)

        # Add the submodel header title
        submodel_jointed_representation = add_str_header(
            "Submodels", submodel_jointed_representation
        )

        # Combine the model and submodel tables
        str_representation = model_table + "\n" + submodel_jointed_representation

        return boxify(str_representation)

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = None
        n_diag_blocks = None
        arrowhead_blocksize = 0
        if isinstance(self.submodels[0], SpatioTemporalSubModel):
            diagonal_blocksize = self.submodels[0].ns
            n_diag_blocks = self.submodels[0].nt

        for i in range(1, len(self.submodels)):
            if isinstance(self.submodels[i], RegressionSubModel):
                arrowhead_blocksize += self.submodels[i].n_latent_parameters

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param

    def construct_a_predict(self) -> sp.sparse.spmatrix:
        """Construct the design matrix for prediction."""

        data = []
        rows = []
        cols = []

        rows_a_predict = 0
        for i, submodel in enumerate(self.submodels):
            # Convert csc_matrix to coo_matrix to allow slicing
            coo_submodel_a_predict = submodel.load_a_predict().tocoo()
            data.append(coo_submodel_a_predict.data)
            rows.append(coo_submodel_a_predict.row)
            cols.append(
                coo_submodel_a_predict.col
                + self.latent_parameters_idx[i]
                * xp.ones(coo_submodel_a_predict.col.size, dtype=int)
            )

            # the number of rows in all of them is the same
            rows_a_predict = coo_submodel_a_predict.shape[0]

        self.a_predict: sp.sparse.spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(rows_a_predict, self.n_latent_parameters),
        )

    def total_number_fixed_effects(self) -> int:
        """Get the number of fixed effects."""
        return self.n_fixed_effects
