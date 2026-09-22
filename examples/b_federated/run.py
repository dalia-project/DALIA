import os
import sys

import numpy as np
import pandas as pd

from dalia.configs import (
    dalia_config,
    likelihood_config,
    models_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models.federated_model import FederatedModel
from dalia.submodels import GenericSubModel, RegressionSubModel
from dalia.utils import print_msg

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Federated Binomial Model ---")

    # Check for parsed parameters
    args = parse_args()

    # random site-specific intercept
    random_intercept = True  # True

    data_type = "trauma"
    family = "binomial"
    if random_intercept:
        n_fixed_effects = 4  # no global intercept, only covariates
        split_folder = f"split_{data_type}_{family}_site_specific_intercept"
        intercept_tag = "site_specific_intercept"
    else:
        n_fixed_effects = 5  # includes global intercept
        split_folder = f"split_{data_type}_{family}_global_intercept"
        intercept_tag = "global_intercept"

    split_root = os.path.join(BASE_DIR, split_folder)
    hospital_dirs = [
        os.path.join(split_root, d)
        for d in os.listdir(split_root)
        if d.startswith("hospital_") and os.path.isdir(os.path.join(split_root, d))
    ]
    hospital_dirs = sorted(
        hospital_dirs,
        key=lambda p: int(os.path.basename(p).split("_")[-1]),
    )
    if len(hospital_dirs) == 0:
        raise ValueError(f"No hospital folders found in {split_root}.")

    models = []
    for hospital_dir in hospital_dirs:
        regression_dict = {
            "type": "regression",
            "input_dir": f"{hospital_dir}/inputs_regression",
            "n_fixed_effects": n_fixed_effects,
            "fixed_effects_prior_precision": 0.1,
        }
        regression = RegressionSubModel(
            config=submodels_config.parse_config(regression_dict),
        )

        submodels = [regression]
        if random_intercept:
            generic_dict = {
                "type": "generic",
                "input_dir": f"{hospital_dir}/inputs_generic",
                "tau": 4,
                "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-5},
            }
            generic = GenericSubModel(
                config=submodels_config.parse_config(generic_dict),
            )
            submodels = [generic, regression]

        likelihood_dict = {
            "type": "binomial",
            "input_dir": hospital_dir,
        }
        model_local = Model(
            submodels=submodels,
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
        models.append(model_local)

    print_msg(f"Constructed {len(models)} local models from split data")

    # Federated setup from local models.
    federated_dict = {
        "type": "federated",
        "n_models": len(models),
        "theta": models[0].theta_external.tolist(),
        "theta_keys": list(models[0].theta_keys),
    }
    federated_model = FederatedModel(
        models=models,
        federated_model_config=models_config.parse_config(federated_dict),
    )

    print_msg(federated_model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-8,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=federated_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    results = dalia.run()

    print_msg("\n--- Results ---")
    fixed_effects_mean = results["x"][-federated_model.n_fixed_effects :]
    print_msg("Mean of the fixed effects:\n", fixed_effects_mean)

    if random_intercept:
        n_random = models[0].submodels[0].n_latent_parameters
        random_effects_mean = results["x"][:n_random]
        print_msg("Mean of the random intercepts:\n", random_effects_mean)

    # Summarize fixed/random effects and save to file.
    var_latent_params = results["marginal_variances_latent"]
    marginals_hp = dalia.marginal_distributions_hp()

    if random_intercept:
        random_sd = np.sqrt(var_latent_params[:n_random])
        random_ci_lower = random_effects_mean - 1.96 * random_sd
        random_ci_upper = random_effects_mean + 1.96 * random_sd

        fixed_sd = np.sqrt(
            var_latent_params[n_random : n_random + federated_model.n_fixed_effects]
        )
        fixed_covariates = ["sex", "age", "ISS", "GCS"]
        random_covariates = [f"site_intercept_{idx}" for idx in range(1, n_random + 1)]

        tau_idx = list(federated_model.theta_keys).index("tau")
        tau_estimate = float(results["theta"][tau_idx])
        tau_quantile_pairs = marginals_hp["hyperparameters"]["tau"]["quantiles"][
            "external"
        ]["pairs"]
        tau_ci_lower = float(tau_quantile_pairs[0][1])
        tau_ci_upper = float(tau_quantile_pairs[-1][1])
        tau_rows = [
            {
                "Estimate": tau_estimate,
                "lower": tau_ci_lower,
                "upper": tau_ci_upper,
                "Method": "DALIA-FED",
                "Covariate": "precision_random_intercept",
            }
        ]
    else:
        fixed_sd = np.sqrt(var_latent_params[: federated_model.n_fixed_effects])
        fixed_covariates = ["(Intercept)", "sex", "age", "ISS", "GCS"]
        random_effects_mean = np.array([])
        random_ci_lower = np.array([])
        random_ci_upper = np.array([])
        random_covariates = []
        tau_rows = []

    fixed_ci_lower = fixed_effects_mean - 1.96 * fixed_sd
    fixed_ci_upper = fixed_effects_mean + 1.96 * fixed_sd

    summary_rows = []
    for covariate, lower, upper, estimate in zip(
        fixed_covariates, fixed_ci_lower, fixed_ci_upper, fixed_effects_mean
    ):
        summary_rows.append(
            {
                "Estimate": estimate,
                "lower": lower,
                "upper": upper,
                "Method": "DALIA-FED",
                "Covariate": covariate,
            }
        )

    for covariate, lower, upper, estimate in zip(
        random_covariates, random_ci_lower, random_ci_upper, random_effects_mean
    ):
        summary_rows.append(
            {
                "Estimate": estimate,
                "lower": lower,
                "upper": upper,
                "Method": "DALIA-FED",
                "Covariate": covariate,
            }
        )

    summary_rows.extend(tau_rows)

    df_dalia = pd.DataFrame(summary_rows)
    summary_path = f"{BASE_DIR}/dalia_summary_{data_type}_{intercept_tag}.csv"
    df_dalia.to_csv(summary_path, index=False)
    print_msg(f"Saved summary to {summary_path}")
    print_msg(df_dalia)

    print_msg("\n--- Finished ---")
