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
from dalia.utils import (
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg(
        "--- Example: Federated Model Regression (optional: site specific intercept) ---"
    )

    # Check for parsed parameters
    args = parse_args()

    # random site-specific intercept
    random_intercept = True  # True

    data_type = "nurses_hom"
    family = "gaussian"
    # covariates excluding intercept
    # must match the covariates written by preprocess_data_w_random_intercept.py
    covariate_names = ["gender", "age", "experience", "wardtype"]

    if random_intercept:
        n_fixed_effects = len(covariate_names)  # no global intercept, only covariates
        split_folder = f"split_{data_type}_{family}_site_specific_intercept"
        intercept_tag = "site_specific_intercept"
    else:
        n_fixed_effects = 1 + len(covariate_names)  # includes global intercept
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
    hospital_ids = [int(os.path.basename(p).split("_")[-1]) for p in hospital_dirs]

    if len(hospital_dirs) == 0:
        raise ValueError(f"No hospital folders found in {split_root}.")

    first_a = np.load(os.path.join(hospital_dirs[0], "inputs_regression", "a.npy"))
    print_msg(f"Detected n_fixed_effects={n_fixed_effects} from first hospital a.npy")

    models = []
    for hospital_dir in hospital_dirs:
        regression_dict = {
            "type": "regression",
            "input_dir": f"{hospital_dir}/inputs_regression",
            "n_fixed_effects": n_fixed_effects,
            "fixed_effects_prior_precision": 0.001,
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
                "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 0.1},
            }
            generic = GenericSubModel(
                config=submodels_config.parse_config(generic_dict),
            )
            submodels = [generic, regression]

        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 1.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 0.1},
            # "prior_hyperparameters": {"type": "gaussian", "mean": 1.0, "precision": 0.5},
        }
        model_local = Model(
            submodels=submodels,
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
            input_dir=hospital_dir,
        )
        models.append(model_local)

    print_msg(f"Constructed {len(models)} local models from split data")

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
            "max_iter": 100,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=federated_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    fixed_effects_mean = results["x"][-federated_model.n_fixed_effects :]
    print_msg(
        "Mean of the fixed effects:\n",
        fixed_effects_mean,
    )
    if random_intercept:
        n_random = models[0].submodels[0].n_latent_parameters
        random_effects_mean = results["x"][:n_random]
        print_msg("Mean of the random intercepts:\n", random_effects_mean)

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    fixed_effects_var = var_latent_params[-federated_model.n_fixed_effects :]
    fixed_effects_sd = np.sqrt(fixed_effects_var)
    fixed_ci_lower = fixed_effects_mean - 1.96 * fixed_effects_sd
    fixed_ci_upper = fixed_effects_mean + 1.96 * fixed_effects_sd

    marginals_hp = dalia.marginal_distributions_hp()
    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]

    ## convert to sigma2 values
    sigma2_quantile_pairs = sorted(
        ((1.0 - p, 1.0 / q) for p, q in quantile_pairs),
        key=lambda pair: pair[0],
    )

    # store parameters in matching format as needed in R
    # make dataframe with columns:
    # - lower (2.5% quantile)
    # - upper (97.5% quantile)
    # - mean (mean of the fixed effect)
    # - Method (DALIA)
    # - covariate (intercept), gender, age, experience, wardtype, sigma2
    sigma2_lower = sigma2_quantile_pairs[0][1]
    sigma2_upper = sigma2_quantile_pairs[-1][1]
    prec_o_idx = list(federated_model.theta_keys).index("prec_o")
    sigma2_mean = 1.0 / float(results["theta"][prec_o_idx])

    if random_intercept:
        random_sd = np.sqrt(var_latent_params[:n_random])
        random_ci_lower = random_effects_mean - 1.96 * random_sd
        random_ci_upper = random_effects_mean + 1.96 * random_sd
        fixed_covariates = covariate_names
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
                "Covariate": "tau",
            }
        ]
    else:
        random_effects_mean = np.array([])
        random_ci_lower = np.array([])
        random_ci_upper = np.array([])
        fixed_covariates = ["(Intercept)"] + covariate_names
        random_covariates = []
        tau_rows = []

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
    summary_rows.append(
        {
            "Estimate": sigma2_mean,
            "lower": sigma2_lower,
            "upper": sigma2_upper,
            "Method": "DALIA-FED",
            "Covariate": "sigma2",
        }
    )

    df_dalia = pd.DataFrame(summary_rows)
    df_dalia.to_csv(
        f"{BASE_DIR}/dalia_summary_{data_type}_{intercept_tag}.csv",
        index=False,
    )
    print_msg(df_dalia)

    # Predict using global federated a_predict and construct hospital_idx from local row counts.
    rows_per_hospital = []
    for hospital_id, model_local in zip(hospital_ids, models):
        try:
            model_local.construct_a_predict()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing apr.npz for hospital_{hospital_id}. Please run preprocessing first."
            )
        rows_per_hospital.append(model_local.a_predict.shape[0])

    a_predict_global = federated_model.construct_a_predict()
    eta_predict_global = np.asarray(
        a_predict_global @ results["x"], dtype=float
    ).reshape(-1)

    hospital_idx_vec = np.repeat(np.array(hospital_ids), rows_per_hospital)

    if hospital_idx_vec.shape[0] != eta_predict_global.shape[0]:
        raise ValueError(
            "Mismatch between hospital_idx vector length and eta_predict rows: "
            f"{hospital_idx_vec.shape[0]} vs {eta_predict_global.shape[0]}"
        )

    df_predict = pd.DataFrame(
        {
            "hospital_idx": hospital_idx_vec,
            "mean_predict": eta_predict_global,
        }
    )

    if not df_predict.empty:
        predictions_path = (
            f"{BASE_DIR}/dalia_predictions_{data_type}_{intercept_tag}.csv"
        )
        df_predict.to_csv(predictions_path, index=False)
        print_msg(f"Saved predictions to {predictions_path}")
        print_msg(df_predict.head(10))
    else:
        print_msg("No prediction rows generated. Make sure apr.npz files exist.")

    print_msg("\n--- Finished ---")
