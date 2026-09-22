import os
import sys

import numpy as np

from dalia import xp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
    save_to_json,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Regression with multiple replicates ---")

    save_dalia_results = False  # Set to True to save results to JSON
    # Check for parsed parameters
    args = parse_args()

    n_replicates = 20  # Must match n_replicates in generate_data.py

    # Configurations of the regression submodel with built-in replication
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nrep{n_replicates}/inputs_regression",
        "n_fixed_effects": 6,
        "n_replicates": n_replicates,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 1e-1},
    }
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
        input_dir=f"{BASE_DIR}/inputs_nrep{n_replicates}",
    )
    print_msg(model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(
        f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/theta_ref.npy"
    )
    x_ref = xp.load(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref) / ||x_ref||:    ",
        f"{xp.sqrt(xp.sum((results['x'] - x_ref) ** 2)) / xp.sqrt(xp.sum(x_ref ** 2)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(
        eta=model.a @ results["x"]
    )
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = dalia.get_marginal_variances_observations()

    var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )
        
    # save estimates to reference outputs folder
    if save_dalia_results:
        save_to_json(
            results=results,
            filename=f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/dalia_estimates.json",
        )

    print_msg("\n--- Finished ---")
