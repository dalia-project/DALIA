import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from dalia import xp, backend_flags
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatialSubModel
from dalia.utils import get_host, print_msg  # , extract_diagonal
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    print_msg("--- Example: Gaussian spatial model with regression ---")
    args = parse_args()

    spatial_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": 0,  #
        "sigma_e": 0,
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.6},
        "ph_e": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
    }
    spatial = SpatialSubModel(
        config=submodels_config.parse_config(spatial_dict),
    )

    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 5,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4,
        # "prior_hyperparameters": {
        #     "type": "penalized_complexity",
        #     "alpha": 0.01,
        #     "u": 5,
        # },
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0},
    }
    model = Model(
        submodels=[spatial, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print_msg(model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "scipy"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(model.theta_external),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", results["theta"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    print_msg(
        "Mean of the fixed effects:\n",
        get_host(results["x"][-model.submodels[-1].n_fixed_effects :]),
    )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(get_host(results['theta_internal']) - get_host(theta_ref)):.4e}",
    )

    # Compare latent parameters
    x_ref = np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(get_host(results['x']) - get_host(x_ref)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    # var_obs = dalia.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    # var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    # print_msg(
    #     "Norm (var_obs - var_obs_ref):    ",
    #     f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    # )

    print_msg("\n--- Finished ---")
