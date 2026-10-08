import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from dalia import xp
from dalia.configs import (
    likelihood_config,
    models_config,
    dalia_config,
    submodels_config,
)
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel, SpatialSubModel
from examples_utils.parser_utils import parse_args

SEED = 63
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg(
        "--- Example: Gaussian Coregional (2 variates) spatial model with regression ---"
    )

    # Check for parsed parameters
    args = parse_args()

    nv = 2
    ns = 1818
    nt = 1
    nb = 2
    dim_theta = 7

    n = nv * (ns * nt) + nb

    theta_ref_file = f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy"
    theta_ref = np.load(theta_ref_file)
    theta_initial = theta_ref + 0.3 * np.random.randn(dim_theta)

    # Configurations of the submodels for the first model
    # . Spatial submodel 1
    spatial_1_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_1 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_1_dict),
    )
    # . Regression submodel 1
    regression_1_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_1 = RegressionSubModel(
        config=submodels_config.parse_config(regression_1_dict),
    )
    # . Likelihood submodel 1
    likelihood_1_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[1],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[1],
            "precision": 0.5,
        },
    }
    # Creation of the first model by combining the submodels and the likelihood
    model_1 = Model(
        submodels=[spatial_1, regression_1],
        likelihood_config=likelihood_config.parse_config(likelihood_1_dict),
    )

    # Configurations of the submodels for the second model
    # . Spatial submodel 2
    spatial_2_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[2],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_2 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_2_dict),
    )
    # . Regression submodel 2
    regression_2_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_2 = RegressionSubModel(
        config=submodels_config.parse_config(regression_2_dict),
    )
    # . Likelihood submodel 2
    likelihood_2_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[3],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[3],
            "precision": 0.5,
        },
    }
    # Creation of the second model by combining the submodels and the likelihood
    model_2 = Model(
        submodels=[spatial_2, regression_2],
        likelihood_config=likelihood_config.parse_config(likelihood_2_dict),
    )

    # Creation of the coregional model by combining the two models
    coreg_dict = {
        "type": "coregional",
        "n_models": 2,
        "sigmas": [theta_initial[4], theta_initial[5]],
        "lambdas": [theta_initial[6]],
        "ph_sigmas": [
            {"type": "gaussian", "mean": theta_ref[4], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[5], "precision": 0.5},
        ],
        "ph_lambdas": [
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        ],
    }
    coreg_model = CoregionalModel(
        models=[model_1, model_2],
        coregional_model_config=models_config.parse_config(coreg_dict),
    )
    print_msg(coreg_model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "eps_hessian_f": 5 * 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=coreg_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    results = dalia.run()

    print_msg("results['theta']: ", results["theta"])

    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    print_msg("mean of the fixed effects: ", results["x"][-nb:])
    print_msg(
        "marginal variances of the fixed effects: ",
        results["marginal_variances_latent"][-nb:],
    )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = np.load(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy")
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(results['theta'] - get_host(theta_ref)):.4e}",
    )

    x_ref = np.load(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref.npy")
    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(results['x'] - get_host(x_ref)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(
        eta=coreg_model.a @ coreg_model.x
    )
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
