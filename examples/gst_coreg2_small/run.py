import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import time

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
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import get_host, print_msg
from examples_utils.parser_utils import parse_args
from examples_utils.infos_utils import summarize_sparse_matrix

SEED = 63
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Coregional (2 variates) spatio-temporal model with regression ---")

    # Check for parsed parameters
    args = parse_args()

    nv = 2
    ns = 354
    nt = 12
    nb = 2
    dim_theta = 9

    n = nv * ns * nt + nb

    theta_ref_file = (
        f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy"
        # f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_interpretS_original_DALIA_perm_{dim_theta}_1.npy"
    )
    theta_ref = np.load(theta_ref_file)
    #theta_ref = np.loadtxt(theta_ref_file)

    perturbation = [
        0.18197867,
        -0.12551227,
        0.19998896,
        0.17226796,
        0.14656176,
        -0.11864931,
        0.17817371,
        -0.13006157,
        0.19308036,
    ]

    theta_initial = theta_ref + np.array(perturbation)

    print_msg(f"theta_initial: {theta_initial}")

    fixed_effects_ref = xp.array([-3.0, 8.0])

    # Configurations of the submodels for the first model
    # . Spatio-temporal submodel 1
    spatio_temporal_1_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "r_t": theta_initial[1],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_ref[0], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_ref[1], 
            "precision": 0.5,
        },
        "ph_st": {
            "type": "gaussian", 
            "mean": 0.0, 
            "precision": 0.5,
        },
    }
    spatio_temporal_1 = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_1_dict),
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
        "prec_o": theta_initial[2],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_initial[2],
            "precision": 0.5,
        },
    }
    # Creation of the first model by combining the submodels and the likelihood
    model_1 = Model(
        submodels=[regression_1, spatio_temporal_1],
        likelihood_config=likelihood_config.parse_config(likelihood_1_dict),
    )

    # Configurations of the submodels for the second model
    # . Spatio-temporal submodel 2
    spatio_temporal_2_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3],
        "r_t": theta_initial[4],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_ref[3], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_ref[4], 
            "precision": 0.5,
        },
        "ph_st": {
            "type": "gaussian", 
            "mean": 0.0, 
            "precision": 0.5,
        },
    }
    spatio_temporal_2 = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_2_dict),
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
        "prec_o": theta_initial[5],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[5],
            "precision": 0.5,
        },
    }
    # Creation of the second model by combining the submodels and the likelihood
    model_2 = Model(
        submodels=[spatio_temporal_2, regression_2],
        likelihood_config=likelihood_config.parse_config(likelihood_2_dict),
    )

    # Creation of the coregional model by combining the models
    coreg_dict = {
        "type": "coregional",
        "n_models": 2,
        "sigmas": [theta_initial[6], theta_initial[7]],
        "lambdas": [theta_initial[8]],
        "ph_sigmas": [
            {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
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
        "solver": {
            "type": "serinv",
            "min_processes": args.solver_min_p,
        },
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(coreg_model.theta_external),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
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

    # Run the optimization
    results = dalia.run()

    print_msg("\n--- DALIA results ---")

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
    #theta_ref = np.loadtxt(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_interpretS_original_pyPerm_9_1.dat")
    #np.save(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy", theta_ref)

    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(get_host(results['theta']) - get_host(theta_ref)):.4e}",
    )

    x_ref = np.load(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref.npy")
    #x_ref = np.loadtxt(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref_8498_1.dat")
    #np.save(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref.npy", x_ref)
    x_ref = x_ref[dalia.model.permutation_latent_variables]
    
    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(get_host(results['x']) - get_host(x_ref)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(
        eta=coreg_model.a @ coreg_model.x
    )
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{xp.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    # var_obs = dalia.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    # var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    # print_msg(
    #     "Norm (var_obs - var_obs_ref):    ",
    #     f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    # )

    print_msg("\n--- Finished ---")
