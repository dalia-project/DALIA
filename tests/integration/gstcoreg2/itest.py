import sys
from pathlib import Path
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
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg, get_host

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_generators import generate_coregional_spatio_temporal_data  # noqa: E402
from itest_utils import generate_data  # noqa: E402

# Values used to generate the data
NX, NY, NT = 8, 8, 6
N_OBS_PER_STEP = 100
R_S, R_T = [np.log(0.4), np.log(0.5)], [np.log(3.0), np.log(2.0)]
PREC_O = [4.0, 6.0]
SIGMAS, LAMBDA_0_1 = [np.log(1.5), np.log(1.0)], 0.8
BETAS = [[1.0], [-2.0]]

ETA_REL_TOL = 2e-1
THETA_TOL = 5e-1
TYPICAL_N_ITER = 23

def gstcoreg2_itest():
    # The data is generated from scratch, no data file of the repository is used
    data_dir = generate_data(
        "gstcoreg2",
        lambda data_dir: generate_coregional_spatio_temporal_data(
            data_dir,
            nx=NX,
            ny=NY,
            nt=NT,
            n_obs_per_step=N_OBS_PER_STEP,
            r_s=R_S,
            r_t=R_T,
            prec_o=PREC_O,
            sigmas=SIGMAS,
            lambda_0_1=LAMBDA_0_1,
            betas=BETAS,
        ),
    )
    theta_original = np.load(f"{data_dir}/reference_outputs/theta_original.npy")
    x_original = np.load(f"{data_dir}/reference_outputs/x_original.npy")
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
    theta_initial = theta_original + np.array(perturbation)

    # Configurations of the submodels for the first model
    # . Spatio-temporal submodel 1
    spatio_temporal_1_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{data_dir}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "r_t": theta_initial[1],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_original[0], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_original[1], 
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
        "input_dir": f"{data_dir}/model_1/inputs_regression",
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
        "input_dir": f"{data_dir}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3],
        "r_t": theta_initial[4],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_original[3], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_original[4], 
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
        "input_dir": f"{data_dir}/model_2/inputs_regression",
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
            "mean": theta_original[5],
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
            {"type": "gaussian", "mean": theta_original[6], "precision": 0.5},
            {"type": "gaussian", "mean": theta_original[7], "precision": 0.5},
        ],
        "ph_lambdas": [
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        ],
    }
    coreg_model = CoregionalModel(
        models=[model_1, model_2],
        coregional_model_config=models_config.parse_config(coreg_dict),
    )
    # Configurations of DALIA
    dalia_dict = {
        "solver": {
            "type": "serinv",
            "min_processes": 1,
        },
        "minimize": {
            "max_iter": 100,
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
        "simulation_dir": f"{data_dir}",
    }
    dalia = DALIA(
        model=coreg_model,
        config=dalia_config.parse_config(dalia_dict),
    )
    # Run the optimization
    results = dalia.run()

    # Check iterations behavior
    success_msg : str = "success"
    if results["optimization_iterations"] > TYPICAL_N_ITER:
        success_msg = "warning_more_iters_than_typical"
    elif results["optimization_iterations"] < TYPICAL_N_ITER:
        success_msg = "success_less_iters_than_typical"

    # Compare hyperparameters to the values used to generate the data
    theta_dalia = get_host(results["theta"])
    print(f"theta_original: {theta_original}")
    print(f"theta_dalia: {theta_dalia}")
    # . relative error for the large values, absolute error for the small ones
    err_theta = np.abs(theta_dalia - theta_original) / np.maximum(1.0, np.abs(theta_original))
    print_msg("Max error (theta - theta_original): ", f"{np.max(err_theta):.4e}")
    if np.max(err_theta) > THETA_TOL:
        return "theta_tol_exceeded"

    # Compare the linear predictor to the one used to generate the data
    x_original = x_original[get_host(dalia.model.permutation_latent_variables)]
    eta_original = get_host(dalia.model.a @ xp.asarray(x_original))
    eta_dalia = get_host(dalia.model.a @ results["x"])
    rel_err_eta = np.linalg.norm(eta_dalia - eta_original) / np.linalg.norm(eta_original)
    print_msg("Normalized norm (eta - eta_original): ", f"{rel_err_eta:.4e}")
    if rel_err_eta > ETA_REL_TOL:
        return "eta_tol_exceeded"

    return success_msg

if __name__ == "__main__":
    gstcoreg2_itest()
