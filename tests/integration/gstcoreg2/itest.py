from pathlib import Path
import numpy as np

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

SCRIPT_DIR = Path(__file__).resolve()
DALIA_DIR = SCRIPT_DIR.parent.parent.parent.parent
EXAMPLE_PATH = DALIA_DIR / "examples" / "gst_coreg2_small"

X_TOL = 1e3 # 1.5e+2
THETA_TOL = 1e2 # 9.e+1
TYPICAL_N_ITER = 80 # On Fritz: 86

def gstcoreg2_itest():
    nv = 2
    ns = 354
    nt = 12
    nb = 2

    theta_ref_file = (
        f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy"
    )
    theta_ref = np.load(theta_ref_file)
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

    # Configurations of the submodels for the first model
    # . Spatio-temporal submodel 1
    spatio_temporal_1_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatio_temporal",
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
        "input_dir": f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_regression",
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
        "input_dir": f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatio_temporal",
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
        "input_dir": f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_regression",
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
        "simulation_dir": f"{EXAMPLE_PATH}",
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

    # Compare hyperparameters
    theta_ref = np.load(f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy")
    print(f"theta_ref: {theta_ref}")
    print(f"theta_dalia: {get_host(results['theta_internal'])}")
    print_msg(
        "Norm (theta - theta_ref): ",
        f"{np.linalg.norm(get_host(results['theta_internal']) - theta_ref):.4e}",
    )
    if np.linalg.norm(get_host(results["theta_internal"]) - theta_ref) > THETA_TOL:
        return "theta_tol_exceeded"

    # Compare latent parameters
    x_ref = np.load(f"{EXAMPLE_PATH}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref.npy")
    x_ref = x_ref[dalia.model.permutation_latent_variables]
    print(f"x_ref: {x_ref}")
    print(f"x_dalia: {get_host(results['x'])}")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(get_host(results['x']) - x_ref):.4e}",
    )
    if np.linalg.norm(get_host(results["x"]) - x_ref) > X_TOL:
        return "x_tol_exceeded"

    return success_msg

if __name__ == "__main__":
    gstcoreg2_itest()   
