from pathlib import Path
import numpy as np

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

SCRIPT_DIR = Path(__file__).resolve()
DALIA_DIR = SCRIPT_DIR.parent.parent.parent.parent
EXAMPLE_PATH = DALIA_DIR / "examples" / "gst_medium"

X_TOL = 1e-1
THETA_TOL = 1e1
TYPICAL_N_ITER = 26

def gst_itest():
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{EXAMPLE_PATH}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0.0,
        "r_t": 2.2,
        "sigma_st": 1.3,
        "manifold": "sphere",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )
    # . Regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{EXAMPLE_PATH}/inputs_regression",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    # Configurations of the likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4,
        "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
    }
    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
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
        },
        "f_reduction_tol": 1e-4,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": f"{EXAMPLE_PATH}",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )
    results = dalia.run()

    # Check iterations behavior
    success_msg : str = "success"
    if results["optimization_iterations"] > TYPICAL_N_ITER:
        success_msg = "warning_more_iters_than_typical"
    elif results["optimization_iterations"] < TYPICAL_N_ITER:
        success_msg = "success_less_iters_than_typical"

    # Compare hyperparameters
    theta_ref = np.array(np.load(f"{EXAMPLE_PATH}/reference_outputs/theta_ref.npy"))
    print(f"theta_ref: {theta_ref}")
    print(f"theta_dalia: {get_host(results['theta_internal'])}")
    print_msg(
        "Norm (theta - theta_ref): ",
        f"{np.linalg.norm(get_host(results['theta_internal']) - theta_ref):.4e}",
    )
    if np.linalg.norm(get_host(results["theta_internal"]) - theta_ref) > THETA_TOL:
        return "theta_tol_exceeded"
    
    # Compare latent parameters
    x_ref = np.array(np.load(f"{EXAMPLE_PATH}/reference_outputs/x_ref.npy"))
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
    gst_itest()