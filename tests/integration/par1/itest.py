from pathlib import Path
import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel, AR1SubModel

SCRIPT_DIR = Path(__file__).resolve()
DALIA_DIR = SCRIPT_DIR.parent.parent.parent.parent
EXAMPLE_PATH = DALIA_DIR / "examples" / "p_ar1"

X_TOL = 1e2
THETA_TOL = 2e-2
TYPICAL_N_ITER = 12

def par1_itest():
    # load reference output
    theta_original = np.load(f"{EXAMPLE_PATH}/reference_outputs/theta_original.npy")
    x_original = np.load(f"{EXAMPLE_PATH}/reference_outputs/x_original.npy")

    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{EXAMPLE_PATH}/inputs_ar1",
        "phi": 0.45,  # has to be between 0 and 1
        "tau": 0.5,  # precision 
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 0.5},
    }
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(ar1_dict),
    )
    
    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{EXAMPLE_PATH}/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{EXAMPLE_PATH}",
    }

    model = Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 100,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(model.theta_external),
        },
        "f_reduction_tol": 1e-3,
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
    theta_user = get_host(results["theta"])
    print("theta_ref: ", theta_original)
    print("theta user: ", theta_user)
    print_msg(
        "Norm (theta - theta_ref): ",
        f"{np.linalg.norm(theta_user - theta_original):.4e}",
    )
    if np.linalg.norm(theta_user - theta_original) > THETA_TOL:
        return "theta_tol_exceeded"

    # Compare latent parameters
    print(f"x_ref: {x_original}")
    print(f"x_dalia: {get_host(results['x'])}")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(get_host(results['x']) - x_original):.4e}",
    )
    if np.linalg.norm(get_host(results["x"]) - x_original) > X_TOL:
        return "x_tol_exceeded"

    return success_msg

if __name__ == "__main__":
    par1_itest()
