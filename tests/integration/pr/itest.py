from pathlib import Path
import numpy as np

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel

SCRIPT_DIR = Path(__file__).resolve()
DALIA_DIR = SCRIPT_DIR.parent.parent.parent.parent
EXAMPLE_PATH = DALIA_DIR / "examples" / "pr"

X_TOL = 1e-5

def pr_itest():
    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{EXAMPLE_PATH}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    # Likelihood
    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{EXAMPLE_PATH}",
    }
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 100,
            "gtol": 1e-1,
            "disp": True,
        },
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

    # Compare latent parameters
    x_ref = np.load(f"{EXAMPLE_PATH}/reference_outputs/x_ref.npy")
    print(f"x_ref: {x_ref}")
    print(f"x_dalia: {get_host(results['x'])}")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(get_host(results['x']) - x_ref):.4e}",
    )
    if np.linalg.norm(get_host(results["x"]) - x_ref) > X_TOL:
        return "x_tol_exceeded"

    return "success"

if __name__ == "__main__":
    pr_itest()