import sys
from pathlib import Path
import numpy as np

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_generators import generate_regression_data  # noqa: E402
from itest_utils import generate_data  # noqa: E402

# Values used to generate the data
N_OBS = 500
BETA = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5]

X_REL_TOL = 5e-2

def pr_itest():
    # The data is generated from scratch, no data file of the repository is used
    data_dir = generate_data(
        "pr",
        lambda data_dir: generate_regression_data(
            data_dir,
            likelihood="poisson",
            n_obs=N_OBS,
            beta=BETA,
        ),
    )
    x_original = np.load(f"{data_dir}/reference_outputs/x_original.npy")

    regression_dict = {
        "type": "regression",
        "input_dir": f"{data_dir}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{data_dir}",
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
        "simulation_dir": f"{data_dir}",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )
    results = dalia.run()

    # Compare latent parameters to the values used to generate the data
    x_dalia = get_host(results["x"])
    print(f"x_original: {x_original}")
    print(f"x_dalia: {x_dalia}")
    rel_err_x = np.linalg.norm(x_dalia - x_original) / np.linalg.norm(x_original)
    print_msg("Normalized norm (x - x_original): ", f"{rel_err_x:.4e}")
    if rel_err_x > X_REL_TOL:
        return "x_tol_exceeded"

    return "success"

if __name__ == "__main__":
    pr_itest()
