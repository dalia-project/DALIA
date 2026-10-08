import sys
from pathlib import Path
import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_generators import generate_spatio_temporal_data  # noqa: E402
from itest_utils import generate_data  # noqa: E402

# Values used to generate the data
NX, NY, NT = 10, 10, 10
N_OBS_PER_STEP = 200
R_S, R_T, SIGMA_ST = np.log(0.4), np.log(3.0), np.log(0.7)
BETA = [1.0, -0.5, 0.5, 0.8, -0.3, 0.4, -0.6, 0.2]

ETA_REL_TOL = 2e-1
THETA_TOL = 5e-1

def pst_itest():
    # The data is generated from scratch, no data file of the repository is used
    data_dir = generate_data(
        "pst",
        lambda data_dir: generate_spatio_temporal_data(
            data_dir,
            likelihood="poisson",
            nx=NX,
            ny=NY,
            nt=NT,
            n_obs_per_step=N_OBS_PER_STEP,
            r_s=R_S,
            r_t=R_T,
            sigma_st=SIGMA_ST,
            beta=BETA,
        ),
    )
    theta_original = np.load(f"{data_dir}/reference_outputs/theta_original.npy")
    x_original = np.load(f"{data_dir}/reference_outputs/x_original.npy")

    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{data_dir}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0,
        "r_t": 0,
        "sigma_st": 0,
        "manifold": "plane",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.1},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 1},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )
    regression_dict = {
        "type": "regression",
        "input_dir": f"{data_dir}/inputs_regression",
        "n_fixed_effects": 8,
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
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "serinv"},
        "minimize": {
            "max_iter": 100,
            "gtol": 1e-3,
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
    results = dalia.minimize()

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
    eta_original = get_host(model.a @ xp.asarray(x_original))
    eta_dalia = get_host(model.a @ results["x"])
    rel_err_eta = np.linalg.norm(eta_dalia - eta_original) / np.linalg.norm(eta_original)
    print_msg("Normalized norm (eta - eta_original): ", f"{rel_err_eta:.4e}")
    if rel_err_eta > ETA_REL_TOL:
        return "eta_tol_exceeded"

    return "success"

if __name__ == "__main__":
    pst_itest()
