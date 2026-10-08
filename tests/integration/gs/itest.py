import sys
from pathlib import Path
import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.utils import print_msg, get_host
from dalia.submodels import RegressionSubModel, SpatialSubModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_generators import generate_spatial_data  # noqa: E402
from itest_utils import generate_data  # noqa: E402

# Values used to generate the data
NX, NY = 15, 15
N_OBS = 600
R_S, SIGMA_E = np.log(0.4), np.log(1.5)
BETA = [1.0, -2.0, 0.5, 3.0, -1.5]
PREC_O = 4.0

ETA_REL_TOL = 1e-1
THETA_TOL = 5e-1
MARG_VAR_TOL = 1e-6
TYPICAL_N_ITER = 12

def gs_itest():
    # The data is generated from scratch, no data file of the repository is used
    data_dir = generate_data(
        "gs",
        lambda data_dir: generate_spatial_data(
            data_dir,
            nx=NX,
            ny=NY,
            n_obs=N_OBS,
            r_s=R_S,
            sigma_e=SIGMA_E,
            beta=BETA,
            prec_o=PREC_O,
        ),
    )
    theta_original = np.load(f"{data_dir}/reference_outputs/theta_original.npy")
    x_original = np.load(f"{data_dir}/reference_outputs/x_original.npy")

    spatial_dict = {
        "type": "spatial",
        "input_dir": f"{data_dir}/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": 0,
        "sigma_e": 0,
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.1},
        "ph_e": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
    }
    spatial = SpatialSubModel(
        config=submodels_config.parse_config(spatial_dict),
    )
    regression_dict = {
        "type": "regression",
        "input_dir": f"{data_dir}/inputs_regression",
        "n_fixed_effects": 5,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0},
    }
    model = Model(
        submodels=[spatial, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "scipy"},
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
        "simulation_dir": f"{data_dir}",
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

    # Compare marginal variances of latent parameters against a dense inverse
    var_latent_params = get_host(results["marginal_variances_latent"])
    dalia.model.theta_internal = results["theta_internal"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - get_host(xp.diag(Qinv_ref))):.4e}",
    )
    if np.linalg.norm(var_latent_params - get_host(xp.diag(Qinv_ref))) > MARG_VAR_TOL:
        return "marg_var_tol_exceeded"

    return success_msg

if __name__ == "__main__":
    gs_itest()
