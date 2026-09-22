import os
import sys


from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config, models_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models import ReplicateModel
from dalia.submodels import LKJSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: LKJ Submodel with Multiple Replicates ---")

    # Check for parsed parameters
    args = parse_args()

    n_replicates = 500  # Must match n_replicates in generate_data.py

    # Create a model for each replicate
    models = []
    for i in range(n_replicates):
        # Configurations of the LKJ submodel
        # 2D latent with LKJ prior on correlation and HalfNormal on variances
        lkj_dict = {
            "type": "lkj",
            "input_dir": f"{BASE_DIR}/inputs_nrep{n_replicates}/replicate_{i+1}/inputs_lkj",
            "sigma1": 1.0,
            "sigma2": 1.0,
            "rho": 0.0,
            # Prior hyperparameters
            "ph_sigma1": {"type": "half_normal", "precision": 0.5},  # variance = 2
            "ph_sigma2": {"type": "half_normal", "precision": 0.5},  # variance = 2
            "ph_rho": {"type": "lkj_2d", "eta": 1.0},
        }
        lkj = LKJSubModel(
            config=submodels_config.parse_config(lkj_dict),
        )

        # Likelihood
        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 1.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
        }
        # Creation of the model by combining the LKJ submodel and the likelihood
        local_model = Model(
            submodels=[lkj],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
        models.append(local_model)

    print_msg(models[0])

    # Create ReplicateModel to combine all replicates
    replicate_dict = {
        "type": "replicate",
        "n_replicates": len(models),
        "theta": models[0].theta_external.tolist(),
        "theta_keys": list(models[0].theta_keys),
    }
    replicate_model = ReplicateModel(
        models=models,
        replicate_model_config=models_config.parse_config(replicate_dict),
    )

    print_msg(f"Internal theta values: {replicate_model.theta_internal}")
    print_msg(f"External theta values: {replicate_model.theta_external}")
    Qprior = replicate_model.construct_Q_prior()

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=replicate_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/theta_ref.npy")
    x_ref = xp.load(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("theta reference:\n", theta_ref)
    theta_est = results["theta"]
    print_msg("theta estimated:\n", theta_est)
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{xp.linalg.norm(results['x'] - x_ref):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(
        eta=replicate_model.a @ replicate_model.x
    )
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{xp.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = dalia.get_marginal_variances_observations(
        theta_external=theta_ref, x_star=x_ref
    )
    var_obs_ref = extract_diagonal(replicate_model.a @ Qinv_ref @ replicate_model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )

    ## construct estimated LKJ covariance matrix of latent parameters
    lkj_est = lkj.lkj_covariance_matrix(theta_est[:3])
    print("LKJ estimated covariance matrix: \n", lkj_est)

    lkj_ref = lkj.lkj_covariance_matrix(theta_ref[:3])
    print_msg("\n--- LKJ Covariance Matrix of Latent Parameters ---")
    print_msg("Estimated LKJ covariance matrix:\n", lkj_est)
    print_msg("Reference LKJ covariance matrix:\n", lkj_ref)

    print_msg("\n--- Finished ---")
