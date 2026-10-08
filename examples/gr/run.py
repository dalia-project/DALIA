import os
import sys

import numpy as np

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel
from dalia.utils import extract_diagonal, get_host, print_msg, plot_marginal_distributions_hp, plot_prior_hp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Regression ---")

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0},
        #"prior_hyperparameters": {"type": "gaussian", "mean": 1.0, "precision": 0.5},
    }
    # Creation of the first model by combining the Regression submodel and the likelihood
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    ## Plot prior of hyperparameter -- identification by [0], [1], ... not amazing but works for now
    theta_interval = [-5, 7]
    prior_hp = model.prior_hyperparameters[0]

    fig, ax = plot_prior_hp("prec_o", theta_interval, prior_hp)
    import matplotlib.pyplot as plt
    plt.show()

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref):.4e}",
    )

    # Compare latent parameters
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{xp.linalg.norm(results['x'] - x_ref):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = dalia.get_marginal_variances_observations(
        theta_external=theta_ref, x_star=x_ref
    )

    var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp() 

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt
    plt.savefig(f"gr_marginal_distributions_hp.png")

    prec_obs = marginals_hp['hyperparameters']['prec_o']
    quantile_pairs = prec_obs['quantiles']['external']['pairs']

    print("Quantile pairs of prec_o:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
