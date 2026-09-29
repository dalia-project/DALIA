from pathlib import Path

import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import ARSubModel, RegressionSubModel
from dalia.utils import print_msg, plot_marginal_distributions_hp

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(3)

    # load reference output
    theta_original = np.load(BASE_DIR / "reference_outputs" / "theta_original.npy")
    print(
        "theta original: ",
        theta_original,
    )

    x_original = np.load(BASE_DIR / "reference_outputs" / "x_original.npy")
    print("x original: ", x_original[:10])
    print("dim(x original): ", x_original.shape)

    # the beta prior is scaled to the support (-1, 1) and Beta(2, 2) is
    # symmetric around 0
    ph_pacf = {"type": "beta", "alpha": 2.0, "beta": 2.0, "support": [-1.0, 1.0]}

    ar_dict = {
        "type": "ar",
        "input_dir": f"{BASE_DIR}/inputs_ar",
        "order": 3,
        # initial guess of the partial autocorrelations, one per lag, each has
        # to be between -1 and 1 ...
        "pacf": [0.3, 0.1, 0.1],
        # ... and one prior per partial autocorrelation
        "ph_pacf": [ph_pacf, ph_pacf, ph_pacf],
        # initial guess on the marginal precision
        "tau": 3,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
    }
    ar = ARSubModel(
        config=submodels_config.parse_config(ar_dict),
    )

    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 20,
        # A gamma prior optimizes prec_o on the log scale, see the g_ar2 example.
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 0.01},
    }

    model = Model(
        submodels=[ar, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    Qprior = model.construct_Q_prior()
    print("Qprior: \n", Qprior.toarray()[:6, :6])
    Qinv = xp.linalg.inv(Qprior.toarray())
    print(
        "Marginal variances of the AR(3) prior (should all be 1 / tau): ",
        Qinv.diagonal()[:5],
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
        "simulation_dir": ".",
        "verbosity": 0,
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    print("initial model theta: ", model.theta_external)

    print("\nCalling DALIA.run()")
    results = dalia.run()

    print_msg("\n--- Results ---")

    theta = results["theta"]
    print("theta keys:     ", model.theta_keys)
    print("theta:          ", np.round(theta, 4))
    print("theta original: ", theta_original)

    print_msg("Covariance of theta:\n", results["cov_theta_internal"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg("\n--- Comparisons ---")
    print("norm(eta - eta_est): ", xp.linalg.norm(model.a @ xp.asarray(x_original) - model.a @ results["x"]))
    print("normalized norm(eta - eta_est): ", xp.linalg.norm(model.a @ xp.asarray(x_original) - model.a @ results["x"]) / xp.linalg.norm(model.a @ xp.asarray(x_original)))

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{xp.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    # derived=True adds the marginal distributions of the AR coefficients phi,
    # which are sampled from the joint distribution of the partial autocorrelations
    marginals_hp = dalia.marginal_distributions_hp(
        derived=True,
        n_samples=10000,
        seed=0,
    )

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt
    plt.show()

    for key in ["pacf1", "pacf2", "pacf3"]:
        quantile_pairs = marginals_hp["hyperparameters"][key]["quantiles"]["external"]["pairs"]
        print(f"Quantile pairs of {key}:")
        for p, q in quantile_pairs:
            print(f"   {p:.3f} quantile: {q:.4f}")

    for key, marginal in marginals_hp["derived"].items():
        print(f"Quantile pairs of {key}:")
        for p, q in marginal["quantiles"]["external"]["pairs"]:
            print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
