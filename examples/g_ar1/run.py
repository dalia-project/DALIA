from pathlib import Path

import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import print_msg, plot_marginal_distributions_hp, plot_prior_hp  # , extract_diagonal

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(3)

    # load reference output
    theta_original = np.load(BASE_DIR / "reference_outputs" / "theta_original.npy")
    print(
        "theta original: ",
        theta_original,
    )

    theta_initial = theta_original #[0.6, 1.0, 3.0]
    print("theta initial: ", theta_initial)

    x_original = np.load(BASE_DIR / "reference_outputs" / "x_original.npy")
    print("x original: ", x_original[:10])
    print("dim(x original): ", x_original.shape)

    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{BASE_DIR}/inputs_ar1",
        "phi": 0.5,  # has to be between 0 and 1
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        # initial guess on the precision
        "tau": 3, # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        # initial guess on the variance
        # "sigma2": 0.33, # has to be positive
        # "ph_sigma2": {"type": "invgamma", "alpha": 2.0, "beta": 1.0}, 
    }
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(ar1_dict),
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
        # "prior_hyperparameters": {
        #     "type": "penalized_complexity",
        #     "alpha": 0.01,
        #     "u": 5,
        # },
        "prior_hyperparameters": {"type": "gaussian", "mean": theta_original[2], "precision": 0.05},
    }

    model = Model(
        submodels=[ar1, regression], #
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)
    
    # plot phi
    # theta_interval = [0, 1]
    # prior_hp = model.prior_hyperparameters[0]
    # fig, ax = plot_prior_hp("phi", theta_interval, prior_hp)

    # plot tau
    theta_interval = [0, 5]
    prior_hp = model.prior_hyperparameters[1]
    fig, ax = plot_prior_hp("tau", theta_interval, prior_hp)

    import matplotlib.pyplot as plt
    plt.show()

    Qprior = model.construct_Q_prior()
    print("Qprior: \n", Qprior.toarray()[:6, :6])
    Qinv = xp.linalg.inv(Qprior.toarray())
    geom_mean = xp.exp(xp.mean(xp.log(Qinv.diagonal())))
    print("Geometric mean of Qinv diagonal: ", geom_mean)

    # in gaussian case x = 0, thus eta = 0
    x_i = xp.zeros(model.n_latent_parameters)
    eta = model.a @ x_i
    Qcond = model.construct_Q_conditional(eta=eta)
    print("Qcond: \n", Qcond.toarray()[:6, :6])

    b = model.construct_information_vector(eta=eta, x_i=x_i)
    print("b: ", b[:10])

    x_est = xp.linalg.solve(Qcond.toarray(), b)
    #print("x est: ", x_est)
    print("norm(x_original - x_est): ", xp.linalg.norm(xp.asarray(x_original) - x_est))

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
    print("theta:          ", np.round(theta, 4))
    print("theta original: ", theta_original)

    print_msg("Covariance of theta:\n", results["cov_theta_internal"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )

    # print("x:          ", results["x"])
    # print("x_original: ", x_original)

    # print("eta: ", model.a @ x_original)
    # print("eta est: ", model.a @ results["x"])

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
    marginals_hp = dalia.marginal_distributions_hp() 

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt
    plt.show()
    
    phi = marginals_hp['hyperparameters']['phi']
    quantile_pairs = phi['quantiles']['external']['pairs']

    print("Quantile pairs of phi:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")
    
    print_msg("\n--- Finished ---")