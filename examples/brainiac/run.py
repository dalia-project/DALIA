import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import BrainiacSubModel
from dalia.utils import print_msg
from plotting import plot_marginal_distributions_hp_external

if __name__ == "__main__":
    print_msg(f"Running BRAINIAC model on synthetic dataset.")

    path_inputs: Path = Path(__file__).parent / "inputs_brainiac"
    path_reference: Path = path_inputs / "reference"

    # 1. Load model parameters
    model_params: dict = np.load(
        path_inputs / "model_params.npy", allow_pickle=True
    ).item()
    n_observations: int = model_params["n_observations"]
    n_features: int = model_params["n_features"]
    n_annotations_per_features: int = model_params["n_annotations_per_features"]
    h2: float = model_params["h2"]
    sigma_a2: float = 5.0

    # 2. Load references
    theta_reference: np.ndarray = np.load(path_reference / "theta.npy")
    x_reference: np.ndarray = np.load(path_reference / "beta.npy")

    # 3. Create starting values for DALIA
    initial_h2: float = theta_reference[0] - 0.1
    initial_alpha: np.ndarray = theta_reference[1:] + 0.5 * np.random.randn(
        n_annotations_per_features
    )

    # 4. Initialize the Brainiac submodel and the DALIA model
    brainiac = BrainiacSubModel(
        config=submodels_config.parse_config(
            {
                "type": "brainiac",
                "input_dir": str(path_inputs.resolve()),
                "h2": initial_h2,
                "alpha": initial_alpha,
                "ph_h2": {"type": "beta", "alpha": 1.0, "beta": 1.0},
                "ph_alpha": {
                    "type": "gaussian_mvn",
                    "mean": xp.zeros(n_annotations_per_features),
                    "precision": (1.0 / sigma_a2) * sp.eye(n_annotations_per_features),
                },
            }
        )
    )
    model = Model(
        submodels=[brainiac],
        likelihood_config=likelihood_config.parse_config(
            {"type": "gaussian", "fix_hyperparameters": True}
        ),
    )
    print_msg(model)

    # 5. Initialize DALIA
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(
            {
                "solver": {"type": "dense"},
                "minimize": {
                    "max_iter": 50,
                    "gtol": 1e-3,
                    "disp": True,
                },
                "inner_iteration_max_iter": 50,
                "eps_inner_iteration": 1e-3,
                "eps_gradient_f": 1e-3,
                "simulation_dir": ".",
            }
        ),
    )

    # 6. Run inference
    tic = time.perf_counter()
    result = dalia.run()
    toc = time.perf_counter()
    print_msg(f"DALIA finished in {toc - tic:.2f} seconds.")

    # 7. Compare to reference solution
    print_msg("\n------ Compare to reference solution ------\n")
    print_msg("theta_reference: ", theta_reference)
    print_msg("theta dalia:", result["theta"])

    print_msg("norm(x_reference - x) = ", np.linalg.norm(xp.asarray(x_reference) - result["x"]))

    # 8. Check marginal variances of latent parameters
    var_latent_params = result["marginal_variances_latent"]
    Q_conditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)

    if sp.issparse(Q_conditional):
        Q_inv_ref = xp.linalg.inv(Q_conditional.toarray())
    else:
        Q_inv_ref = xp.linalg.inv(Q_conditional)
    print_msg(
        f"Norm (marginal variances of latent parameters - reference): {xp.linalg.norm(var_latent_params - xp.diag(Q_inv_ref)):.4e}",
    )

    # 9. Compute marginal distributions of the hyperparameters
    marginals_hyperparameters = dalia.marginal_distributions_hp()

    # 10. Plot marginal distributions of hyperparameters
    # fig, axes = plot_marginal_distributions_hp(marginals_hyperparameters)
    # plt.savefig("marginal_distributions_hyperparameters.png")

    fig, axes = plot_marginal_distributions_hp_external(
        marginals_hyperparameters, theta_reference
    )
    plt.savefig("marginal_distributions_hyperparameters.png")

    h2 = marginals_hyperparameters["hyperparameters"]["h2"]
    quantile_pairs = h2["quantiles"]["external"]["pairs"]

    print("Quantile pairs of h2:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
