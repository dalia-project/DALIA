# Helpers for the linear-constraint tests: a Gaussian model with 20 AR1 random
# effects and 1 intercept (the last entry of x).

import numpy as np
import scipy.sparse as sps

from dalia.configs import constraints_config, likelihood_config, submodels_config
from dalia.core.model import Model
from dalia.submodels import AR1SubModel, RegressionSubModel

N = 20

GAUSSIAN_LIKELIHOOD = {
    "type": "gaussian",
    "prec_o": 4.0,
    "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 0.00005},
}


def _create_model_inputs(path):
    rng = np.random.default_rng(1)
    (path / "inputs_ar1").mkdir()
    (path / "inputs_regression").mkdir()
    sps.save_npz(path / "inputs_ar1" / "a.npz", sps.identity(N, format="csr"))
    sps.save_npz(path / "inputs_regression" / "a.npz", sps.csr_matrix(np.ones((N, 1))))
    np.save(path / "y.npy", 2.0 + rng.normal(size=N))
    return path


def _build_model(input_dir, ar1_constraints=(), regression_constraints=(), constraints=()):
    """Constraints are given as dicts; the model-level ones are parsed here, as a
    user would with constraints_config.parse_config."""
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(
            {
                "type": "ar1",
                "input_dir": str(input_dir / "inputs_ar1"),
                "phi": 0.5,
                "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
                "tau": 3,
                "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
                "constraints": list(ar1_constraints),
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(input_dir / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
                "constraints": list(regression_constraints),
            }
        )
    )
    return Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(dict(GAUSSIAN_LIKELIHOOD)),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def _create_poisson_inputs(path):
    """Counts y ~ Poisson(exp(1 + u)) with a smooth u."""
    rng = np.random.default_rng(2)
    (path / "inputs_ar1").mkdir()
    (path / "inputs_regression").mkdir()
    sps.save_npz(path / "inputs_ar1" / "a.npz", sps.identity(N, format="csr"))
    sps.save_npz(path / "inputs_regression" / "a.npz", sps.csr_matrix(np.ones((N, 1))))
    u = np.convolve(rng.normal(size=N + 4), np.ones(5) / 5, mode="valid")
    np.save(path / "y.npy", rng.poisson(np.exp(1.0 + u)).astype(float))
    return path


def _build_poisson_model(input_dir, ar1_constraints=(), constraints=()):
    """AR1 + intercept with a Poisson likelihood; constraints given as dicts."""
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(
            {
                "type": "ar1",
                "input_dir": str(input_dir / "inputs_ar1"),
                "phi": 0.5,
                "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
                "tau": 3,
                "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
                "constraints": list(ar1_constraints),
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(input_dir / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    return Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(
            {"type": "poisson", "input_dir": str(input_dir)}
        ),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )
