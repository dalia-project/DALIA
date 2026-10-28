import numpy as np
import pytest

from dalia import xp
from dalia.configs import dalia_config
from dalia.core.dalia import DALIA
from tests import ATOLS, RTOLS

from .constraints_utils import N

# Internal-scale hyperparameters: (phi, tau, prec_o) for Gaussian, (phi, tau) for Poisson
THETA_GAUSSIAN = np.array([0.3, 1.0, 4.0])
THETA_POISSON = np.array([0.3, 1.0])

C_GENERAL = np.random.default_rng(7).normal(size=(2, N + 1))
E_GENERAL = np.array([1.5, -2.0])

# name: (AR1 submodel constraints, model constraints)
CASES = {
    "none": ([], []),
    "sum_to_zero": ([{"type": "sum_to_zero"}], []),
    "general": ([], [{"type": "linear", "C": C_GENERAL, "e": E_GENERAL}]),
}


def make_config(solver):
    return dalia_config.parse_config(
        {
            "solver": {"type": solver},
            "verbosity": 0,
            "inner_iteration_max_iter": 100,
            "eps_inner_iteration": 1e-12,
            "minimize": {"max_iter": 10, "gtol": 1e-3, "maxcor": 3},
        }
    )


def reference_covariance(model, theta_external, x_star):
    """Dense covariance of the (constrained) Gaussian approximation at theta and x_star:
    Q_c^-1, minus Q_c^-1 C^T (C Q_c^-1 C^T)^-1 C Q_c^-1 with constraints."""
    model.theta_external = xp.asarray(theta_external)
    model.construct_Q_prior()
    Q_c = model.construct_Q_conditional(model.a @ xp.asarray(x_star)).toarray()
    S = np.linalg.inv(Q_c)
    if model.C is not None:
        C = np.asarray(model.C)
        V = S @ C.T
        S = S - V @ np.linalg.solve(C @ V, V.T)
    return S


def mode(dalia, model, theta_internal):
    """Evaluate f at theta and return (theta in external scale, latent mode / mean)."""
    dalia._evaluate_f(xp.asarray(theta_internal))
    return np.asarray(model.theta_external).copy(), np.asarray(model.x).copy()


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("case", ["none", "sum_to_zero", "general"])
def test_gaussian_marginal_variances(input_dir, build_model, solver, case):
    ar1_constraints, model_constraints = CASES[case]
    model = build_model(
        input_dir, ar1_constraints=ar1_constraints, constraints=model_constraints
    )
    dalia = DALIA(model=model, config=make_config(solver))
    theta_external, x_star = mode(dalia, model, THETA_GAUSSIAN)

    variances = dalia.get_marginal_variances_latent_parameters(
        xp.asarray(theta_external), xp.asarray(x_star)
    )
    reference = np.diag(reference_covariance(model, theta_external, x_star))

    np.testing.assert_allclose(
        np.asarray(variances), reference, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("case", ["none", "sum_to_zero", "general"])
def test_poisson_marginal_variances(poisson_input_dir, build_poisson_model, solver, case):
    model = build_poisson_model(poisson_input_dir, *CASES[case])
    dalia = DALIA(model=model, config=make_config(solver))
    theta_external, x_star = mode(dalia, model, THETA_POISSON)

    variances = dalia.get_marginal_variances_latent_parameters(
        xp.asarray(theta_external), xp.asarray(x_star)
    )
    reference = np.diag(reference_covariance(model, theta_external, x_star))

    np.testing.assert_allclose(
        np.asarray(variances), reference, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )
