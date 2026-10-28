import numpy as np
import pytest
import scipy.linalg

from dalia import xp
from dalia.configs import dalia_config
from dalia.core.dalia import DALIA
from tests import ATOLS, RTOLS

from .constraints_utils import N

# Internal-scale hyperparameters (phi, tau)
THETAS = [
    np.array([0.3, 1.0]),
    np.array([-0.4, 0.2]),
    np.array([1.2, 1.7]),
]


def make_config(solver):
    return dalia_config.parse_config(
        {
            "solver": {"type": solver},
            "verbosity": 0,
            "inner_iteration_max_iter": 100,
            "eps_inner_iteration": 1e-12,
            "minimize": {"max_iter": 10, "gtol": 1e-3, "maxcor": 2},
        }
    )


def reference(model, theta_internal, C, e):
    """Dense Laplace approximation in reduced coordinates x = x0 + N z, with N an
    orthonormal basis of null(C): -log p(y | theta, Cx=e) - log p(theta) up to a
    theta-independent constant, and the constrained mode. Poisson likelihood."""
    model.theta_internal = xp.asarray(theta_internal)
    Q = model.construct_Q_prior().toarray()
    a = model.a.toarray()
    y = np.asarray(model.y)
    n = Q.shape[0]

    if C is None:
        x0, null = np.zeros(n), np.eye(n)
    else:
        x0 = C.T @ np.linalg.solve(C @ C.T, e)
        null = scipy.linalg.null_space(C)

    # Constrained prior in z: N(m, P^-1)
    P = null.T @ Q @ null
    m = -np.linalg.solve(P, null.T @ Q @ x0)

    z = np.zeros(null.shape[1])
    for _ in range(100):
        eta = a @ (x0 + null @ z)
        H = P + null.T @ a.T @ np.diag(np.exp(eta)) @ a @ null
        g = null.T @ a.T @ (y - np.exp(eta)) - P @ (z - m)
        step = np.linalg.solve(H, g)
        z += step
        if np.linalg.norm(step) < 1e-13:
            break

    eta = a @ (x0 + null @ z)
    H = P + null.T @ a.T @ np.diag(np.exp(eta)) @ a @ null
    log_likelihood = y @ eta - np.exp(eta).sum()
    log_prior = 0.5 * np.linalg.slogdet(P)[1] - 0.5 * (z - m) @ P @ (z - m)
    log_marginal_likelihood = log_likelihood + log_prior - 0.5 * np.linalg.slogdet(H)[1]

    f = -(log_marginal_likelihood + float(model.evaluate_log_prior_hyperparameters()))
    return f, x0 + null @ z


def general_constraint():
    """Two dense rows over all columns, intercept included, with e != 0."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(2, N + 1)), np.array([1.5, -2.0])


C_GENERAL, E_GENERAL = general_constraint()

# name: (AR1 submodel constraints, model constraints)
CASES = {
    "none": ([], []),
    "sum_to_zero": ([{"type": "sum_to_zero"}], []),
    "general": ([], [{"type": "linear", "C": C_GENERAL, "e": E_GENERAL}]),
}


def constraint_matrix(model):
    if model.C is None:
        return None, None
    return np.asarray(model.C), np.asarray(model.e)


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("case", ["none", "sum_to_zero", "general"])
def test_mode_matches_reference(poisson_input_dir, build_poisson_model, solver, case):
    model = build_poisson_model(poisson_input_dir, *CASES[case])
    dalia = DALIA(model=model, config=make_config(solver))
    C, e = constraint_matrix(model)

    dalia._evaluate_f(xp.asarray(THETAS[0]))
    _, x_ref = reference(model, THETAS[0], C, e)

    np.testing.assert_allclose(
        np.asarray(model.x), x_ref, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )


@pytest.mark.parametrize("solver", ["dense", "scipy"])
def test_infeasible_warm_start_reaches_constrained_mode(
    poisson_input_dir, build_poisson_model, solver
):
    model = build_poisson_model(poisson_input_dir, *CASES["general"])
    dalia = DALIA(model=model, config=make_config(solver))
    model.x[:] = xp.asarray(np.random.default_rng(5).normal(size=N + 1))

    dalia._evaluate_f(xp.asarray(THETAS[0]))
    _, x_ref = reference(model, THETAS[0], C_GENERAL, E_GENERAL)

    np.testing.assert_allclose(
        np.asarray(model.x), x_ref, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("case", ["none", "sum_to_zero", "general"])
def test_f_matches_reference(poisson_input_dir, build_poisson_model, solver, case):
    model = build_poisson_model(poisson_input_dir, *CASES[case])
    dalia = DALIA(model=model, config=make_config(solver))
    C, e = constraint_matrix(model)

    f = [float(dalia._evaluate_f(xp.asarray(theta))) for theta in THETAS]
    f_ref = [reference(model, theta, C, e)[0] for theta in THETAS]

    np.testing.assert_allclose(
        np.diff(f), np.diff(f_ref), rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )
