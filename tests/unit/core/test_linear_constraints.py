import numpy as np
import pytest

from dalia import xp
from dalia.configs import dalia_config
from dalia.core.dalia import DALIA
from tests import ATOLS, RTOLS

from .constraints_utils import N

# Internal-scale hyperparameters (phi, tau, prec_o)
THETAS = [
    np.array([0.3, 1.0, 4.0]),
    np.array([-0.4, 0.2, 2.0]),
    np.array([1.2, 1.7, 7.0]),
]

SUM_TO_ZERO = {"type": "sum_to_zero"}


def linear(C, e):
    return {"type": "linear", "C": np.asarray(C).tolist(), "e": np.asarray(e).tolist()}


def make_config(solver):
    return dalia_config.parse_config(
        {
            "solver": {"type": solver},
            "verbosity": 0,
            "minimize": {"max_iter": 10, "gtol": 1e-3, "maxcor": 3},
        }
    )


def reference_f(model, theta_internal, C, e):
    """-log p(y | theta, Cx=e) - log p(theta), up to a theta-independent constant.

    Closed form for a Gaussian likelihood: y | theta, Cx=e ~ N(a m, a S a^T + I / tau_y),
    with (m, S) the mean and covariance of the prior conditioned on Cx = e.
    """
    model.theta_internal = xp.asarray(theta_internal)
    Q = model.construct_Q_prior().toarray()
    a = model.a.toarray()
    tau_y = float(model.theta_external[model.theta_keys.index("prec_o")])

    S = np.linalg.inv(Q)
    m = np.zeros(Q.shape[0])
    if C is not None:
        B = S @ C.T
        W = C @ B
        m = B @ np.linalg.solve(W, e)
        S = S - B @ np.linalg.solve(W, B.T)

    cov_y = a @ S @ a.T + np.eye(a.shape[0]) / tau_y
    r = model.y - a @ m
    log_marginal_likelihood = (
        -0.5 * np.linalg.slogdet(cov_y)[1] - 0.5 * r @ np.linalg.solve(cov_y, r)
    )

    return -(log_marginal_likelihood + float(model.evaluate_log_prior_hyperparameters()))


def reference_constrained_mean(model, theta_internal, C, e):
    """Posterior mean Q_c^{-1} b conditioned on Cx = e, computed densely.
    For a Gaussian likelihood Q_c = Q_prior + tau_y a^T a and b = tau_y a^T y."""
    model.theta_internal = xp.asarray(theta_internal)
    Q_prior = model.construct_Q_prior().toarray()
    a = model.a.toarray()
    tau_y = float(model.theta_external[model.theta_keys.index("prec_o")])

    Q_c = Q_prior + tau_y * a.T @ a
    mu = np.linalg.solve(Q_c, tau_y * a.T @ model.y)
    B = np.linalg.solve(Q_c, C.T)
    return mu - B @ np.linalg.solve(C @ B, C @ mu - e)


def general_constraint():
    """Two dense rows over all columns, intercept included, with e != 0."""
    rng = np.random.default_rng(7)
    C = rng.normal(size=(2, N + 1))
    e = np.array([1.5, -2.0])
    return C, e


def f_differences(dalia, model, C, e):
    f = [float(dalia._evaluate_f(xp.asarray(theta))) for theta in THETAS]
    f_ref = [reference_f(model, theta, C, e) for theta in THETAS]
    return np.diff(f), np.diff(f_ref)


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("sum_to_zero", [False, True])
def test_f_matches_reference_zero_e(input_dir, build_model, solver, sum_to_zero):
    model = build_model(input_dir, ar1_constraints=[SUM_TO_ZERO] if sum_to_zero else [])
    dalia = DALIA(model=model, config=make_config(solver))
    C = None if model.C is None else np.asarray(model.C)
    e = None if model.e is None else np.asarray(model.e)

    df, df_ref = f_differences(dalia, model, C, e)

    np.testing.assert_allclose(df, df_ref, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"])


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_f_matches_reference_nonzero_e(input_dir, build_model, solver, scale):
    C, e = general_constraint()
    e = scale * e
    model = build_model(input_dir, constraints=[linear(C, e)])
    dalia = DALIA(model=model, config=make_config(solver))

    df, df_ref = f_differences(dalia, model, C, e)

    np.testing.assert_allclose(df, df_ref, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"])


@pytest.mark.parametrize("solver", ["dense", "scipy"])
def test_constrained_mean_matches_reference_nonzero_e(input_dir, build_model, solver):
    C, e = general_constraint()
    model = build_model(input_dir, constraints=[linear(C, e)])
    dalia = DALIA(model=model, config=make_config(solver))

    dalia._evaluate_f(xp.asarray(THETAS[0]))
    x_ref = reference_constrained_mean(model, THETAS[0], C, e)

    np.testing.assert_allclose(
        np.asarray(model.x), x_ref, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )
    np.testing.assert_allclose(
        C @ np.asarray(model.x), e, rtol=0, atol=ATOLS["relaxed"]
    )
