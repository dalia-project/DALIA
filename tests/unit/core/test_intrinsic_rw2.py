# Intrinsic RW2 submodel on irregular locations, with its automatic sum-to-zero
# and zero-linear-trend constraints, and an intercept, against the exact intrinsic
# model computed densely without any diagonal (reduced coordinates, normaliser
# (n - 2) / 2 log tau, structure matrix H^T A^{-1} H of Lindgren & Rue 2008).

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sps

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel, RW2SubModel

N = 20
PREC_INTERCEPT = 0.001
RANK_DEFICIENCY = 2
# irregular locations, gaps between 0.3 and 2.1
S = np.cumsum(0.3 + 1.8 * (np.arange(N) * 7 % 11) / 10.0)

THETAS = {
    "gaussian": [np.array([1.0, 4.0]), np.array([0.2, 2.0]), np.array([1.7, 7.0])],
    "poisson": [np.array([1.0]), np.array([0.2]), np.array([1.7])],
}


def build_model(path, likelihood, diagonal):
    rng = np.random.default_rng(2)
    (path / "inputs_rw").mkdir()
    (path / "inputs_regression").mkdir()
    sps.save_npz(path / "inputs_rw" / "a.npz", sps.identity(N, format="csr"))
    np.save(path / "inputs_rw" / "locations.npy", S)
    sps.save_npz(path / "inputs_regression" / "a.npz", sps.csr_matrix(np.ones((N, 1))))
    u = np.cumsum(np.cumsum(rng.normal(size=N) * 0.2))
    u = u - u.mean()
    if likelihood == "gaussian":
        np.save(path / "y.npy", 2.0 + u + rng.normal(size=N) * 0.5)
    else:
        np.save(path / "y.npy", rng.poisson(np.exp(1.0 + u)).astype(float))

    rw = RW2SubModel(
        config=submodels_config.parse_config(
            {
                "type": "rw2",
                "input_dir": str(path / "inputs_rw"),
                "tau": 3.0,
                "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
                "diagonal": diagonal,
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(path / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": PREC_INTERCEPT,
            }
        )
    )
    if likelihood == "gaussian":
        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 4.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 0.00005},
        }
    else:
        likelihood_dict = {"type": "poisson", "input_dir": str(path)}
    return Model(
        submodels=[rw, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )


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


def structure_matrix(s):
    """R = H^T A^{-1} H of Lindgren & Rue (2008), Section 2, built densely."""
    n = len(s)
    d = np.diff(s)
    H = np.zeros((n, n))
    A = np.ones(n)
    for i in range(1, n - 1):
        H[i, i - 1] = 1 / d[i - 1]
        H[i, i] = -(1 / d[i - 1] + 1 / d[i])
        H[i, i + 1] = 1 / d[i]
        A[i] = (d[i - 1] + d[i]) / 2
    return H.T @ np.diag(1 / A) @ H


def reference(model, theta_internal, likelihood):
    """Exact intrinsic RW2, no diagonal: Laplace approximation in reduced
    coordinates x = Z z, Z an orthonormal basis of null(C). Returns
    -log p(y | theta, Cx=0) - log p(theta) up to a theta-independent constant,
    the constrained mode and the marginal sd."""
    model.theta_internal = xp.asarray(theta_internal)
    tau = float(model.theta_external[model.theta_keys.index("tau")])
    a = model.a.toarray()
    y = np.asarray(model.y)
    Z = scipy.linalg.null_space(np.asarray(model.C))

    Q = np.zeros((N + 1, N + 1))
    Q[:N, :N] = tau * structure_matrix(S)
    Q[N, N] = PREC_INTERCEPT

    if likelihood == "gaussian":
        tau_y = float(model.theta_external[model.theta_keys.index("prec_o")])
        H = Z.T @ (Q + tau_y * a.T @ a) @ Z
        x = Z @ np.linalg.solve(H, Z.T @ (tau_y * a.T @ y))
        r = y - a @ x
        log_likelihood = 0.5 * N * np.log(tau_y) - 0.5 * tau_y * r @ r
    else:
        z = np.zeros(Z.shape[1])
        for _ in range(100):
            eta = a @ Z @ z
            H = Z.T @ (Q + a.T @ np.diag(np.exp(eta)) @ a) @ Z
            step = np.linalg.solve(H, Z.T @ (a.T @ (y - np.exp(eta)) - Q @ Z @ z))
            z += step
            if np.linalg.norm(step) < 1e-13:
                break
        x = Z @ z
        eta = a @ x
        H = Z.T @ (Q + a.T @ np.diag(np.exp(eta)) @ a) @ Z
        log_likelihood = y @ eta - np.exp(eta).sum()

    log_prior = 0.5 * (N - RANK_DEFICIENCY) * np.log(tau) - 0.5 * x @ Q @ x
    log_marginal_likelihood = log_likelihood + log_prior - 0.5 * np.linalg.slogdet(H)[1]
    f = -(log_marginal_likelihood + float(model.evaluate_log_prior_hyperparameters()))
    sd = np.sqrt(np.einsum("ij,jk,ik->i", Z, np.linalg.inv(H), Z))
    return f, x, sd


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("likelihood", ["gaussian", "poisson"])
def test_f_matches_intrinsic_reference(tmp_path, solver, likelihood):
    model = build_model(tmp_path, likelihood, diagonal=1e-8)
    dalia = DALIA(model=model, config=make_config(solver))
    thetas = THETAS[likelihood]

    f = [float(dalia._evaluate_f(xp.asarray(theta))) for theta in thetas]
    f_ref = [reference(model, theta, likelihood)[0] for theta in thetas]

    np.testing.assert_allclose(np.diff(f), np.diff(f_ref), rtol=0, atol=1e-5)


@pytest.mark.parametrize("solver", ["dense", "scipy"])
@pytest.mark.parametrize("likelihood", ["gaussian", "poisson"])
def test_mode_and_sd_match_intrinsic_reference(tmp_path, solver, likelihood):
    model = build_model(tmp_path, likelihood, diagonal=1e-8)
    dalia = DALIA(model=model, config=make_config(solver))
    theta = THETAS[likelihood][0]

    dalia._evaluate_f(xp.asarray(theta))
    x = np.asarray(model.x).copy()
    sd = np.sqrt(np.asarray(dalia.get_marginal_variances_latent_parameters()))
    _, x_ref, sd_ref = reference(model, theta, likelihood)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sd, sd_ref, rtol=1e-6)
    assert abs(x[:N].sum()) < 1e-10
    assert abs((S - S.mean()) @ x[:N]) < 1e-9
