import numpy as np
import pytest
import scipy.sparse as sps
from pydantic import ValidationError

from dalia.configs import submodels_config
from dalia.submodels import RW2SubModel

N = 12
# irregular locations: the gaps are 1, 0.5, 2, 0.2, 3, 1, 1, 0.3, 4, 1.5, 0.7
S = np.cumsum([0.0, 1.0, 0.5, 2.0, 0.2, 3.0, 1.0, 1.0, 0.3, 4.0, 1.5, 0.7])


def galerkin_structure_matrix(s):
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


def build(tmp_path, locations=None, **extra):
    sps.save_npz(tmp_path / "a.npz", sps.identity(N, format="csr"))
    if locations is not None:
        np.save(tmp_path / "locations.npy", np.asarray(locations, dtype=float))
    config = {
        "type": "rw2",
        "input_dir": str(tmp_path),
        "tau": 2.0,
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        **extra,
    }
    return RW2SubModel(config=submodels_config.parse_config(config))


def test_Q_prior_is_tau_R_plus_diagonal(tmp_path):
    """Without locations the grid is regular and R = D2^T D2."""
    submodel = build(tmp_path, diagonal=1e-3)
    D2 = np.diff(np.eye(N), n=2, axis=0)

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * D2.T @ D2 + 1e-3 * np.eye(N), rtol=1e-14, atol=1e-14)


def test_Q_prior_on_irregular_locations_is_the_galerkin_precision(tmp_path):
    """R = H^T A^{-1} H, rank n - 2 with the constant and s in its null space."""
    submodel = build(tmp_path, locations=S, diagonal=1e-3)
    R = galerkin_structure_matrix(S)

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * R + 1e-3 * np.eye(N), rtol=1e-13, atol=1e-13)
    assert np.linalg.matrix_rank(R) == N - 2
    np.testing.assert_allclose(R @ np.ones(N), 0.0, atol=1e-12)
    np.testing.assert_allclose(R @ S, 0.0, atol=1e-12)


@pytest.mark.parametrize("locations", [S[:-1], np.r_[S[:5], S[4], S[6:]]])
def test_locations_must_be_one_per_parameter_and_increasing(tmp_path, locations):
    with pytest.raises(ValueError):
        build(tmp_path, locations=locations)


def test_non_positive_diagonal_is_rejected(tmp_path):
    """Without the diagonal the precision is singular and cannot be factorized."""
    with pytest.raises(ValidationError):
        build(tmp_path, diagonal=0.0)


def test_given_constraints_come_after_the_identifiability_ones(tmp_path):
    """Identifiability rows first, sum(x) = 0 and sum((s - mean s) x) = 0 with
    s the locations, then the constraints of the config."""
    row = np.zeros((1, N))
    row[0, 0] = 1.0
    constraints = [{"type": "linear", "C": row.tolist(), "e": [0.5]}]

    C, e = build(tmp_path, locations=S, constraints=constraints).construct_constraints()

    np.testing.assert_allclose(
        np.asarray(C), np.vstack([np.ones((1, N)), S - S.mean(), row])
    )
    np.testing.assert_array_equal(np.asarray(e), np.array([0.0, 0.0, 0.5]))
