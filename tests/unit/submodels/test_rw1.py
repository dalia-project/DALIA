import numpy as np
import pytest
import scipy.sparse as sps
from pydantic import ValidationError

from dalia.configs import submodels_config
from dalia.submodels import AR1SubModel, AR2SubModel, RegressionSubModel, RW1SubModel

N = 12
# irregular locations: the gaps are 1, 0.5, 2, 0.2, 3, 1, 1, 0.3, 4, 1.5, 0.7
S = np.cumsum([0.0, 1.0, 0.5, 2.0, 0.2, 3.0, 1.0, 1.0, 0.3, 4.0, 1.5, 0.7])


def build(tmp_path, n=N, locations=None, **extra):
    sps.save_npz(tmp_path / "a.npz", sps.identity(n, format="csr"))
    if locations is not None:
        np.save(tmp_path / "locations.npy", np.asarray(locations, dtype=float))
    config = {
        "type": "rw1",
        "input_dir": str(tmp_path),
        "tau": 2.0,
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        **extra,
    }
    return RW1SubModel(config=submodels_config.parse_config(config))


def test_Q_prior_is_tau_R_plus_diagonal(tmp_path):
    """Without locations the grid is regular and R = D^T D."""
    submodel = build(tmp_path, diagonal=1e-3)
    D = np.diff(np.eye(N), axis=0)

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * D.T @ D + 1e-3 * np.eye(N), rtol=1e-14, atol=1e-14)


def test_Q_prior_on_irregular_locations_is_the_wiener_process_precision(tmp_path):
    """x_{i+1} - x_i ~ N(0, d_i / tau) with d_i = s_{i+1} - s_i, so R = D^T diag(1 / d) D
    (Rue & Held 2005, Chapter 3; R-INLA rw1 with values = s): rank n - 1, constant
    in the null space."""
    submodel = build(tmp_path, locations=S, diagonal=1e-3)
    D = np.diff(np.eye(N), axis=0)
    R = D.T @ np.diag(1 / np.diff(S)) @ D

    Q = submodel.construct_Q_prior(tau=2.5).toarray()

    np.testing.assert_allclose(Q, 2.5 * R + 1e-3 * np.eye(N), rtol=1e-13, atol=1e-13)
    assert np.linalg.matrix_rank(R) == N - 1
    np.testing.assert_allclose(R @ np.ones(N), 0.0, atol=1e-12)


@pytest.mark.parametrize("locations", [S[:-1], np.r_[S[:5], S[4], S[6:]]])
def test_locations_must_be_one_per_parameter_and_increasing(tmp_path, locations):
    with pytest.raises(ValueError):
        build(tmp_path, locations=locations)


@pytest.mark.parametrize("diagonal", [0.0, -1e-6])
def test_non_positive_diagonal_is_rejected(tmp_path, diagonal):
    """Without the diagonal the precision is singular and cannot be factorized."""
    with pytest.raises(ValidationError):
        build(tmp_path, diagonal=diagonal)


def test_given_constraints_come_after_the_sum_to_zero(tmp_path):
    row = np.zeros((1, N))
    row[0, 0] = 1.0
    constraints = [{"type": "linear", "C": row.tolist(), "e": [0.5]}]

    C, e = build(tmp_path, constraints=constraints).construct_constraints()

    np.testing.assert_array_equal(np.asarray(C), np.vstack([np.ones((1, N)), row]))
    np.testing.assert_array_equal(np.asarray(e), np.array([0.0, 0.5]))


def test_only_intrinsic_submodels_are_wrapped():
    """The diagonal is added by SubModel, through a wrapper around the
    construct_Q_prior of the submodels with intrinsic = True."""
    assert RW1SubModel.intrinsic
    assert RW1SubModel.construct_Q_prior.__name__ == "wrapper"
    for submodel_class in (AR1SubModel, AR2SubModel, RegressionSubModel):
        assert not submodel_class.intrinsic
        assert submodel_class.construct_Q_prior.__name__ == "construct_Q_prior"


def test_submodels_have_no_identifiability_constraints_by_default(tmp_path):
    for submodel_class in (AR1SubModel, AR2SubModel, RegressionSubModel):
        assert submodel_class._construct_identifiability_constraints(None) == []

    assert len(build(tmp_path)._construct_identifiability_constraints()) == 1
