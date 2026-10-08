# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Tests of the stationary AR(p) submodel."""

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

from dalia.configs import submodels_config
from dalia.submodels import ARSubModel
from dalia.utils.gpu_utils import get_host

BETA_PM1 = {"type": "beta", "alpha": 2.0, "beta": 2.0, "support": [-1.0, 1.0]}
GAMMA = {"type": "gamma", "alpha": 2.0, "beta": 1.0}


def _make_ar(tmp_path, n, order) -> ARSubModel:
    scipy.sparse.save_npz(tmp_path / "a.npz", scipy.sparse.eye(n, format="csc"))
    config = submodels_config.parse_config(
        {
            "type": "ar",
            "input_dir": str(tmp_path),
            "order": order,
            "pacf": [0.0] * order,
            "ph_pacf": [dict(BETA_PM1) for _ in range(order)],
            "tau": 1.0,
            "ph_tau": dict(GAMMA),
        }
    )
    return ARSubModel(config=config)


def _kwargs(pacf, tau):
    return {f"pacf{k}": v for k, v in enumerate(pacf, start=1)} | {"tau": tau}


@pytest.mark.parametrize("tau", [0.5, 8.0])
@pytest.mark.parametrize(
    "pacf",
    [[0.7], [0.6, -0.4], [0.6, -0.4, 0.3], [0.9, -0.8, 0.7, -0.6, 0.5]],
    ids=["ar1", "ar2", "ar3", "ar5"],
)
# covers n < p, n = p, n = p + 1 (boundary blocks overlap) and n > 2p
@pytest.mark.parametrize("n", [1, 2, 4, 7, 20])
def test_Q_is_a_stationary_precision(tmp_path, n, pacf, tau):
    """Validate if the PACF parameters construct a stationary precision matrix.

    A process is weakly stationary when its mean is constant and its
    autocovariance Cov(x_s, x_t) depends on the lag |s - t| only. The covariance
    matrix of n consecutive observations is then a symmetric Toeplitz matrix,
    constant along every diagonal, with the marginal variance on the main one.

    References:
        Brockwell, P. J. and Davis, R. A. (1991). Time Series: Theory and
            Methods, 2nd ed. Springer. (Stationarity and the autocovariance
            function; covariance matrices of stationary sequences.)
        Gray, R. M. (2006). Toeplitz and Circulant Matrices: A Review.
            Foundations and Trends in Communications and Information Theory,
            2(3), 155-239. (Toeplitz structure of stationary covariances.)
    """
    ar = _make_ar(tmp_path, n, order=len(pacf))
    Q = np.asarray(get_host(ar.construct_Q_prior(**_kwargs(pacf, tau)).toarray()))

    # a stationary covariance depends on the lag only (symmetric Toeplitz)
    Sigma = np.linalg.inv(Q)
    np.testing.assert_allclose(Sigma, scipy.linalg.toeplitz(Sigma[0]), rtol=0, atol=1e-9)
    # with the marginal variance 1 / tau at every position, boundaries included
    np.testing.assert_allclose(np.diag(Sigma), 1 / tau, rtol=1e-9)


# --- Invalid hyperparameters, including ones that do not enter the matrix --------
@pytest.mark.parametrize(
    "n, kwargs, match",
    [
        pytest.param(6, {"pacf1": 0.2, "tau": 1.0}, "Partial autocorrelations", id="pacf2 missing"),
        pytest.param(6, {"pacf1": 0.2, "pacf2": 0.1}, "tau", id="tau missing"),
        pytest.param(6, _kwargs([0.2, 0.1], None), "tau", id="tau None"),
        pytest.param(6, _kwargs([np.nan, 0.2], 1.0), "finite", id="pacf nan"),
        pytest.param(6, _kwargs([0.2, np.inf], 1.0), "finite", id="pacf inf"),
        pytest.param(6, _kwargs([1.0, 0.2], 1.0), r"\(-1, 1\)", id="pacf1 = 1"),
        pytest.param(6, _kwargs([0.2, -1.0], 1.0), r"\(-1, 1\)", id="pacf2 = -1"),
        pytest.param(6, _kwargs([1.5, 0.2], 1.0), r"\(-1, 1\)", id="pacf1 > 1"),
        pytest.param(6, _kwargs([0.2, 0.1], 0.0), "positive", id="tau = 0"),
        pytest.param(6, _kwargs([0.2, 0.1], -3.0), "positive", id="tau < 0"),
        pytest.param(6, _kwargs([0.2, 0.1], np.nan), "tau", id="tau nan"),
        pytest.param(6, _kwargs([0.2, 0.1], np.inf), "tau", id="tau inf"),
        # n < p: pacf2 does not enter Q for n = 1, it is validated nonetheless
        pytest.param(1, _kwargs([0.2, 1.0], 1.0), r"\(-1, 1\)", id="unused pacf2 = 1"),
        pytest.param(1, {"pacf1": 0.2, "tau": 1.0}, "Partial autocorrelations", id="unused pacf2 missing"),
    ],
)
def test_construct_Q_prior_rejects_invalid_hyperparameters(tmp_path, n, kwargs, match):
    ar = _make_ar(tmp_path, n, order=2)
    with pytest.raises(ValueError, match=match):
        ar.construct_Q_prior(**kwargs)
