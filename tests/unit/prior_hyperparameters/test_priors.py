# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
import scipy.stats as st

from dalia import xp
from dalia.configs import priorhyperparameters_config as pc
from dalia.prior_hyperparameters import (
    BetaPriorHyperparameters,
    GammaPriorHyperparameters,
    GaussianPriorHyperparameters,
    HalfCauchyPriorHyperparameters,
    HalfNormalPriorHyperparameters,
    InverseGammaPriorHyperparameters,
    LKJCorrPriorHyperparameters,
)
from dalia.utils import get_host


def _lkj_logpdf(rho, eta):
    # 2x2 LKJ: (rho + 1) / 2 ~ Beta(eta, eta)
    return st.beta.logpdf((rho + 1) / 2, eta, eta) - np.log(2)


# (id, prior, scipy reference log-density in external space, internal integration range)
PRIORS = [
    (
        f"gamma-{a}-{b}",
        GammaPriorHyperparameters(
            pc.GammaPriorHyperparametersConfig(type="gamma", alpha=a, beta=b)
        ),
        lambda x, a=a, b=b: st.gamma.logpdf(x, a=a, scale=1 / b),
        (-40.0, 12.0),
    )
    for a, b in [(1.0, 0.5), (3.0, 1.0), (1.0, 0.1)]
]
PRIORS += [
    (
        f"inverse_gamma-{a}-{b}",
        InverseGammaPriorHyperparameters(
            pc.InverseGammaPriorHyperparametersConfig(
                type="inverse_gamma", alpha=a, beta=b
            )
        ),
        lambda x, a=a, b=b: st.invgamma.logpdf(x, a=a, scale=b),
        (-12.0, 40.0),
    )
    for a, b in [(2.0, 0.5), (3.0, 2.0)]
]
PRIORS += [
    (
        f"half_normal-{p}",
        HalfNormalPriorHyperparameters(
            pc.HalfNormalPriorHyperparametersConfig(type="half_normal", precision=p)
        ),
        lambda x, p=p: st.halfnorm.logpdf(x, scale=1 / np.sqrt(p)),
        (-30.0, 80.0),
    )
    for p in [0.5, 0.001]
]
PRIORS += [
    (
        f"half_cauchy-{s}",
        HalfCauchyPriorHyperparameters(
            pc.HalfCauchyPriorHyperparametersConfig(type="half_cauchy", scale=s)
        ),
        lambda x, s=s: st.halfcauchy.logpdf(x, scale=s),
        (-80.0, 80.0),
    )
    for s in [1.0, 25.0]
]
PRIORS += [
    (
        f"lkj_2d-{e}",
        LKJCorrPriorHyperparameters(
            pc.LKJCorrPriorHyperparametersConfig(type="lkj_2d", eta=e)
        ),
        lambda x, e=e: _lkj_logpdf(x, e),
        (-40.0, 40.0),
    )
    for e in [0.5, 1.0, 2.0, 4.0]
]
PRIORS += [
    (
        f"gaussian-{m}-{p}",
        GaussianPriorHyperparameters(
            pc.GaussianPriorHyperparametersConfig(type="gaussian", mean=m, precision=p)
        ),
        lambda x, m=m, p=p: st.norm.logpdf(x, loc=m, scale=1 / np.sqrt(p)),
        (-40.0, 40.0),
    )
    for m, p in [(0.0, 0.5), (-2.0, 2.0)]
]
PRIORS += [
    (
        f"beta-{a}-{b}",
        BetaPriorHyperparameters(
            pc.BetaPriorHyperparametersConfig(type="beta", alpha=a, beta=b)
        ),
        lambda x, a=a, b=b: st.beta.logpdf(x, a, b),
        (-700.0, 700.0),
    )
    for a, b in [(2.0, 2.0), (5.0, 1.5)]
]
PRIORS += [
    (
        f"beta-{a}-{b}-support-{lo}-{up}",
        BetaPriorHyperparameters(
            pc.BetaPriorHyperparametersConfig(
                type="beta", alpha=a, beta=b, support=(lo, up)
            )
        ),
        lambda x, a=a, b=b, lo=lo, up=up: st.beta.logpdf(
            x, a, b, loc=lo, scale=up - lo
        ),
        (-700.0, 700.0),
    )
    for a, b, lo, up in [(2.0, 2.0, -1.0, 1.0), (3.0, 1.5, 2.0, 5.0)]
]

FD_STEP = 1e-6


@pytest.fixture(params=PRIORS, ids=[p[0] for p in PRIORS])
def prior_case(request):
    return request.param[1:]


@pytest.fixture
def theta_internal():
    return xp.linspace(-3.0, 3.0, 12)


@pytest.mark.mpi_skip()
def test_rescale_round_trip(prior_case, theta_internal):
    """backward and forward are inverse of each other."""
    prior, _, _ = prior_case
    rescale = prior.rescale_hyperparameters_to_internal
    theta_external = rescale(theta_internal, "backward")
    assert xp.allclose(rescale(theta_external, "forward"), theta_internal, atol=1e-10)


@pytest.mark.mpi_skip()
def test_forward_jacobian(prior_case, theta_internal):
    """forward_jacobian equals |d forward / d theta_external|."""
    prior, _, _ = prior_case
    rescale = prior.rescale_hyperparameters_to_internal
    theta_external = rescale(theta_internal, "backward")
    finite_diff = (
        rescale(theta_external + FD_STEP, "forward")
        - rescale(theta_external - FD_STEP, "forward")
    ) / (2 * FD_STEP)
    assert xp.allclose(
        rescale(theta_external, "forward_jacobian"), xp.abs(finite_diff), rtol=1e-5
    )


@pytest.mark.mpi_skip()
def test_backward_log_jacobian(prior_case, theta_internal):
    """backward_log_jacobian equals log|d backward / d theta_internal|."""
    prior, _, _ = prior_case
    rescale = prior.rescale_hyperparameters_to_internal
    finite_diff = (
        rescale(theta_internal + FD_STEP, "backward")
        - rescale(theta_internal - FD_STEP, "backward")
    ) / (2 * FD_STEP)
    assert xp.allclose(
        rescale(theta_internal, "backward_log_jacobian"),
        xp.log(xp.abs(finite_diff)),
        atol=1e-5,
    )


@pytest.mark.mpi_skip()
def test_log_prior_matches_scipy(prior_case, theta_internal):
    """External log prior and prior match the scipy reference density."""
    prior, reference_logpdf, _ = prior_case
    theta_external = prior.rescale_hyperparameters_to_internal(
        theta_internal, "backward"
    )
    reference = reference_logpdf(get_host(theta_external))
    assert np.allclose(
        get_host(prior.evaluate_log_prior(theta_external)), reference, atol=1e-9
    )
    assert np.allclose(
        get_host(prior.evaluate_prior(theta_external)), np.exp(reference), atol=1e-9
    )


@pytest.mark.mpi_skip()
def test_internal_log_prior_is_normalized(prior_case):
    """The density in internal space integrates to one."""
    prior, _, (lower, upper) = prior_case
    grid = xp.linspace(lower, upper, 400001)
    density = xp.exp(prior.evaluate_internal_log_prior(grid))
    density = xp.where(xp.isfinite(density), density, 0.0)
    integral = float(xp.sum((density[1:] + density[:-1]) * xp.diff(grid)) / 2)
    assert abs(integral - 1.0) < 1e-6


@pytest.mark.mpi_skip()
def test_internal_log_prior_scalar_input(prior_case):
    """0-d array input, as passed by Model.evaluate_log_prior_hyperparameters."""
    prior, _, _ = prior_case
    theta = xp.asarray([0.3, 0.1])
    value = prior.evaluate_internal_log_prior(theta[0])
    assert np.isfinite(float(value))
