# Marginal variances of the linear predictor, diag(A Sigma A^T), computed by
# DALIA from the selected inverse of the serinv solver (read from its lower
# triangle) and compared against a dense inverse on the gst_small example.

from pathlib import Path

import numpy as np

from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import get_host
from tests import ATOLS, RTOLS

GST_SMALL = Path(__file__).resolve().parents[2] / "examples" / "gst_small"


def build_gst_small():
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": str(GST_SMALL / "inputs_spatio_temporal"),
                "spatial_domain_dimension": 2,
                "r_s": 0,
                "r_t": 0,
                "sigma_st": 0,
                "manifold": "sphere",
                "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
                "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
                "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(GST_SMALL / "inputs_regression"),
                "n_fixed_effects": 6,
            }
        )
    )
    return Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(
            {
                "type": "gaussian",
                "prec_o": 4.0,
                "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 2.0},
            }
        ),
    )


def test_marginal_variances_linear_predictor_matches_dense_inverse():
    model = build_gst_small()
    config = dalia_config.parse_config({"solver": {"type": "serinv"}, "verbosity": 0})
    dalia = DALIA(model=model, config=config)
    dalia._evaluate_f(model.theta_internal)
    theta_external, x_star = model.theta_external, model.x.copy()

    variances = dalia.get_marginal_variances_linear_predictor(theta_external, x_star)

    A = get_host(model.a.toarray())
    Sigma = np.linalg.inv(get_host(model.Q_conditional.toarray()))
    reference = np.diag(A @ Sigma @ A.T)
    np.testing.assert_allclose(
        get_host(variances), reference, rtol=RTOLS["relaxed"], atol=ATOLS["relaxed"]
    )
