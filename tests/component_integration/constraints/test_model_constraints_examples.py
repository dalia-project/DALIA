from pathlib import Path

import numpy as np
import pytest

from dalia.configs import (
    constraints_config,
    likelihood_config,
    models_config,
    submodels_config,
)
from dalia.core.model import Model
from dalia.models import CoregionalModel
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

REPO = Path(__file__).resolve().parents[3]
GST_SMALL = REPO / "examples" / "gst_small"
GST_COREG = REPO / "examples" / "gst_coreg2_small" / "inputs_nv2_ns354_nt12_nb2"
GAUSSIAN_PRIOR = {"type": "gaussian", "mean": 0.0, "precision": 0.5}


def build_gst_small(regression_constraints=(), constraints=()):
    """Regression declared first; Model moves the spatio-temporal submodel to the
    front, so the regression block ends up last (columns 460..465)."""
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
                "constraints": list(regression_constraints),
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
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def coregional_variate(i, constraints=()):
    variate_dir = GST_COREG / f"model_{i}"
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(
            {
                "type": "spatio_temporal",
                "input_dir": str(variate_dir / "inputs_spatio_temporal"),
                "spatial_domain_dimension": 2,
                "r_s": 0.0,
                "r_t": 0.0,
                "sigma_st": 0.0,
                "manifold": "plane",
                "ph_s": dict(GAUSSIAN_PRIOR),
                "ph_t": dict(GAUSSIAN_PRIOR),
                "ph_st": dict(GAUSSIAN_PRIOR),
            }
        )
    )
    regression = RegressionSubModel(
        config=submodels_config.parse_config(
            {
                "type": "regression",
                "input_dir": str(variate_dir / "inputs_regression"),
                "n_fixed_effects": 1,
                "fixed_effects_prior_precision": 0.001,
            }
        )
    )
    return Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(
            {"type": "gaussian", "prec_o": 1.0, "prior_hyperparameters": dict(GAUSSIAN_PRIOR)}
        ),
        constraints=[constraints_config.parse_config(c) for c in constraints],
    )


def coregional_config():
    return models_config.parse_config(
        {
            "type": "coregional",
            "n_models": 2,
            "sigmas": [0.0, 0.0],
            "lambdas": [0.0],
            "ph_sigmas": [dict(GAUSSIAN_PRIOR), dict(GAUSSIAN_PRIOR)],
            "ph_lambdas": [dict(GAUSSIAN_PRIOR)],
        }
    )


def test_submodel_constraint_follows_reordering():
    model = build_gst_small(
        regression_constraints=[{"type": "linear", "C": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]}]
    )

    assert isinstance(model.submodels[0], SpatioTemporalSubModel)
    C = np.asarray(model.C)
    assert C.shape == (1, 466)
    np.testing.assert_array_equal(C[0, :460], np.zeros(460))
    np.testing.assert_array_equal(C[0, 460:], np.arange(1.0, 7.0))


def test_model_constraint_uses_final_latent_order():
    row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    C_model = np.zeros((1, 466))
    C_model[0, 460:] = row

    on_submodel = build_gst_small(regression_constraints=[{"type": "linear", "C": [row]}])
    on_model = build_gst_small(constraints=[{"type": "linear", "C": C_model}])

    np.testing.assert_array_equal(np.asarray(on_model.C), np.asarray(on_submodel.C))


def test_coregional_model_rejects_constraints():
    constrained = coregional_variate(1, constraints=[{"type": "sum_to_zero"}])

    with pytest.raises(NotImplementedError, match="CoregionalModel"):
        CoregionalModel(
            models=[constrained, coregional_variate(2)],
            coregional_model_config=coregional_config(),
        )


def test_coregional_model_rejects_model_level_constraints():
    with pytest.raises(NotImplementedError, match="CoregionalModel"):
        CoregionalModel(
            models=[coregional_variate(1), coregional_variate(2)],
            coregional_model_config=coregional_config(),
            constraints=[constraints_config.parse_config({"type": "sum_to_zero"})],
        )
