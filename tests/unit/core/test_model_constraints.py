import numpy as np
import pytest

from .constraints_utils import N


def test_regression_constraint_is_placed_on_its_block(input_dir, build_model):
    model = build_model(
        input_dir, regression_constraints=[{"type": "linear", "C": [[2.0]], "e": [1.0]}]
    )

    expected = np.zeros((1, N + 1))
    expected[0, N] = 2.0
    np.testing.assert_array_equal(np.asarray(model.C), expected)
    np.testing.assert_array_equal(np.asarray(model.e), np.array([1.0]))


def test_submodel_and_model_constraints_are_stacked(input_dir, build_model):
    C_model = np.zeros((1, N + 1))
    C_model[0, 0] = 1.0
    model = build_model(
        input_dir,
        ar1_constraints=[{"type": "sum_to_zero"}],
        constraints=[{"type": "linear", "C": C_model, "e": [2.0]}],
    )

    expected = np.zeros((2, N + 1))
    expected[0, :N] = 1.0
    expected[1, 0] = 1.0
    np.testing.assert_array_equal(np.asarray(model.C), expected)
    np.testing.assert_array_equal(np.asarray(model.e), np.array([0.0, 2.0]))


def test_redundant_constraints_raise(input_dir, build_model):
    row = np.zeros((1, N + 1))
    row[0, :N] = 1.0

    with pytest.raises(ValueError, match="full row rank"):
        build_model(
            input_dir,
            ar1_constraints=[{"type": "sum_to_zero"}],
            constraints=[{"type": "linear", "C": row}],
        )


def test_model_constraint_from_file_matches_inline(input_dir, build_model, tmp_path):
    C = np.random.default_rng(7).normal(size=(2, N + 1))
    e = np.array([1.5, -2.0])
    path = tmp_path / "model_constraints.npz"
    np.savez(path, C=C, e=e)

    from_file = build_model(input_dir, constraints=[{"type": "linear", "input_file": str(path)}])
    inline = build_model(input_dir, constraints=[{"type": "linear", "C": C, "e": e}])

    np.testing.assert_array_equal(np.asarray(from_file.C), np.asarray(inline.C))
    np.testing.assert_array_equal(np.asarray(from_file.e), np.asarray(inline.e))
