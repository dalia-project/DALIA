import pytest


@pytest.fixture
def input_dir(tmp_path):
    from .constraints_utils import _create_model_inputs

    return _create_model_inputs(tmp_path)


@pytest.fixture
def build_model():
    from .constraints_utils import _build_model

    return _build_model


@pytest.fixture
def poisson_input_dir(tmp_path):
    from .constraints_utils import _create_poisson_inputs

    return _create_poisson_inputs(tmp_path)


@pytest.fixture
def build_poisson_model():
    from .constraints_utils import _build_poisson_model

    return _build_poisson_model
