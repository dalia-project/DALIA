# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

NUM_RHS = [
    pytest.param(1, id="num_rhs=1"),
    pytest.param(3, id="num_rhs=3"),
    pytest.param(5, id="num_rhs=5"),
]


@pytest.fixture(params=NUM_RHS)
def num_rhs(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def create_rhs():
    from .utils import _create_rhs

    return _create_rhs


@pytest.fixture
def reference_cholesky():
    from .utils import _reference_cholesky

    return _reference_cholesky


@pytest.fixture
def reference_solve():
    from .utils import _reference_solve

    return _reference_solve


@pytest.fixture
def reference_logdet():
    from .utils import _reference_logdet

    return _reference_logdet


@pytest.fixture
def reference_inversion():
    from .utils import _reference_inversion

    return _reference_inversion


@pytest.fixture
def allclose_ndarrays():
    from .utils import _allclose_ndarrays

    return _allclose_ndarrays


@pytest.fixture
def allclose_floats():
    from .utils import _allclose_floats

    return _allclose_floats
