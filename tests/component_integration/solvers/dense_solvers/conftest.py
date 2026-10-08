# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

MATRIX_SIZE = [
    pytest.param(1, id="matrix_size=1"),
    pytest.param(2, id="matrix_size=2"),
    pytest.param(10, id="matrix_size=10"),
    pytest.param(100, id="matrix_size=100"),
]


@pytest.fixture(params=MATRIX_SIZE, autouse=True)
def matrix_size(request: pytest.FixtureRequest) -> int:
    return request.param


MATRIX_TYPE = [
    pytest.param("dense", id="matrix_type=dense"),
    pytest.param("sparse", id="matrix_type=sparse"),
]


@pytest.fixture(params=MATRIX_TYPE, autouse=True)
def matrix_type(request: pytest.FixtureRequest) -> int:
    return request.param


SOLVER_TYPES = [
    pytest.param("dense", id="solver_type=dense"),
]


@pytest.fixture(params=SOLVER_TYPES)
def solver_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def create_solver():
    from .utils import _create_solver

    return _create_solver


@pytest.fixture
def generate_spd_spmatrix():
    from .utils import _generate_spd_spmatrix

    return _generate_spd_spmatrix
