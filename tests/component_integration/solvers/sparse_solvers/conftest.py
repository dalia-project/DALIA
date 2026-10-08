# Copyright 2024-2025 DALIA authors. All rights reserved.

import importlib.util

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


DENSITY = [
    pytest.param(0.1, id="density=0.1"),
    pytest.param(0.5, id="density=0.5"),
    pytest.param(1.0, id="density=1.0"),
]


@pytest.fixture(params=DENSITY, autouse=True)
def density(request: pytest.FixtureRequest) -> int:
    return request.param


SOLVERS_TYPES = [
    pytest.param("scipy", id="solver_type=scipy"),
    pytest.param(
        "stiles",
        id="solver_type=stiles",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("sTiles") is None, reason="sTiles not installed"
        ),
    ),
]


@pytest.fixture(params=SOLVERS_TYPES)
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
