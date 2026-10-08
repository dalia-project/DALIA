# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from tests.structured_solvers_utils import (
    _allclose_dense_structured,
    _create_pobt,
    _create_pobta,
    _create_solver,
)

DIAGONAL_BLOCKSIZE = [
    pytest.param(2, id="diagonal_blocksize=2"),
    pytest.param(3, id="diagonal_blocksize=3"),
]


@pytest.fixture(params=DIAGONAL_BLOCKSIZE, autouse=True)
def diagonal_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param


ARROWHEAD_BLOCKSIZE = [
    pytest.param(0, id="arrowhead_blocksize=0"),
    pytest.param(2, id="arrowhead_blocksize=2"),
    pytest.param(3, id="arrowhead_blocksize=3"),
]


@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param


SOLVERS_TYPES = [
    pytest.param("serinv", id="solvers_types=serinv"),
]


@pytest.fixture(params=SOLVERS_TYPES)
def solver_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def create_solver():
    return _create_solver


@pytest.fixture
def create_pobta():
    return _create_pobta


@pytest.fixture
def create_pobt():
    return _create_pobt


@pytest.fixture
def allclose_dense_structured():
    return _allclose_dense_structured
