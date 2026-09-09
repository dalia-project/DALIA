# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from tests.structured_solvers_utils import (
    _allclose_dense_structured,
    _create_pobt,
    _create_pobta,
)

pytest.importorskip("sTiles", reason="The sTiles package is required for these tests.")


DIAGONAL_BLOCKSIZE = [
    pytest.param(1, id="diagonal_blocksize=1"),
    pytest.param(3, id="diagonal_blocksize=3"),
    pytest.param(8, id="diagonal_blocksize=8"),
]


@pytest.fixture(params=DIAGONAL_BLOCKSIZE, autouse=True)
def diagonal_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param


ARROWHEAD_BLOCKSIZE = [
    pytest.param(0, id="arrowhead_blocksize=0"),
    pytest.param(2, id="arrowhead_blocksize=2"),
]


@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param


N_DIAG_BLOCKS = [
    pytest.param(1, id="n_diag_blocks=1"),
    pytest.param(4, id="n_diag_blocks=4"),
    pytest.param(11, id="n_diag_blocks=11"),
]


@pytest.fixture(params=N_DIAG_BLOCKS, autouse=True)
def n_diag_blocks(request: pytest.FixtureRequest) -> int:
    return request.param


N_THREADS = [
    pytest.param(1, id="n_threads=1"),
    pytest.param(4, id="n_threads=4"),
]


@pytest.fixture(params=N_THREADS)
def n_threads(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def create_solver():
    """Return a factory producing a fresh STilesSolver, closing it at teardown.

    sTiles allows a single live handle per process, hence the explicit close.
    """
    from dalia.configs.dalia_config import SolverConfig
    from dalia.solvers import STilesSolver

    solvers = []

    def _create(n_threads: int = 1):
        solver = STilesSolver(
            config=SolverConfig(type="stiles", stiles_threads=n_threads)
        )
        solvers.append(solver)
        return solver

    yield _create

    for solver in solvers:
        solver.close()


@pytest.fixture
def create_spd_matrix():
    """Random SPD block-tridiagonal (arrowhead) matrix in sparse `csc` format."""
    from dalia import sp, xp

    def _create(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize):
        if arrowhead_blocksize > 0:
            A = _create_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:
            A = _create_pobt(diagonal_blocksize, n_diag_blocks)
        # `_create_pobt` only adds a scaled identity, which does not guarantee
        # positive definiteness for large blocks: make it diagonally dominant.
        A[xp.arange(A.shape[0]), xp.arange(A.shape[0])] += xp.sum(xp.abs(A), axis=1)
        return sp.sparse.csc_matrix(A)

    return _create


@pytest.fixture
def allclose_dense_structured():
    return _allclose_dense_structured
