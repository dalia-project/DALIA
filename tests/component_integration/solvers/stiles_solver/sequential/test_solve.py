# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from dalia import xp


@pytest.mark.mpi_skip()
def test_solve_correctness(
    reference_solve,
    allclose_ndarrays,
    create_solver,
    create_spd_matrix,
    create_rhs,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
    num_rhs,
    n_threads,
):
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    n = A.shape[0]
    sparsity = "bta" if arrowhead_blocksize > 0 else "bt"

    b = create_rhs(n_rhs=num_rhs, matrix_size=n)
    if num_rhs == 1:
        b = b.ravel()  # DALIA always passes 1D right-hand sides
    x_ref = reference_solve(A, b.copy()).reshape(b.shape)

    solver = create_solver(n_threads)
    solver.factorize(A, sparsity=sparsity)
    x_solver = solver.solve(rhs=b.copy(), sparsity=sparsity)

    assert x_solver.shape == b.shape
    assert type(x_solver) is type(b)
    allclose_ndarrays(a_reference=x_ref, b_toverify=x_solver, relaxed_tolerance=True)


@pytest.mark.mpi_skip()
def test_solve_many_times_is_stable(
    reference_solve,
    allclose_ndarrays,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    """Repeated threaded solves must all be correct (sparse tile mode is reproducible)."""
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    b = xp.random.rand(A.shape[0])
    x_ref = reference_solve(A, b.copy())

    solver = create_solver(4)
    solver.factorize(A)
    for _ in range(20):
        allclose_ndarrays(x_ref, solver.solve(rhs=b.copy()), relaxed_tolerance=True)
