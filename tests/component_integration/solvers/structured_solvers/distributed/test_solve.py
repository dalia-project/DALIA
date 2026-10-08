# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi(min_size=2)
def test_solve_correctness(
    reference_solve,
    allclose_ndarrays,
    create_solver,
    create_pobta,
    create_pobt,
    create_rhs,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks_per_process,
    arrowhead_blocksize,
    num_rhs,
    non_uniform_partition,
):
    """Test Triangular Solve correctness against NumPy/CuPy reference."""
    import mpi4py.MPI as MPI

    # Generate test matrix based on sparsity pattern
    n_diag_blocks = n_diag_blocks_per_process * MPI.COMM_WORLD.Get_size() + (
        1 if non_uniform_partition else 0
    )

    if arrowhead_blocksize > 0:  # bta
        A = create_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    else:  # bt
        A = create_pobt(diagonal_blocksize, n_diag_blocks)

    # Generate rhs
    b = create_rhs(
        n_rhs=num_rhs,
        matrix_size=n_diag_blocks * diagonal_blocksize + arrowhead_blocksize,
    )

    # Compute reference
    x_ref = reference_solve(A, b.copy())

    # Create solver
    solver = create_solver(
        solver_type,
        diagonal_blocksize,
        n_diag_blocks,
        arrowhead_blocksize,
        distributed=True,
    )

    # Run solver factorize
    solver.factorize(A, sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Run solver solve
    x_solver = solver.solve(rhs=b, sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Verify results
    allclose_ndarrays(
        a_reference=x_ref,
        b_toverify=x_solver,
    )
