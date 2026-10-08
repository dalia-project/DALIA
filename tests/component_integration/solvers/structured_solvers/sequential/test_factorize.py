# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_factorize_correctness(
    allclose_dense_structured,
    reference_cholesky,
    create_solver,
    create_pobta,
    create_pobt,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    """Test Cholesky decomposition correctness against NumPy/CuPy dense reference."""
    # Generate test matrix based on sparsity pattern
    if arrowhead_blocksize > 0:  # bta
        A = create_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    else:  # bt
        A = create_pobt(diagonal_blocksize, n_diag_blocks)

    # Create solver
    solver = create_solver(
        solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
    )

    # Run solver factorize
    solver.factorize(A, sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Compute reference
    L_ref = reference_cholesky(A)

    L_solver = solver._structured_to_spmatrix(
        A,
        sparsity="bta" if arrowhead_blocksize > 0 else "bt",
        symmetrize=False,
    )

    L_solver_dense = L_solver.toarray() if hasattr(L_solver, "toarray") else L_solver

    allclose_dense_structured(
        A_reference=L_ref,
        B_toverify=L_solver_dense,
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
    )
