# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_logdet_correctness(
    reference_logdet,
    allclose_floats,
    create_solver,
    create_pobta,
    create_pobt,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    """Test Cholesky decomposition correctness against NumPy/CuPy reference."""
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

    # Compute logdet using solver
    logdet_solver = solver.logdet(sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Compute reference
    logdet_ref = reference_logdet(A)

    allclose_floats(
        a_reference=logdet_ref,
        b_toverify=logdet_solver,
    )
