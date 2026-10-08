# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_factorize_correctness(
    reference_cholesky,
    allclose_ndarrays,
    generate_spd_spmatrix,
    create_solver,
    solver_type,
    matrix_size,
    matrix_type,
):
    """Test correctness of the factorization step of the solver."""
    # Generate test case
    A = generate_spd_spmatrix(matrix_type, matrix_size)

    # Solver to compare
    solver = create_solver(solver_type, matrix_size)

    solver.factorize(A)

    # Reference
    L_ref = reference_cholesky(A)

    # Compare
    allclose_ndarrays(
        a_reference=L_ref,
        b_toverify=solver.L,
    )
