# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_logdet_correctness(
    reference_logdet,
    allclose_floats,
    generate_spd_spmatrix,
    create_solver,
    solver_type,
    matrix_size,
    matrix_type,
):
    """Test correctness of the log-determinant computation of the solver."""
    # Generate test case
    A = generate_spd_spmatrix(matrix_type, matrix_size)

    # Solver to compare
    solver = create_solver(solver_type, matrix_size)

    solver.factorize(A)

    logdet_solver = solver.logdet()

    # Reference
    logdet_ref = reference_logdet(A)

    # Compare
    allclose_floats(
        a_reference=logdet_ref,
        b_toverify=logdet_solver,
    )
