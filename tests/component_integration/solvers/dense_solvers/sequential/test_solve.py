# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_solve_correctness(
    reference_solve,
    allclose_ndarrays,
    generate_spd_spmatrix,
    create_rhs,
    create_solver,
    solver_type,
    matrix_size,
    matrix_type,
    num_rhs,
):
    """Test correctness of the solve step of the solver."""
    # Generate test case
    A = generate_spd_spmatrix(matrix_type, matrix_size)

    b = create_rhs(n_rhs=num_rhs, matrix_size=matrix_size)

    # Solver to compare
    solver = create_solver(solver_type, matrix_size)

    solver.factorize(A)

    x_solver = solver.solve(b.copy())

    # Reference
    x_ref = reference_solve(A, b)

    # Compare
    allclose_ndarrays(
        a_reference=x_ref,
        b_toverify=x_solver,
    )
