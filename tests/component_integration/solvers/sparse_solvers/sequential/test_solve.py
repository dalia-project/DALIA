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
    density,
    num_rhs,
):
    A = generate_spd_spmatrix(matrix_size, density)

    b = create_rhs(
        n_rhs=num_rhs,
        matrix_size=matrix_size,
    )

    solver = create_solver(solver_type)

    # Test that factorization is callable without errors
    solver.factorize(A)

    x_solver = solver.solve(rhs=b.copy())

    x_ref = reference_solve(A, b)

    allclose_ndarrays(
        a_reference=x_ref,
        b_toverify=x_solver,
    )
