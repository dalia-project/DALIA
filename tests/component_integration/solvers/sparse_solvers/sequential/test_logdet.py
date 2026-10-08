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
    density,
):
    A = generate_spd_spmatrix(matrix_size, density)

    solver = create_solver(solver_type)

    # Test that factorization is callable without errors
    solver.factorize(A)

    logdet_solver = solver.logdet()

    logdet_ref = reference_logdet(A)

    allclose_floats(
        a_reference=logdet_ref,
        b_toverify=logdet_solver,
    )
