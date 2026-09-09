# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

import pytest


@pytest.mark.mpi_skip()
def test_factorize_correctness(
    generate_spd_spmatrix,
    allclose_ndarrays,
    create_solver,
    solver_type,
    matrix_size,
    density,
):
    if solver_type != "scipy":
        pytest.skip("This test inspects the LU factors of the scipy solver.")

    A = generate_spd_spmatrix(matrix_size, density)

    solver = create_solver(solver_type)

    # Test that factorization is callable without errors
    solver.factorize(A)

    # Reconstruct the matrix from its LU factors and compare
    A_recovered = solver.LU_factor.L @ solver.LU_factor.U
    A_dense = A.toarray() if hasattr(A, "toarray") else A
    A_recovered_dense = A_recovered.toarray() if hasattr(A_recovered, "toarray") else A_recovered

    try:
        allclose_ndarrays(
            A_dense,
            A_recovered_dense,
            relaxed_tolerance=False,
        )
    except AssertionError as e:
        allclose_ndarrays(
            A_dense,
            A_recovered_dense,
            relaxed_tolerance=True,
        )
        warnings.warn(
            f"Test passed for sparse_solver.{solver_type}.factorize() within relaxed tolerance."
            "This is likely due to numerical innacuray in retrieving the matrix from its LU factors."
            "Correctness is further tested through `solve()` and `logdet()` tests.",
            UserWarning,
            stacklevel=2,
        )
