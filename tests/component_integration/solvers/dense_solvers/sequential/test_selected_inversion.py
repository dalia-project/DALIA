# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

import pytest

from tests import ATOLS, RTOLS


@pytest.mark.mpi_skip()
def test_selected_inversion_correctness(
    reference_inversion,
    allclose_ndarrays,
    generate_spd_spmatrix,
    create_solver,
    solver_type,
    matrix_size,
    matrix_type,
):
    """Test correctness of the selected inversion step of the solver."""
    # Generate test case
    A = generate_spd_spmatrix(matrix_type, matrix_size)

    # Solver to compare
    solver = create_solver(solver_type, matrix_size)

    solver.factorize(A)

    X_solver = solver.selected_inversion()

    # Reference
    X_ref = reference_inversion(A)

    # Compare
    # Tolerance is relaxed due to numerical differences in selected inversion implementation
    # of the dense solver (uses trsm on L and L.t less precise than xp.linalg.inv())
    allclose_ndarrays(
        a_reference=X_ref,
        b_toverify=X_solver,
        relaxed_tolerance=True,
    )

    # Warn that this test only checks callability, not numerical correctness
    warnings.warn(
        f"Test passed but numerical accuracy relaxed due to differences in numerical approaches. "
        f"Relaxed accuracy: rtol<{RTOLS['relaxed']} and atol<{ATOLS['relaxed']}",
        UserWarning,
        stacklevel=2,
    )
