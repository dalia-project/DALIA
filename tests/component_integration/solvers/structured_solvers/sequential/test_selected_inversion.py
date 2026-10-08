# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_selected_inversion_correctness(
    reference_inversion,
    allclose_dense_structured,
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

    # Run solver selected inversion
    solver.selected_inversion(sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Get the computed selected inverse in sparse format
    A_selinv_solver = solver._structured_to_spmatrix(
        A,
        sparsity="bta" if arrowhead_blocksize > 0 else "bt",
        symmetrize=True,
    )

    # Reference dense inversion
    A_inv_ref = reference_inversion(A)

    # Assert correctness within sparsity pattern
    allclose_dense_structured(
        A_reference=A_inv_ref,
        B_toverify=(
            A_selinv_solver.toarray()
            if hasattr(A_selinv_solver, "toarray")
            else A_selinv_solver
        ),
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
        assert_upper_triangle=True,
    )
