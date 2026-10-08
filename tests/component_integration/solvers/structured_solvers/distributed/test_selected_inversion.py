# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi(min_size=2)
def test_selected_inversion_correctness(
    reference_inversion,
    allclose_dense_structured,
    create_solver,
    create_pobta,
    create_pobt,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks_per_process,
    arrowhead_blocksize,
    non_uniform_partition,
):
    """Test Distributed Selected Inversion correctness against NumPy/CuPy dense reference."""
    import mpi4py.MPI as MPI

    # Generate test matrix based on sparsity pattern
    n_diag_blocks = n_diag_blocks_per_process * MPI.COMM_WORLD.Get_size() + (
        1 if non_uniform_partition else 0
    )

    if arrowhead_blocksize > 0:  # bta
        A = create_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    else:  # bt
        A = create_pobt(diagonal_blocksize, n_diag_blocks)

    # Create solver
    solver = create_solver(
        solver_type,
        diagonal_blocksize,
        n_diag_blocks,
        arrowhead_blocksize,
        distributed=True,
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
