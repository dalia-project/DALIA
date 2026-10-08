# Copyright 2024-2025 DALIA authors. All rights reserved.

import copy

import pytest


@pytest.mark.mpi(min_size=2)
def test_spmatrix_mapping(
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
    """Test the mapping functions from structured to spmatrix and back."""
    import mpi4py.MPI as MPI

    # Generate test matrix based on sparsity pattern
    n_diag_blocks = n_diag_blocks_per_process * MPI.COMM_WORLD.Get_size() + (
        1 if non_uniform_partition else 0
    )

    if arrowhead_blocksize > 0:  # bta
        A_initial = create_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    else:  # bt
        A_initial = create_pobt(diagonal_blocksize, n_diag_blocks)

    # Make deep copies of A_initial to keep as references
    A_reference = copy.deepcopy(A_initial)
    A_pattern = copy.deepcopy(A_initial)

    # Create solver
    solver = create_solver(
        solver_type,
        diagonal_blocksize,
        n_diag_blocks,
        arrowhead_blocksize,
        distributed=True,
    )

    # Run mapping functions
    for _ in range(2):
        # Run twice to test for potential JIT caching issues
        solver._spmatrix_to_structured(
            A_initial,
            sparsity="bta" if arrowhead_blocksize > 0 else "bt",
        )

        A_solver = solver._structured_to_spmatrix(
            A_pattern,
            sparsity="bta" if arrowhead_blocksize > 0 else "bt",
            symmetrize=True,
        )

    # Verify that the initial matrix is identical to the mapped one
    allclose_dense_structured(
        A_reference=(
            A_reference.toarray() if hasattr(A_reference, "toarray") else A_reference
        ),
        B_toverify=A_solver.toarray() if hasattr(A_solver, "toarray") else A_solver,
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
    )

    # Verify that the 2 matrices given to the mapping functions are untouched
    allclose_dense_structured(
        A_reference=(
            A_reference.toarray() if hasattr(A_reference, "toarray") else A_reference
        ),
        B_toverify=A_initial.toarray() if hasattr(A_initial, "toarray") else A_initial,
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
    )

    allclose_dense_structured(
        A_reference=(
            A_reference.toarray() if hasattr(A_reference, "toarray") else A_reference
        ),
        B_toverify=A_pattern.toarray() if hasattr(A_pattern, "toarray") else A_pattern,
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
    )
