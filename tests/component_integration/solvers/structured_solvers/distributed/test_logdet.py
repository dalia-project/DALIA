# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi(min_size=2)
def test_logdet_correctness(
    reference_logdet,
    allclose_floats,
    create_solver,
    create_pobta,
    create_pobt,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks_per_process,
    arrowhead_blocksize,
    non_uniform_partition,
):
    """Test logdet correctness against NumPy/CuPy reference."""
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

    # Compute logdet using solver
    logdet_solver = solver.logdet(sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Compute reference
    logdet_ref = reference_logdet(A)

    allclose_floats(
        a_reference=logdet_ref,
        b_toverify=logdet_solver,
    )
