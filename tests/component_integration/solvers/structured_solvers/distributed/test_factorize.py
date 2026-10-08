# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

import pytest


@pytest.mark.mpi(min_size=2)
def test_factorize_correctness(
    create_solver,
    create_pobta,
    create_pobt,
    solver_type,
    diagonal_blocksize,
    n_diag_blocks_per_process,
    arrowhead_blocksize,
    non_uniform_partition,
):
    """Test Distributed Cholesky decomposition correctness against NumPy/CuPy dense reference."""
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
        solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
    )

    # Run solver factorize
    solver.factorize(A, sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    # Warn that this test only checks callability, not numerical correctness
    warnings.warn(
        f"Test passed but only verified `structured_solvers.distributed.{solver_type}.factorize()` is callable. "
        "Cannot verify against reference Cholesky as distributed Cholesky factorization is not equal to sequential one."
        "Correctness is still tested through `solve()`, `logdet()`, and `selected_inversion()` tests.",
        UserWarning,
        stacklevel=2,
    )
