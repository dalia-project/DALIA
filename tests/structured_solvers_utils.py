# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np

from dalia import ArrayLike, backend_flags, xp
from tests import ATOLS, RANDOM_SEED, RTOLS

np.random.seed(RANDOM_SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(RANDOM_SEED))


def _create_solver(
    solver_type: str,
    diagonal_blocksize: int,
    n_diag_blocks: int,
    arrowhead_blocksize: int = 0,
    distributed: bool = False,
):
    from dalia.configs.dalia_config import SolverConfig

    config = SolverConfig(type=solver_type)

    if solver_type == "serinv":
        if not distributed:
            from dalia.solvers import SerinvSolver

            return SerinvSolver(
                config=config,
                diagonal_blocksize=diagonal_blocksize,
                n_diag_blocks=n_diag_blocks,
                arrowhead_blocksize=arrowhead_blocksize,
            )
        else:
            import mpi4py.MPI as MPI

            from dalia.solvers import DistSerinvSolver

            return DistSerinvSolver(
                config=config,
                diagonal_blocksize=diagonal_blocksize,
                arrowhead_blocksize=arrowhead_blocksize,
                n_diag_blocks=n_diag_blocks,
                comm=MPI.COMM_WORLD,
                nccl_comm=None,
            )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def _create_pobta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal arrowhead matrix.

    Parameters
    ----------
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, block tridiagonal arrowhead matrix.
    """

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=xp.float64,
    )

    # Fill the arrowhead blocks
    A[-arrowhead_blocksize:, :-arrowhead_blocksize] = xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    A[-arrowhead_blocksize:, -arrowhead_blocksize:] = xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        # Fill the off-diagonal blocks
        if i < n_diag_blocks - 1:
            A[
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant (hence, if symmetric, SPD)
    for i in range(A.shape[0]):
        A[i, i] += 1 + xp.sum(A[i, :])

    # Make symmetric
    A = A + A.T

    return A


def _create_pobt(
    diagonal_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal matrix.

    Parameters
    ----------
    diagonal_blocksize : int
        Size of the diagonal blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, block tridiagonal matrix.
    """

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks,
            diagonal_blocksize * n_diag_blocks,
        ),
        dtype=xp.float64,
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        # Fill the off-diagonal blocks
        if i < n_diag_blocks - 1:
            A[
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make symmetric
    A = A + A.T

    # Make positive definite by adding scaled identity
    A = A + (xp.max(xp.abs(A)) + 1.0) * xp.eye(A.shape[0])

    return A


def _allclose_dense_structured(
    A_reference: ArrayLike,
    B_toverify: ArrayLike,
    diagonal_blocksize: int,
    n_diag_blocks: int,
    arrowhead_blocksize: int = 0,
    assert_upper_triangle: bool = False,
):
    """Check block-wise correctness of two structured matrices in dense storage format.

    Parameters
    ----------
    A_reference : ArrayLike
        First structured matrix to compare.
    B_toverify : ArrayLike
        Second structured matrix to compare.

    diagonal_blocksize : int
        Size of the diagonal blocks.
    n_diag_blocks : int
        Number of diagonal blocks.
    arrowhead_blocksize : int, optional
        Size of the arrowhead blocks, by default 0.
    assert_upper_triangle : bool, optional
        Whether to assert the upper triangle blocks as well, by default False.

    Raises
    ------
    AssertionError
        If any of the corresponding blocks are not close enough.
    """
    if arrowhead_blocksize > 0:
        # Lower arrow blocks
        assert xp.allclose(
            A_reference[-arrowhead_blocksize:, :-arrowhead_blocksize],
            B_toverify[-arrowhead_blocksize:, :-arrowhead_blocksize],
            rtol=RTOLS["strict"],
            atol=ATOLS["strict"],
        )
        if assert_upper_triangle:
            # Upper arrow blocks
            assert xp.allclose(
                A_reference[:-arrowhead_blocksize, -arrowhead_blocksize:],
                B_toverify[:-arrowhead_blocksize, -arrowhead_blocksize:],
                rtol=RTOLS["strict"],
                atol=ATOLS["strict"],
            )

        # Tip of the arrowhead
        assert xp.allclose(
            A_reference[-arrowhead_blocksize:, -arrowhead_blocksize:],
            B_toverify[-arrowhead_blocksize:, -arrowhead_blocksize:],
            rtol=RTOLS["strict"],
            atol=ATOLS["strict"],
        )

    # Check the diagonal blocks
    for i in range(n_diag_blocks):
        assert xp.allclose(
            A_reference[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            B_toverify[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            rtol=RTOLS["strict"],
            atol=ATOLS["strict"],
        )

        # Check the off-diagonal (lower) blocks
        if i < n_diag_blocks - 1:
            assert xp.allclose(
                A_reference[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ],
                B_toverify[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ],
                rtol=RTOLS["strict"],
                atol=ATOLS["strict"],
            )

            if assert_upper_triangle:
                # Check the off-diagonal (upper) blocks
                assert xp.allclose(
                    A_reference[
                        i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                        (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    ],
                    B_toverify[
                        i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                        (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    ],
                    rtol=RTOLS["strict"],
                    atol=ATOLS["strict"],
                )
