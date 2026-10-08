# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np

from dalia import backend_flags, xp, sp
from tests import ATOLS, RANDOM_SEED, RTOLS

np.random.seed(RANDOM_SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(RANDOM_SEED))


def _to_ndarray(A):
    """Convert input to ndarray.

    Parameters
    ----------
    A : ArrayLike
        Input array.

    Returns
    -------
    ndarray
        Converted ndarray.
    """
    A_dense = A.toarray() if hasattr(A, "toarray") else A
    A_dense = xp.asarray(A_dense)
    return A_dense


def _create_rhs(n_rhs: int, matrix_size: int):
    """Returns a random right-hand side.

    Parameters
    ----------
    n_rhs : int
        Number of right-hand sides.
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    B : ArrayLike
        Random right-hand side.
    """

    B = xp.random.rand(matrix_size, n_rhs)

    return B


def _reference_cholesky(A):
    """Compute reference Cholesky decomposition using NumPy/CuPy.

    Parameters
    ----------
    A : ArrayLike
        Input matrix to decompose.

    Returns
    -------
    L : ndarray
        Lower triangular Cholesky factor.
    """
    return xp.linalg.cholesky(_to_ndarray(A))


def _reference_solve(A, rhs):
    """Solve linear system using NumPy/CuPy.

    The reference solution is computed using:
        1. Cholesky decomposition A = L L^T
        2. Solve for y: L y = rhs
        3. Solve for x: L^T x = y
    
    Parameters
    ----------
    A : ArrayLike
        System matrix.
    rhs : ArrayLike
        Right-hand side.

    Returns
    -------
    x : ndarray
        Solution vector.
    """
    L = xp.linalg.cholesky(_to_ndarray(A))
    y = sp.linalg.solve_triangular(L, _to_ndarray(rhs), lower=True)
    x = sp.linalg.solve_triangular(L, y, trans="T", lower=True)
    return x


def _reference_logdet(A):
    """Compute log determinant using NumPy/CuPy Cholesky.

    The log determinant is computed using:
        1. Cholesky decomposition A = L L^T
        2. logdet(A) = 2 * sum(log(diag(L)))

    Parameters
    ----------
    A : ArrayLike
        Input matrix.

    Returns
    -------
    logdet : float
        Log determinant of the matrix.
    """
    L = xp.linalg.cholesky(_to_ndarray(A))
    return 2.0 * xp.sum(xp.log(xp.diag(L)))


def _reference_inversion(A):
    """Compute matrix inverse using NumPy/CuPy.

    The reference inverse is computed using:
        1. Cholesky decomposition A = L L^T
        2. Solve for L_inv: L L_inv = I
        3. Compute A_inv = L_inv^T L_inv

    Parameters
    ----------
    A : ArrayLike
        Input matrix.

    Returns
    -------
    A_inv : ndarray
        Inverse matrix.
    """
    L = xp.linalg.cholesky(_to_ndarray(A))
    L_inv = sp.linalg.solve_triangular(
        L, xp.eye(L.shape[0]), lower=True, overwrite_b=False
    )
    A_inv = L_inv.T @ L_inv
    return A_inv


def _allclose_ndarrays(
    a_reference: np.ndarray,
    b_toverify: np.ndarray,
    relaxed_tolerance: bool = False,
):
    """Check correctness of two ndarrays.

    Parameters
    ----------
    A_reference : ndarray
        Reference vector.
    B_toverify : ndarray
        Vector to verify.
    relaxed_tolerance : bool, optional
        Whether to use relaxed tolerance for comparison, by default False.

    Raises
    ------
    AssertionError
        If the vectors are not close enough.
    """

    assert xp.allclose(
        a_reference,
        b_toverify,
        rtol=RTOLS["relaxed"] if relaxed_tolerance else RTOLS["strict"],
        atol=ATOLS["relaxed"] if relaxed_tolerance else ATOLS["strict"],
    )


def _allclose_floats(
    a_reference: float,
    b_toverify: float,
    relaxed_tolerance: bool = False,
):
    """Check correctness of two floats.

    Parameters
    ----------
    a_reference : float
        Reference float.
    b_toverify : float
        Float to verify.
    relaxed_tolerance : bool, optional
        Whether to use relaxed tolerance for comparison, by default False.

    Raises
    ------
    AssertionError
        If the floats are not close enough.
    """
    assert xp.isclose(
        a_reference,
        b_toverify,
        rtol=RTOLS["relaxed"] if relaxed_tolerance else RTOLS["strict"],
        atol=ATOLS["relaxed"] if relaxed_tolerance else ATOLS["strict"],
    )
