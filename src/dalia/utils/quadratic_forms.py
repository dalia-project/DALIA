# Copyright 2024-2025 DALIA authors. All rights reserved.

import math

from numba import njit, prange

from dalia import NDArray, sp, xp

# Quadratic forms a_i^T Sigma a_i of the rows of a sparse A with a sparse Sigma.
# The row kernel is written once and compiled for the backend in use: the CPU
# (numpy) or, as a device function, the GPU (cupy).
if xp.__name__ == "cupy":
    from numba import cuda

    jit = cuda.jit(device=True)
else:
    jit = njit


@jit
def _row_quadratic_form(i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values):
    """q_i = sum_{j,k} A_ij Sigma_jk A_ik over the nonzeros of row i of A.

    Sigma is symmetric, so only k >= j is visited (upper triangle of Sigma) and
    the off-diagonal terms count twice. The columns of A and of Sigma are sorted:
    Sigma_jk is reached by advancing in row j of Sigma from the last hit. Returns
    nan if Sigma_jk is not stored.
    """
    variance = 0.0
    for p in range(a_ptr[i], a_ptr[i + 1]):
        j = a_cols[p]
        lo, hi = s_ptr[j], s_ptr[j + 1]
        for q in range(p, a_ptr[i + 1]):
            k = a_cols[q]
            # advance in row j of Sigma up to column k
            while lo < hi and s_cols[lo] < k:
                lo += 1
            if lo == hi or s_cols[lo] != k:
                return math.nan
            # Sigma is symmetric: diagonal terms once, off-diagonal terms twice
            factor = 1.0 if p == q else 2.0
            variance += factor * a_values[p] * s_values[lo] * a_values[q]
    return variance


if xp.__name__ == "cupy":
    @cuda.jit
    def _row_quadratic_forms_kernel(a_ptr, a_cols, a_values, s_ptr, s_cols, s_values, forms):
        i = cuda.grid(1)
        if i < forms.size:
            forms[i] = _row_quadratic_form(
                i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values
            )
else:
    @njit(parallel=True)
    def _row_quadratic_forms_kernel(a_ptr, a_cols, a_values, s_ptr, s_cols, s_values, forms):
        for i in prange(forms.size):
            forms[i] = _row_quadratic_form(
                i, a_ptr, a_cols, a_values, s_ptr, s_cols, s_values
            )


def row_quadratic_forms(
    a: sp.sparse.spmatrix, sigma: sp.sparse.spmatrix
) -> NDArray:
    """Quadratic forms a_i^T Sigma a_i of the rows a_i of A, i.e. diag(A Sigma A^T).

    Computed from selected entries of Sigma, without forming A Sigma.

    Parameters
    ----------
    a : sp.sparse.spmatrix
        Matrix A of shape (m, n).
    sigma : sp.sparse.spmatrix
        Upper triangle of the symmetric matrix Sigma of shape (n, n). It must
        store Sigma_jk for every j <= k such that A_ij and A_ik are nonzeros of
        the same row i.

    Returns
    -------
    forms : NDArray
        Quadratic forms, one per row of A, of shape (m,).

    Notes
    -----
    .. math:: q_i = (A Sigma A^T)_ii = sum_k sum_j A_ij Sigma_jk A_ik
    restricted to the nonzeros of row i of A, Eq. (C.0.3) in [1]. In DALIA
    these are the marginal variances of the linear predictor when Sigma is the
    covariance of x.

    References
    ----------
    .. [1] L. Gaedke-Merzhäuser, "A High-Performance Computing Approach to
       Approximate Bayesian Inference", Doctoral Dissertation 2024INF009,
       Faculty of Informatics, Università della Svizzera italiana, Lugano,
       June 2024. Appendix C, "Marginal Variances and Matrix-Matrix
       Multiplication", pp. 88-89.
    """
    # the kernel walks the rows of A and of Sigma by sorted column index
    a = sp.sparse.csr_matrix(a)
    a.sum_duplicates()
    a.sort_indices()

    sigma = sp.sparse.csr_matrix(sigma)
    sigma.sum_duplicates()
    sigma.sort_indices()

    forms = xp.empty(a.shape[0], dtype=xp.float64)
    args = (
        a.indptr,
        a.indices,
        a.data,
        sigma.indptr,
        sigma.indices,
        sigma.data,
        forms,
    )

    if xp.__name__ == "cupy":
        # one GPU thread per row of A
        threads_per_block = 256
        blocks = (forms.size + threads_per_block - 1) // threads_per_block
        _row_quadratic_forms_kernel[blocks, threads_per_block](*args)
    else:
        _row_quadratic_forms_kernel(*args)

    if bool(xp.isnan(forms).any()):
        raise ValueError(
            "nan in the quadratic forms: an entry of Sigma required by a row of A "
            "is not stored, or the inputs contain nan."
        )

    return forms
