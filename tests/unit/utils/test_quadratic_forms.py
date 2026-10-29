# Quadratic forms of the rows of A, diag(A Sigma A^T), computed from the stored
# upper triangle of Sigma: the kernel against the dense product, and the error
# raised when a required entry of Sigma is missing.

import numpy as np
import pytest
import scipy.sparse as sps

from dalia import sp, xp
from dalia.utils import get_host, row_quadratic_forms


def random_inputs(m=40, n=15):
    """Sparse A with an empty row and a dense SPD Sigma, both as host arrays."""
    rng = np.random.default_rng(3)
    A = sps.random(m, n, density=0.3, random_state=rng, format="csr")
    A.data[:] = rng.normal(size=A.nnz)
    A = A.tolil()
    A[5, :] = 0.0
    A = sps.csr_matrix(A)
    A.sort_indices()
    B = rng.normal(size=(n, n))
    return A, B @ B.T + n * np.eye(n)


def test_row_quadratic_forms_match_dense_product():
    A, Sigma = random_inputs()
    upper = sp.sparse.csr_matrix(xp.asarray(np.triu(Sigma)))

    forms = row_quadratic_forms(sp.sparse.csr_matrix(A), upper)

    A = A.toarray()
    reference = np.diag(A @ Sigma @ A.T)
    np.testing.assert_allclose(get_host(forms), reference, rtol=1e-12, atol=1e-12)


def test_row_quadratic_forms_raise_on_a_missing_entry():
    A, Sigma = random_inputs()
    # only the diagonal of Sigma is stored, the rows of A need off-diagonal entries
    diagonal = sp.sparse.csr_matrix(xp.asarray(np.diag(np.diag(Sigma))))

    with pytest.raises(ValueError):
        row_quadratic_forms(sp.sparse.csr_matrix(A), diagonal)
