# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from dalia import sp, xp
from tests import ATOLS, RTOLS


@pytest.mark.mpi_skip()
def test_selected_inversion_correctness(
    reference_inversion,
    allclose_dense_structured,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
    n_threads,
):
    """Selected inverse restricted to the pattern of A matches the dense inverse."""
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    sparsity = "bta" if arrowhead_blocksize > 0 else "bt"

    solver = create_solver(n_threads)
    solver.factorize(A, sparsity=sparsity)
    solver.selected_inversion(sparsity=sparsity)

    A_selinv = solver._structured_to_spmatrix(A, sparsity=sparsity)
    assert sp.sparse.issparse(A_selinv)
    assert A_selinv.nnz == A.nnz

    allclose_dense_structured(
        A_reference=reference_inversion(A),
        B_toverify=A_selinv.toarray(),
        diagonal_blocksize=diagonal_blocksize,
        n_diag_blocks=n_diag_blocks,
        arrowhead_blocksize=arrowhead_blocksize,
        assert_upper_triangle=True,
    )


@pytest.mark.mpi_skip()
def test_marginal_variances_from_identity_pattern(
    reference_inversion,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    """DALIA extracts marginal variances with an identity pattern."""
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    n = A.shape[0]

    solver = create_solver()
    solver.factorize(A)
    solver.selected_inversion()

    variances = solver._structured_to_spmatrix(sp.sparse.eye(n, dtype=xp.float64))

    assert xp.allclose(
        variances.diagonal(),
        xp.diag(reference_inversion(A)),
        rtol=RTOLS["relaxed"],
        atol=ATOLS["relaxed"],
    )


@pytest.mark.mpi_skip()
def test_selected_inversion_before_factorize_raises(create_solver):
    solver = create_solver()
    with pytest.raises(ValueError, match="factoriz"):
        solver.selected_inversion()
