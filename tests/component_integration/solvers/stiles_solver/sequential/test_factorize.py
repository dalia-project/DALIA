# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from dalia import sp


@pytest.mark.mpi_skip()
def test_factorize_logdet_correctness(
    reference_logdet,
    allclose_floats,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
    n_threads,
):
    """The Cholesky factor is only observable through logdet: check it."""
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)

    solver = create_solver(n_threads)
    solver.factorize(A, sparsity="bta" if arrowhead_blocksize > 0 else "bt")

    allclose_floats(
        a_reference=reference_logdet(A),
        b_toverify=solver.logdet(),
        relaxed_tolerance=True,
    )


@pytest.mark.mpi_skip()
def test_factorize_accepts_dense_and_coo_inputs(
    reference_logdet,
    allclose_floats,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    solver = create_solver()

    solver.factorize(A.toarray())
    allclose_floats(reference_logdet(A), solver.logdet(), relaxed_tolerance=True)

    solver.factorize(sp.sparse.coo_matrix(A))
    allclose_floats(reference_logdet(A), solver.logdet(), relaxed_tolerance=True)


@pytest.mark.mpi_skip()
def test_factorize_rejects_non_positive_definite(
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    n = A.shape[0]
    # Flip the sign of the diagonal: symmetric but negative definite.
    A_neg = sp.sparse.csc_matrix(A - 2.0 * sp.sparse.diags(A.diagonal()))

    solver = create_solver()
    with pytest.raises(ValueError, match="positive definite"):
        solver.factorize(A_neg)
    assert n == A_neg.shape[0]
