# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest
import scipy.sparse as sparse_host

from dalia import sp, xp


def _drop_arrow(A, arrowhead_blocksize):
    """Copy of `A` (BTA) without its arrow blocks: a BT matrix with the tip kept."""
    A_host = (A.get() if hasattr(A, "get") else A).tolil()
    n = A_host.shape[0]
    A_host[n - arrowhead_blocksize :, : n - arrowhead_blocksize] = 0.0
    A_host[: n - arrowhead_blocksize, n - arrowhead_blocksize :] = 0.0
    A_host = sparse_host.csc_matrix(A_host)
    A_host.eliminate_zeros()
    return sp.sparse.csc_matrix(A_host)


@pytest.mark.mpi_skip()
def test_alternating_sparsity_patterns_reuse_one_analysis(
    reference_logdet,
    reference_solve,
    allclose_floats,
    allclose_ndarrays,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
):
    """DALIA alternates Q_prior ('bt') and Q_conditional ('bta') every
    evaluation. The 'bt' pattern is a subset of the 'bta' one, so after the
    first 'bta' the solver must keep a single symbolic analysis and only
    remap values."""
    arrowhead_blocksize = 2
    A_bta = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    n = A_bta.shape[0]
    A_bt = _drop_arrow(A_bta, arrowhead_blocksize)

    solver = create_solver()

    # bt first: analysis on the small pattern
    solver.factorize(A_bt, sparsity="bt")
    allclose_floats(
        reference_logdet(A_bt), solver.logdet(sparsity="bt"), relaxed_tolerance=True
    )
    assert solver.n_analyses == 1

    # bta: pattern grows -> exactly one re-analysis on the union pattern
    solver.factorize(A_bta, sparsity="bta")
    allclose_floats(
        reference_logdet(A_bta), solver.logdet(sparsity="bta"), relaxed_tolerance=True
    )
    assert solver.n_analyses == 2

    # back and forth: no further analyses, values remapped into the union
    for _ in range(3):
        solver.factorize(A_bt, sparsity="bt")
        allclose_floats(
            reference_logdet(A_bt), solver.logdet(sparsity="bt"), relaxed_tolerance=True
        )
        solver.factorize(A_bta, sparsity="bta")
        allclose_floats(
            reference_logdet(A_bta),
            solver.logdet(sparsity="bta"),
            relaxed_tolerance=True,
        )
        b = xp.random.rand(n)
        allclose_ndarrays(
            reference_solve(A_bta, b.copy()),
            solver.solve(b.copy(), sparsity="bta"),
            relaxed_tolerance=True,
        )
    assert solver.n_analyses == 2

    # a new value set on a known pattern must be detected even under the same key
    A_scaled = sp.sparse.csc_matrix(2.0 * A_bta)
    solver.factorize(A_scaled, sparsity="bta")
    allclose_floats(
        reference_logdet(A_scaled),
        solver.logdet(sparsity="bta"),
        relaxed_tolerance=True,
    )
    assert solver.n_analyses == 2


@pytest.mark.mpi_skip()
def test_selected_inversion_after_pattern_growth(
    reference_inversion,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
):
    """The selected inverse must be extracted on the caller's pattern, not on
    the (larger) union pattern the solver analysed."""
    A_bta = create_spd_matrix(diagonal_blocksize, n_diag_blocks, 2)
    A_bt = _drop_arrow(A_bta, 2)

    solver = create_solver()
    solver.factorize(A_bta, sparsity="bta")
    solver.factorize(A_bt, sparsity="bt")
    solver.selected_inversion(sparsity="bt")
    A_selinv = solver._structured_to_spmatrix(A_bt, sparsity="bt")

    assert A_selinv.nnz == A_bt.nnz
    ref = reference_inversion(A_bt)
    coo = A_selinv.tocoo()
    assert xp.allclose(coo.data, ref[coo.row, coo.col], rtol=1e-10, atol=1e-12)
