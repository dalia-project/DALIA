# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest


@pytest.mark.mpi_skip()
def test_logdet_correctness(
    reference_logdet,
    allclose_floats,
    create_solver,
    create_spd_matrix,
    diagonal_blocksize,
    n_diag_blocks,
    arrowhead_blocksize,
):
    A = create_spd_matrix(diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
    sparsity = "bta" if arrowhead_blocksize > 0 else "bt"

    solver = create_solver()
    solver.factorize(A, sparsity=sparsity)

    allclose_floats(
        a_reference=reference_logdet(A),
        b_toverify=solver.logdet(sparsity=sparsity),
        relaxed_tolerance=True,
    )


@pytest.mark.mpi_skip()
def test_logdet_before_factorize_raises(create_solver):
    solver = create_solver()
    with pytest.raises(ValueError, match="factoriz"):
        solver.logdet()
