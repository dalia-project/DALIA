# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np

from dalia import backend_flags, sp, xp
from tests import RANDOM_SEED

np.random.seed(RANDOM_SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(RANDOM_SEED))


def _create_solver(
    solver_type: str,
    matrix_size: int,
):
    from dalia.configs.dalia_config import SolverConfig
    from dalia.solvers import DenseSolver

    config = SolverConfig(type=solver_type)

    if solver_type == "dense":
        return DenseSolver(config=config, n=matrix_size)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def _generate_spd_spmatrix(
    matrix_type: str,
    n: int,
):
    """Returns a random, positive definite, matrix.

    Parameters
    ----------
    matrix_type : str
        Type of the matrix: "sparse" or "dense".
    n : int
        Size of the matrix.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, matrix.
    """

    if matrix_type == "sparse":
        L = sp.sparse.random(n, n, density=0.5)
        L = L + sp.sparse.eye(n, format="csr") # Make diagonal entries positive
        L = sp.sparse.tril(L, format="csr")  # lower triangular
        return L @ L.T  # SPD and sparse (but denser than L)
    else:
        L = xp.tril(xp.random.rand(n, n))
        L = L + xp.eye(n)  # Make diagonal entries positive
        return L @ L.T  # SPD and dense
