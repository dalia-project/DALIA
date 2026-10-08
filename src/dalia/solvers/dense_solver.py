# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.utils import synchronize_gpu


## check if sparse matrix is diagonal
def is_diagonal(A: NDArray) -> bool:
    """Check if a matrix is diagonal."""

    coo = A.tocoo()
    return xp.all(coo.row == coo.col)


class DenseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the DenseSolver class.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver.

        Returns
        -------
        None
        """
        super().__init__(config)

        self.n: int = kwargs.get("n", None)
        assert self.n is not None, "The size of the matrix must be provided."

        self.L: NDArray = xp.zeros((self.n, self.n), dtype=xp.float64)
        self.A_inv = None

        # Solver Metrics
        self.t_factorize = 0.0
        self.t_solve = 0.0

    def factorize(self, A: NDArray, **kwargs) -> None:
        """Compute the Cholesky decomposition of a matrix.

        Parameters
        ----------
        A : NDArray
            The input matrix to decompose.

        Returns
        -------
        None

        Note:
        -----
        Uses the Cholesky decomposition.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        if sp.sparse.issparse(A):
            # if A is diagonal, we can use the diagonal directly
            if is_diagonal(A):
                self.L[:] = 0
                self.L[xp.arange(self.n), xp.arange(self.n)] = xp.sqrt(A.diagonal())
                return

            else:
                self.L[:] = A.todense()
        else:
            self.L[:] = A

        self.L = xp.linalg.cholesky(self.L)

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_factorize += toc - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor.

        Parameters
        ----------
        rhs : NDArray
            Right-hand side of the linear system.

        Returns
        -------
        NDArray
            Solution of the linear system.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        rhs[:] = sp.linalg.solve_triangular(
            self.L,
            rhs,
            lower=True,
        )
        rhs[:] = sp.linalg.solve_triangular(
            self.L,
            rhs,
            trans="T",
            lower=True,
        )

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute the log determinant of the matrix.

        Returns
        -------
        float
            The log determinant of the matrix.
        """
        return 2 * xp.sum(xp.log(xp.diag(self.L)))

    def selected_inversion(self, **kwargs) -> None:
        L_inv = sp.linalg.solve_triangular(
            self.L,
            xp.eye(self.L.shape[0]),
            lower=True,
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes."""
        solver_mem = 2 * self.n * self.n * xp.dtype(xp.float64).itemsize

        return solver_mem
