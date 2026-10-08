# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod

from dalia import ArrayLike, NDArray
from dalia.configs.dalia_config import SolverConfig


class Solver(ABC):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver.
        """
        self.config = config

    @abstractmethod
    def factorize(self, A: ArrayLike, **kwargs) -> None:
        """Compute the decomposition of a matrix.

        Parameters
        ----------
        A : NDArray | sp.sparse.spmatrix
            The input matrix to decompose.

        Returns
        -------
        None

        Note:
        -----
        Even though precision matrices are known to be positive definite, depending on the underlying solver implementation, this could be Cholesky, LU, or other factorizations.
        """
        ...

    @abstractmethod
    def solve(self, rhs: NDArray, **kwargs) -> NDArray:
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
        ...

    @abstractmethod
    def logdet(self, **kwargs) -> float:
        """Compute the log determinant of the matrix.

        Returns
        -------
        float
            The log determinant of the matrix.
        """
        ...

    @abstractmethod
    def selected_inversion(self, **kwargs) -> NDArray:
        """Compute selected inversion of input matrix using Cholesky factor."""
        ...

    @abstractmethod
    def _structured_to_spmatrix(self, **kwargs) -> None:
        """Convert structured matrix to sparse matrix."""
        ...

    @abstractmethod
    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        ...
