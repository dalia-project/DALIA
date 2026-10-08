# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.utils import synchronize_gpu

# This is a workaround a problem in cupyx, where linalg is not properly namespaced (directly accessible).
# May be removed in future versions of cupy (tested on cupy 13.4.1).
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import splu
else:
    from scipy.sparse.linalg import splu


class SparseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.LU_factor = None  # Store the LU factorization object
        self.A_inv = None  # Store the inverse of A if needed

        # Solver Metrics
        self.t_factorize = 0.0
        self.t_solve = 0.0

    def factorize(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute the decomposition of a matrix.

        Note: This uses LU decomposition since sparse Cholesky is not readily available.


        Parameters
        ----------
        A : sp.sparse.spmatrix
            The input matrix to decompose.

        Returns
        -------
        None

        Note:
        -----
        Uses the LU decomposition by default as scipy.sparse doesn't implement Cholesky.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        A = sp.sparse.csc_matrix(A)

        # Use LU decomposition as the factorization method
        self.LU_factor = splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

        # Check if the matrix appears to be positive definite
        if not (self.LU_factor.U.diagonal() > 0).all():
            raise ValueError("The matrix does not appear to be positive definite")

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_factorize += toc - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using LU factorization.

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

        if self.LU_factor is None:
            raise ValueError("Matrix factorization not computed")

        # Handle multiple RHS cases
        if rhs.ndim == 1:
            # Single RHS as 1D array
            x = self.LU_factor.solve(rhs)
        elif rhs.ndim == 2 and rhs.shape[1] == 1:
            # Single RHS as column vector
            x = self.LU_factor.solve(rhs.flatten())
            x = x.reshape(rhs.shape)
        elif rhs.ndim == 2 and rhs.shape[1] > 1:
            # Multiple RHS (batched) - scipy splu can handle this directly
            x = self.LU_factor.solve(rhs)
        else:
            raise ValueError(f"Unsupported RHS shape: {rhs.shape}")

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return x

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

        if self.LU_factor is None:
            raise ValueError("Matrix factorization not computed")

        # For LU decomposition: det(A) = det(L) * det(U)
        # Since L has 1s on diagonal: det(L) = 1
        # So det(A) = det(U) = product of diagonal elements of U
        log_det_U = xp.sum(xp.log(xp.abs(self.LU_factor.U.diagonal())))

        return float(log_det_U)

    def selected_inversion(self, batch_size: int = 64, **kwargs) -> sp.sparse.spmatrix:
        """Compute selected entries of the inverse using sparsity pattern of L and U factors.

        This implementation:
        - Processes the matrix in batches to avoid storing full dense inverse in memory
        - Leverages the existing self.solve() method with sparse LU factors
        - Only stores entries matching the sparsity pattern of L and U

        Parameters
        ----------
        batch_size : int, optional
            Number of columns to process in each batch (default: 64).
            Smaller batches use less memory but may be slower.

        Returns
        -------
        sp.sparse.spmatrix
            Sparse matrix with the inverse entries at positions where L or U have non-zeros
        """

        n = self.LU_factor.L.shape[0]

        # Get the combined sparsity pattern of L and U factors
        # This determines which entries of A_inv we need to extract
        L_coo = self.LU_factor.L.tocoo()
        U_coo = self.LU_factor.U.tocoo()

        # Combine patterns: collect all (row, col) pairs where L or U have non-zeros
        # . move to CPU if on GPU to handle set operations
        if xp.__name__ == "cupy":
            L_rows = L_coo.row.get()
            L_cols = L_coo.col.get()
            U_rows = U_coo.row.get()
            U_cols = U_coo.col.get()
        else:
            L_rows = L_coo.row
            L_cols = L_coo.col
            U_rows = U_coo.row
            U_cols = U_coo.col

        # Combine patterns and remove duplicates
        pattern_set = set()
        for r, c in zip(L_rows, L_cols):
            pattern_set.add((int(r), int(c)))
        for r, c in zip(U_rows, U_cols):
            pattern_set.add((int(r), int(c)))

        # Create a set for fast lookup: entries to extract
        pattern_entries = pattern_set

        # Storage for sparse result
        data = []
        row_indices = []
        col_indices = []

        # Process matrix in batches of columns
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_cols = batch_end - batch_start

            # Build batched RHS: identity columns for this batch
            rhs_batch = xp.zeros((n, batch_cols))
            for i in range(batch_cols):
                rhs_batch[batch_start + i, i] = 1.0

            # Solve A @ X = RHS using the existing batched solve method
            # This leverages the sparse LU factorization efficiently
            X = self.solve(rhs_batch)

            # Extract only the entries that match the sparsity pattern
            for row, col in pattern_entries:
                # Check if this (row, col) pair is in the current batch
                if batch_start <= col < batch_end:
                    batch_idx = col - batch_start
                    # Move to CPU if on GPU for data extraction
                    if xp.__name__ == "cupy":
                        value = float(X[row, batch_idx].get())
                    else:
                        value = float(X[row, batch_idx])
                    data.append(value)
                    row_indices.append(row)
                    col_indices.append(col)

        # Convert lists to proper 1D arrays for sparse matrix construction
        data_array = xp.array(data, dtype=xp.float64)
        row_array = xp.array(row_indices, dtype=xp.int32)
        col_array = xp.array(col_indices, dtype=xp.int32)

        # Create sparse matrix from the selected entries
        self.A_inv = sp.sparse.coo_matrix(
            (data_array, (row_array, col_array)), shape=(n, n)
        ).tocsr()

        return self.A_inv

    def _structured_to_spmatrix(
        self, A: sp.sparse.spmatrix, **kwargs
    ) -> sp.sparse.spmatrix:
        """Convert the A_inv matrix to a sparse matrix masked using the given sparsity pattern A.

        Extracts entries from self.A_inv at positions specified by the non-zero pattern of A,
        maintaining sparsity throughout the operation.

        Parameters
        ----------
        A : sp.sparse.spmatrix
            Sparse matrix defining the sparsity pattern to extract.

        Returns
        -------
        sp.sparse.spmatrix
            Sparse matrix with values from A_inv at positions matching A's pattern.
        """

        B = A.tocoo()
        # Extract values from A_inv using element-wise indexing
        B.data = xp.array(
            [float(self.A_inv[int(r), int(c)]) for r, c in zip(B.row, B.col)]
        )
        return B.tocsr()

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.LU_factor is None:
            return 0

        # Estimate memory usage from L and U matrices
        L_memory = (
            self.LU_factor.L.data.nbytes
            + self.LU_factor.L.indptr.nbytes
            + self.LU_factor.L.indices.nbytes
        )
        U_memory = (
            self.LU_factor.U.data.nbytes
            + self.LU_factor.U.indptr.nbytes
            + self.LU_factor.U.indices.nbytes
        )

        if self.A_inv is not None:
            A_inv_memory = self.A_inv.nbytes
            return L_memory + U_memory + A_inv_memory

        return L_memory + U_memory
