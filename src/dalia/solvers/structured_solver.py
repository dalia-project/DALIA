# Copyright 2024-2025 DALIA authors. All rights reserved.

import time
from warnings import warn

from dalia import NDArray, sp, xp, xp_host
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.kernels.blockmapping import compute_block_slice, compute_block_sort_index
from dalia.utils import print_msg, synchronize_gpu

try:
    from serinv.algs import pobtaf, pobtas, pobtasi, pobtf, pobts, pobtsi
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        diagonal_blocksize: int,
        n_diag_blocks: int,
        arrowhead_blocksize: int = 0,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = diagonal_blocksize
        self.arrowhead_blocksize: int = arrowhead_blocksize
        self.n_diag_blocks: int = n_diag_blocks

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks: NDArray = xp.empty(
            (self.n_diag_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=xp.float64,
        )
        self.A_lower_diagonal_blocks: NDArray = xp.empty(
            (
                self.n_diag_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=xp.float64,
        )
        self.A_arrow_bottom_blocks: NDArray = None
        self.A_arrow_tip_block: NDArray = None
        if self.arrowhead_blocksize > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (self.n_diag_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_blocksize, self.arrowhead_blocksize),
                dtype=xp.float64,
            )

        # If running with cupy, initialize the caching strategy
        self.bta_cache_block_sort_index = None
        self.bt_cache_block_sort_index = None

        # Solver Metrics
        self.total_bytes: int = 0

        self.t_factorize = 0.0
        self.t_solve = 0.0

    def factorize(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Compute the decomposition of a matrix.

        Parameters
        ----------
        A : sp.sparse.spmatrix
            The input matrix to decompose.
        sparsity : str
            The sparsity pattern of the matrix. Either 'bt' or 'bta'.

        Returns
        -------
        None

        Note:
        -----
        Uses the Cholesky decomposition.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        self._spmatrix_to_structured(A, sparsity)
        if sparsity == "bta":
            pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        elif sparsity == "bt":
            pobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_factorize += toc - tic

    def solve(
        self,
        rhs: NDArray,
        sparsity: str,
    ) -> NDArray:
        """Solve linear system using Cholesky factor.

        Parameters
        ----------
        rhs : NDArray
            Right-hand side of the linear system.
        sparsity : str
            The sparsity pattern of the matrix. Either 'bt' or 'bta'.

        Returns
        -------
        NDArray
            Solution of the linear system.

        Raises
        ------
        ValueError
            If the sparsity pattern is unknown.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        if sparsity == "bta":
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="N",
            )
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="C",
            )
        elif sparsity == "bt":
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="N",
            )
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="C",
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return rhs

    def logdet(
        self,
        sparsity: str,
    ) -> float:
        """Compute the log determinant of the matrix.

        Parameters
        ----------
        sparsity : str
            The sparsity pattern of the matrix. Either 'bt' or 'bta'.

        Returns
        -------
        float
            The log determinant of the matrix.
        """
        logdet: float = 0.0
        for i in range(self.n_diag_blocks):
            logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        if sparsity == "bta":
            logdet += xp.sum(xp.log(self.A_arrow_tip_block.diagonal()))

        if xp.isnan(logdet):
            raise ValueError("Logdet is NaN. Check the input matrix.")

        return 2 * logdet

    def selected_inversion(
        self,
        sparsity: str,
    ) -> None:
        """Compute selected inversion of input matrix using Cholesky factor."""
        # self._spmatrix_to_structured(A, sparsity)

        if sparsity == "bta":
            pobtasi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        elif sparsity == "bt":
            pobtsi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

    def _spmatrix_to_structured(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Map sp.spmatrix to BT or BTA."""
        self.A_diagonal_blocks[:] = 0.0
        self.A_lower_diagonal_blocks[:] = 0.0
        if self.A_arrow_bottom_blocks is not None:
            self.A_arrow_bottom_blocks[:] = 0.0
        if self.A_arrow_tip_block is not None:
            self.A_arrow_tip_block[:] = 0.0

        if xp.__name__ == "cupy":
            if sparsity == "bta":
                self._spmatrix_to_bta(A)
            elif sparsity == "bt":
                self._spmatrix_to_bt(A)
            else:
                raise ValueError(
                    f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
                )
        else:
            # CPU fall-back strategy
            A_csc = sp.sparse.csc_matrix(A)
            for i in range(self.n_diag_blocks):
                block_slice = A_csc[
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                self.A_diagonal_blocks[i][
                    block_slice.row, block_slice.col
                ] = block_slice.data

                if i < self.n_diag_blocks - 1:
                    block_slice = A_csc[
                        (i + 1)
                        * self.diagonal_blocksize : (i + 2)
                        * self.diagonal_blocksize,
                        i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    ].tocoo()
                    self.A_lower_diagonal_blocks[i][
                        block_slice.row, block_slice.col
                    ] = block_slice.data

                if sparsity == "bta":
                    block_slice = A_csc[
                        -self.arrowhead_blocksize :,
                        i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    ].tocoo()
                    self.A_arrow_bottom_blocks[i][
                        block_slice.row, block_slice.col
                    ] = block_slice.data

            if sparsity == "bta":
                block_slice = A_csc[
                    -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
                ].tocoo()
                self.A_arrow_tip_block[block_slice.row, block_slice.col] = (
                    block_slice.data
                )

    def _spmatrix_to_bta(
        self,
        A: sp.sparse.spmatrix,
    ) -> None:
        coo = sp.sparse.coo_matrix(A)

        if self.bta_cache_block_sort_index is None:
            block_sizes = xp_host.array(
                [self.diagonal_blocksize for _ in range(self.n_diag_blocks)]
                + [self.arrowhead_blocksize]
            )
            self.bta_cache_block_sort_index = compute_block_sort_index(
                coo.row, coo.col, block_sizes
            )
            block_offsets = xp_host.hstack(
                ([0], xp_host.cumsum(block_sizes)), dtype=xp_host.int32
            )

            rows = coo.row[self.bta_cache_block_sort_index]
            cols = coo.col[self.bta_cache_block_sort_index]

            self.bta_diag_rows = []
            self.bta_diag_cols = []
            self.bta_diag_slices = []

            self.bta_lower_rows = []
            self.bta_lower_cols = []
            self.bta_lower_slice = []

            self.bta_arrow_bottom_rows = []
            self.bta_arrow_bottom_cols = []
            self.bta_arrow_bottom_slice = []

            self.bta_arrow_tip_rows = None
            self.bta_arrow_tip_cols = None
            self.bta_arrow_tip_slice = None

            for i in range(self.n_diag_blocks):
                inds = compute_block_slice(
                    rows, cols, block_offsets, block_row=i, block_col=i
                )
                slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
                self.bta_diag_rows.append(rows[slice_idx] - block_offsets[i])
                self.bta_diag_cols.append(cols[slice_idx] - block_offsets[i])
                self.bta_diag_slices.append(slice_idx)

                if i < self.n_diag_blocks - 1:
                    inds = compute_block_slice(
                        rows, cols, block_offsets, block_row=i + 1, block_col=i
                    )
                    slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
                    self.bta_lower_rows.append(rows[slice_idx] - block_offsets[i + 1])
                    self.bta_lower_cols.append(cols[slice_idx] - block_offsets[i])
                    self.bta_lower_slice.append(slice_idx)

                inds = compute_block_slice(
                    rows,
                    cols,
                    block_offsets,
                    block_row=self.n_diag_blocks,
                    block_col=i,
                )
                slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
                self.bta_arrow_bottom_rows.append(
                    rows[slice_idx] - block_offsets[self.n_diag_blocks]
                )
                self.bta_arrow_bottom_cols.append(cols[slice_idx] - block_offsets[i])
                self.bta_arrow_bottom_slice.append(slice_idx)

            # Arrow tip block
            inds = compute_block_slice(
                rows,
                cols,
                block_offsets,
                block_row=self.n_diag_blocks,
                block_col=self.n_diag_blocks,
            )
            slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
            self.bta_arrow_tip_rows = (
                rows[slice_idx] - block_offsets[self.n_diag_blocks]
            )
            self.bta_arrow_tip_cols = (
                cols[slice_idx] - block_offsets[self.n_diag_blocks]
            )
            self.bta_arrow_tip_slice = slice_idx

            self.bta_diag_rows = xp.array(self.bta_diag_rows, dtype=xp.int32)
            self.bta_diag_cols = xp.array(self.bta_diag_cols, dtype=xp.int32)
            self.bta_lower_rows = xp.array(self.bta_lower_rows, dtype=xp.int32)
            self.bta_lower_cols = xp.array(self.bta_lower_cols, dtype=xp.int32)
            self.bta_arrow_bottom_rows = xp.array(
                self.bta_arrow_bottom_rows, dtype=xp.int32
            )
            self.bta_arrow_bottom_cols = xp.array(
                self.bta_arrow_bottom_cols, dtype=xp.int32
            )
            self.bta_arrow_tip_rows = xp.array(self.bta_arrow_tip_rows, dtype=xp.int32)
            self.bta_arrow_tip_cols = xp.array(self.bta_arrow_tip_cols, dtype=xp.int32)

            # Print the allocated memory for the BTA-array
            total_bta_bytes: int = (
                self.bta_diag_rows.nbytes
                + self.bta_diag_cols.nbytes
                + self.bta_lower_rows.nbytes
                + self.bta_lower_cols.nbytes
                + self.bta_arrow_bottom_rows.nbytes
                + self.bta_arrow_bottom_cols.nbytes
                + self.bta_arrow_tip_rows.nbytes
                + self.bta_arrow_tip_cols.nbytes
            )
            self.total_bytes += total_bta_bytes
            print_msg(
                f"Allocated an extra {total_bta_bytes / (1024**3):.2f} GB for BTA_mapping caching (total: {self.total_bytes / (1024**3):.2f})"
            )

        # Sort the data:
        data = coo.data[self.bta_cache_block_sort_index]
        for i in range(self.n_diag_blocks):
            self.A_diagonal_blocks[i][
                self.bta_diag_rows[i],
                self.bta_diag_cols[i],
            ] = data[self.bta_diag_slices[i]]

            if i < self.n_diag_blocks - 1:
                self.A_lower_diagonal_blocks[i][
                    self.bta_lower_rows[i],
                    self.bta_lower_cols[i],
                ] = data[self.bta_lower_slice[i]]

            self.A_arrow_bottom_blocks[i][
                self.bta_arrow_bottom_rows[i],
                self.bta_arrow_bottom_cols[i],
            ] = data[self.bta_arrow_bottom_slice[i]]

        # Arrow tip block
        self.A_arrow_tip_block[
            self.bta_arrow_tip_rows,
            self.bta_arrow_tip_cols,
        ] = data[self.bta_arrow_tip_slice]

    def _spmatrix_to_bt(
        self,
        A: sp.sparse.spmatrix,
    ) -> None:
        coo = sp.sparse.coo_matrix(A)

        if self.bt_cache_block_sort_index is None:
            block_sizes = xp_host.array(
                [self.diagonal_blocksize for _ in range(self.n_diag_blocks)]
                + [self.arrowhead_blocksize]
            )
            self.bt_cache_block_sort_index = compute_block_sort_index(
                coo.row, coo.col, block_sizes
            )
            block_offsets = xp_host.hstack(
                ([0], xp_host.cumsum(block_sizes)), dtype=xp_host.int32
            )

            rows = coo.row[self.bt_cache_block_sort_index]
            cols = coo.col[self.bt_cache_block_sort_index]

            self.bt_diag_rows = []
            self.bt_diag_cols = []
            self.bt_diag_slices = []

            self.bt_lower_rows = []
            self.bt_lower_cols = []
            self.bt_lower_slice = []

            for i in range(self.n_diag_blocks):
                inds = compute_block_slice(
                    rows, cols, block_offsets, block_row=i, block_col=i
                )
                slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
                self.bt_diag_rows.append(rows[slice_idx] - block_offsets[i])
                self.bt_diag_cols.append(cols[slice_idx] - block_offsets[i])
                self.bt_diag_slices.append(slice_idx)

                if i < self.n_diag_blocks - 1:
                    inds = compute_block_slice(
                        rows, cols, block_offsets, block_row=i + 1, block_col=i
                    )
                    slice_idx = slice(int(inds[0]), int(inds[-1] + 1), 1)
                    self.bt_lower_rows.append(rows[slice_idx] - block_offsets[i + 1])
                    self.bt_lower_cols.append(cols[slice_idx] - block_offsets[i])
                    self.bt_lower_slice.append(slice_idx)

            self.bt_diag_rows = xp.array(self.bt_diag_rows, dtype=xp.int32)
            self.bt_diag_cols = xp.array(self.bt_diag_cols, dtype=xp.int32)
            self.bt_lower_rows = xp.array(self.bt_lower_rows, dtype=xp.int32)
            self.bt_lower_cols = xp.array(self.bt_lower_cols, dtype=xp.int32)

            # Print the allocated memory for the BT array
            total_bt_bytes: int = (
                self.bt_diag_rows.nbytes
                + self.bt_diag_cols.nbytes
                + self.bt_lower_rows.nbytes
                + self.bt_lower_cols.nbytes
            )
            self.total_bytes += total_bt_bytes
            print_msg(
                f"Allocated an extra {total_bt_bytes / (1024**3):.2f} GB for BT_mapping caching (total: {self.total_bytes / (1024**3):.2f})"
            )

        # Sort the data:
        data = coo.data[self.bt_cache_block_sort_index]
        for i in range(self.n_diag_blocks):
            self.A_diagonal_blocks[i][
                self.bt_diag_rows[i],
                self.bt_diag_cols[i],
            ] = data[self.bt_diag_slices[i]]

            if i < self.n_diag_blocks - 1:
                self.A_lower_diagonal_blocks[i][
                    self.bt_lower_rows[i],
                    self.bt_lower_cols[i],
                ] = data[self.bt_lower_slice[i]]

    def _structured_to_spmatrix(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
        symmetrize: bool = True,
    ) -> sp.sparse.spmatrix:
        """Map a BT or BTA structured matrix to a sparse `csc` format given the sparsity pattern.

        Parameters
        ----------
        A : sp.sparse.spmatrix
            Input matrix in sparse format.
        sparsity : str
            The sparsity pattern of the matrix. Either 'bt' or 'bta'.
        symmetrize : bool, optional
            Whether to symmetrize the output matrix, by default True.

        Returns
        -------
        sp.sparse.spmatrix
            The output matrix in sparse format.

        Notes
        -----
        By default this function symmetrize the matrix as it is mostly use in DALIA on the selected
        inverse of SPD matrices. Because of further matrix operations on these matrices, we symmetrize
        them to avoid numerical asymmetries.

        Raises
        ------
        ValueError
            If the sparsity pattern is unknown.

        """

        # A is assumed to be symmetric, only use lower triangular part
        B = sp.sparse.csc_matrix(sp.sparse.tril(sp.sparse.csc_matrix(A)))

        data = []
        rows = []
        cols = []

        for i in range(self.n_diag_blocks):
            # Extract the sparsity pattern of the current Diagonal, Lower, and Arrowhead blocks
            B_coo = B[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            row_diag, col_diag = B_coo.row, B_coo.col
            data.append(self.A_diagonal_blocks[i, row_diag, col_diag].flatten())
            rows.append(i * self.diagonal_blocksize + row_diag)
            cols.append(i * self.diagonal_blocksize + col_diag)

            if i < self.n_diag_blocks - 1:
                B_coo = B[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_lower, col_lower = B_coo.row, B_coo.col
                data.append(
                    self.A_lower_diagonal_blocks[i, row_lower, col_lower].flatten()
                )
                rows.append((i + 1) * self.diagonal_blocksize + row_lower)
                cols.append(i * self.diagonal_blocksize + col_lower)

            if sparsity == "bta":
                B_coo = B[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_arrow, col_arrow = B_coo.row, B_coo.col
                data.append(
                    self.A_arrow_bottom_blocks[i, row_arrow, col_arrow].flatten()
                )
                rows.append(self.n_diag_blocks * self.diagonal_blocksize + row_arrow)
                cols.append(i * self.diagonal_blocksize + col_arrow)

        if sparsity == "bta":
            B_coo = B[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :].tocoo()
            row_arrow_tip, col_arrow_tip = B_coo.row, B_coo.col
            data.append(self.A_arrow_tip_block[row_arrow_tip, col_arrow_tip].flatten())
            rows.append(self.n_diag_blocks * self.diagonal_blocksize + row_arrow_tip)
            cols.append(self.n_diag_blocks * self.diagonal_blocksize + col_arrow_tip)

        # Create the sparse matrix from the data, rows, and cols
        data = xp.concatenate(data)
        rows = xp.concatenate(rows)
        cols = xp.concatenate(cols)

        B_out = sp.sparse.coo_matrix((data, (rows, cols)), shape=B.shape).tocsc()

        if symmetrize:
            return B_out + sp.sparse.tril(B_out, k=-1).T
        else:
            return B_out

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        bytes_pobtars: int = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        self.total_bytes += bytes_pobtars

        return self.total_bytes
