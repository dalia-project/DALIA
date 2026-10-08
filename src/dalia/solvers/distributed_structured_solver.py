# Copyright 2024-2025 DALIA authors. All rights reserved.

import time
from warnings import warn

from dalia import NDArray, backend_flags, sp, xp, xp_host
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.kernels.blockmapping import compute_block_slice, compute_block_sort_index
from dalia.utils import allgather, allreduce, print_msg, synchronize

if backend_flags["mpi_avail"]:
    from mpi4py import MPI
    from mpi4py.MPI import Comm as mpi_comm
else:
    mpi_comm = None

try:
    from serinv.utils import allocate_pobtax_permutation_buffers
    from serinv.wrappers import (
        allocate_pobtars,
        allocate_pobtrs,
        ppobtaf,
        ppobtas,
        ppobtasi,
        ppobtf,
        ppobts,
        ppobtsi,
    )
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")

if backend_flags["cupy_avail"]:
    import cupyx as cpx


class DistSerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        diagonal_blocksize: int,
        n_diag_blocks: int,
        comm: mpi_comm,
        arrowhead_blocksize: int = 0,
        nccl_comm: object = None,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = diagonal_blocksize
        self.arrowhead_blocksize: int = arrowhead_blocksize
        self.n_diag_blocks: int = n_diag_blocks
        self.comm: mpi_comm = comm
        self.rank: int = self.comm.Get_rank()
        self.comm_size: int = self.comm.size
        self.nccl_comm = nccl_comm

        # Allocating the local slices of the system matrix
        self.n_locals = [n_diag_blocks // self.comm_size] * self.comm_size
        remainder = n_diag_blocks % self.comm_size
        self.n_locals[0] += remainder

        print_msg(
            f"Distributed solver slicing: {self.n_locals}",
            flush=True,
        )

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks: NDArray = xp.empty(
            (
                self.n_locals[self.rank],
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=xp.float64,
        )
        # Here is the "arrowhead" slicing
        if self.rank == self.comm_size - 1:
            self.A_lower_diagonal_blocks: NDArray = xp.empty(
                (
                    self.n_locals[self.rank] - 1,
                    self.diagonal_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
        else:
            self.A_lower_diagonal_blocks: NDArray = xp.empty(
                (
                    self.n_locals[self.rank],
                    self.diagonal_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
        self.A_arrow_bottom_blocks: NDArray = None
        self.A_arrow_tip_block: NDArray = None
        if self.arrowhead_blocksize > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (
                    self.n_locals[self.rank],
                    self.arrowhead_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_blocksize, self.arrowhead_blocksize),
                dtype=xp.float64,
            )

        n_rhs = 1
        self.dist_rhs: NDArray = xp.empty(
            (
                self.n_locals[self.rank] * self.diagonal_blocksize
                + self.arrowhead_blocksize,
                n_rhs,
            ),
            dtype=xp.float64,
        )

        self.max_n_locals = max(self.n_locals)
        # self.local_remainder = self.max_n_locals - self.n_locals[self.rank]
        self.remainders = [
            self.max_n_locals - self.n_locals[r] for r in range(self.comm_size)
        ]

        self.send_rhs: NDArray = None
        self.recv_rhs: NDArray = None
        if (
            backend_flags["array_module"] == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not backend_flags["nccl_avail"]
        ):
            # Allocate pinned_memory communciation array
            self.send_rhs = cpx.zeros_pinned(
                (self.max_n_locals * self.diagonal_blocksize,),
                dtype=self.dist_rhs.dtype,
            )
            self.recv_rhs = cpx.zeros_pinned(
                (self.max_n_locals * self.comm_size * self.diagonal_blocksize,),
                dtype=self.dist_rhs.dtype,
            )
        else:
            self.send_rhs = xp.zeros(
                (self.max_n_locals * self.diagonal_blocksize,),
                dtype=self.dist_rhs.dtype,
            )
            self.recv_rhs = xp.zeros(
                (self.max_n_locals * self.comm_size * self.diagonal_blocksize,),
                dtype=self.dist_rhs.dtype,
            )

        # --- Allocate reduced system and distribution buffers
        self.buffer = allocate_pobtax_permutation_buffers(
            self.A_diagonal_blocks,
        )

        if self.arrowhead_blocksize > 0:
            self.reduced_system: dict = allocate_pobtars(
                A_diagonal_blocks=self.A_diagonal_blocks,
                A_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                A_lower_arrow_blocks=self.A_arrow_bottom_blocks,
                A_arrow_tip_block=self.A_arrow_tip_block,
                B=self.dist_rhs,
                comm=self.comm,
                array_module=xp.__name__,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )
        else:
            self.reduced_system: dict = allocate_pobtrs(
                A_diagonal_blocks=self.A_diagonal_blocks,
                A_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                B=self.dist_rhs,
                comm=self.comm,
                array_module=xp.__name__,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )

        # Initialize the caching strategy
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
        synchronize(comm=self.comm)
        tic = time.perf_counter()

        # Reset the tip block for reccurrent calls
        # print(f"WorldRank {self.rank} ENTERING {sparsity} cholesky.", flush=True)
        self._spmatrix_to_structured(A, sparsity)

        if sparsity == "bta":
            ppobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                buffer=self.buffer,
                pobtars=self.reduced_system,
                comm=self.comm,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )
        elif sparsity == "bt":
            ppobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                buffer=self.buffer,
                pobtrs=self.reduced_system,
                comm=self.comm,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        synchronize(comm=self.comm)
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
        synchronize(comm=self.comm)
        tic = time.perf_counter()

        # Store the original shape of rhs to reshape the solution back after solving
        in_rhs_shape = rhs.shape

        # Ensure rhs is a 2D array
        if rhs.ndim == 1:
            rhs = rhs[:, None]

        # Handle multiple RHS by processing each column separately
        for col in range(rhs.shape[1]):
            rhs_col = rhs[:, col : col + 1]  # Keep 2D shape with single column

            self._slice_rhs(rhs_col, sparsity)

            if sparsity == "bta":
                ppobtas(
                    L_diagonal_blocks=self.A_diagonal_blocks,
                    L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                    L_lower_arrow_blocks=self.A_arrow_bottom_blocks,
                    L_arrow_tip_block=self.A_arrow_tip_block,
                    B=self.dist_rhs,
                    buffer=self.buffer,
                    pobtars=self.reduced_system,
                    comm=self.comm,
                    strategy="allgather",
                    nccl_comm=self.nccl_comm,
                )
            elif sparsity == "bt":
                ppobts(
                    L_diagonal_blocks=self.A_diagonal_blocks,
                    L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                    B=(
                        self.dist_rhs[: -self.arrowhead_blocksize]
                        if self.arrowhead_blocksize > 0
                        else self.dist_rhs
                    ),
                    buffer=self.buffer,
                    pobtrs=self.reduced_system,
                    comm=self.comm,
                    strategy="allgather",
                    nccl_comm=self.nccl_comm,
                )
            else:
                raise ValueError(
                    f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
                )

            self._gather_rhs(rhs_col, sparsity)

            # Copy the solved column back to the original rhs
            rhs[:, col : col + 1] = rhs_col

        synchronize(comm=self.comm)
        toc = time.perf_counter()
        self.t_solve += toc - tic

        # Reshape the solution back to the original shape if needed
        if in_rhs_shape != rhs.shape:
            rhs = rhs.reshape(in_rhs_shape)

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
        logdet = xp.array(0.0, dtype=xp.float64)

        if self.rank == 0:
            # Do its local blocks
            for i in range(self.n_locals[self.rank] - 1):
                logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

            # Rank 0 do the reduced system; The loop start from 1 because of the
            # AllGather strategy and the size of the reduced system associated.
            _n = self.reduced_system["A_diagonal_blocks"].shape[0]
            for i in range(1, _n):
                logdet += xp.sum(
                    xp.log(self.reduced_system["A_diagonal_blocks"][i].diagonal())
                )

            if sparsity == "bta":
                logdet += xp.sum(
                    xp.log(self.reduced_system["A_arrow_tip_block"].diagonal())
                )
        else:
            for i in range(1, self.n_locals[self.rank] - 1):
                logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        synchronize(comm=self.comm)
        logdet = allreduce(
            logdet,
            op="sum",
            comm=self.comm,
        )
        synchronize(comm=self.comm)

        if xp.isnan(logdet):
            raise ValueError(
                f"WorldRank {MPI.COMM_WORLD.rank} logdet is NaN for {sparsity} matrix."
            )

        return 2 * logdet

    def selected_inversion(
        self,
        sparsity: str,
    ) -> None:
        """Compute selected inversion of input matrix using Cholesky factor."""
        if sparsity == "bta":
            ppobtasi(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                L_lower_arrow_blocks=self.A_arrow_bottom_blocks,
                L_arrow_tip_block=self.A_arrow_tip_block,
                buffer=self.buffer,
                pobtars=self.reduced_system,
                comm=self.comm,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )
        elif sparsity == "bt":
            ppobtsi(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                buffer=self.buffer,
                pobtrs=self.reduced_system,
                comm=self.comm,
                strategy="allgather",
                nccl_comm=self.nccl_comm,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        synchronize(comm=self.comm)

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

        if xp.__name__ == "cupy" and sparsity == "bta":
            if sparsity == "bta":
                self._spmatrix_to_bta(A)
            elif sparsity == "bt":
                self._spmatrix_to_bt(A)
            else:
                raise ValueError(
                    f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
                )
        else:
            A_csc = sp.sparse.csc_matrix(A)

            n_idx = xp.array([0] + self.n_locals)
            start_idx = int(xp.cumsum(n_idx)[self.rank])
            end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
            for i_A in range(start_idx, end_idx):
                i_S = i_A - start_idx
                block_slice = A_csc[
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].tocoo()
                self.A_diagonal_blocks[i_S][
                    block_slice.row, block_slice.col
                ] = block_slice.data

                if i_A < self.n_diag_blocks - 1:
                    block_slice = A_csc[
                        (i_A + 1)
                        * self.diagonal_blocksize : (i_A + 2)
                        * self.diagonal_blocksize,
                        i_A
                        * self.diagonal_blocksize : (i_A + 1)
                        * self.diagonal_blocksize,
                    ].tocoo()
                    self.A_lower_diagonal_blocks[i_S][
                        block_slice.row, block_slice.col
                    ] = block_slice.data

                if sparsity == "bta":
                    block_slice = A_csc[
                        -self.arrowhead_blocksize :,
                        i_A
                        * self.diagonal_blocksize : (i_A + 1)
                        * self.diagonal_blocksize,
                    ].tocoo()
                    self.A_arrow_bottom_blocks[i_S][
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

            n_idx = xp.array([0] + self.n_locals)
            start_idx = int(xp.cumsum(n_idx)[self.rank])
            end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
            for i in range(start_idx, end_idx):
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
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        for i in range(self.n_locals[self.rank]):
            self.A_diagonal_blocks[i][
                self.bta_diag_rows[i],
                self.bta_diag_cols[i],
            ] = data[self.bta_diag_slices[i]]

            i_global = i + start_idx
            if i_global < self.n_diag_blocks - 1:
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

            n_idx = xp.array([0] + self.n_locals)
            start_idx = int(xp.cumsum(n_idx)[self.rank])
            end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
            for i in range(start_idx, end_idx):
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

            # Print the allocated memory for the BT-array
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
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        for i in range(self.n_locals[self.rank]):
            self.A_diagonal_blocks[i][
                self.bt_diag_rows[i],
                self.bt_diag_cols[i],
            ] = data[self.bt_diag_slices[i]]

            i_global = i + start_idx
            if i_global < self.n_diag_blocks - 1:
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
        """Map BT or BTA matrix to sp.spmatrix using sparsity pattern provided in A."""
        # A is assumed to be symmetric, only use lower triangular part
        B = sp.sparse.csc_matrix(sp.sparse.tril(sp.sparse.csc_matrix(A)))

        data = []
        rows = []
        cols = []

        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
        for i_A in range(start_idx, end_idx):
            i_S = i_A - start_idx
            # Extract the sparsity pattern of the current Diagonal, Lower, and Arrowhead blocks
            B_coo = B[
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
            ].tocoo()
            row_diag, col_diag = B_coo.row, B_coo.col
            data.append(self.A_diagonal_blocks[i_S, row_diag, col_diag].flatten())
            rows.append(i_A * self.diagonal_blocksize + row_diag)
            cols.append(i_A * self.diagonal_blocksize + col_diag)

            if i_A < self.n_diag_blocks - 1:
                B_coo = B[
                    (i_A + 1)
                    * self.diagonal_blocksize : (i_A + 2)
                    * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_lower, col_lower = B_coo.row, B_coo.col
                data.append(
                    self.A_lower_diagonal_blocks[i_S, row_lower, col_lower].flatten()
                )
                rows.append((i_A + 1) * self.diagonal_blocksize + row_lower)
                cols.append(i_A * self.diagonal_blocksize + col_lower)

            if sparsity == "bta":
                B_coo = B[
                    -self.arrowhead_blocksize :,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_arrow, col_arrow = B_coo.row, B_coo.col
                data.append(
                    self.A_arrow_bottom_blocks[i_S, row_arrow, col_arrow].flatten()
                )
                rows.append(self.n_diag_blocks * self.diagonal_blocksize + row_arrow)
                cols.append(i_A * self.diagonal_blocksize + col_arrow)

        if sparsity == "bta":
            if self.rank == 0:
                # only rank 0 contribute the tip of the arrow
                B_coo = B[
                    -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
                ].tocoo()
                row_arrow_tip, col_arrow_tip = B_coo.row, B_coo.col
                data.append(
                    self.A_arrow_tip_block[row_arrow_tip, col_arrow_tip].flatten()
                )
                rows.append(
                    self.n_diag_blocks * self.diagonal_blocksize + row_arrow_tip
                )
                cols.append(
                    self.n_diag_blocks * self.diagonal_blocksize + col_arrow_tip
                )

        data = xp.concatenate(data)
        rows = xp.concatenate(rows)
        cols = xp.concatenate(cols)

        # Need to communicate to agregates/Map the local B matrix to all ranks
        # Need to operate on the datas
        l_data = xp.concatenate(allgather(data, comm=self.comm))
        l_rows = xp.concatenate(allgather(rows, comm=self.comm))
        l_cols = xp.concatenate(allgather(cols, comm=self.comm))
        synchronize(comm=self.comm)

        B_out = sp.sparse.coo_matrix((l_data, (l_rows, l_cols)), shape=B.shape).tocsc()

        if symmetrize:
            return B_out + sp.sparse.tril(B_out, k=-1).T
        else:
            return B_out

    def _slice_rhs(
        self,
        rhs: NDArray,
        sparsity: str,
    ) -> NDArray:
        """Slice the right-hand side vector."""
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])

        # rhs is guaranteed to be 2D with shape (n, 1) at this point
        # print(f"Rank {self.rank} rhs.shape: {rhs.shape}, self.dist_rhs.shape: {self.dist_rhs.shape}, start_idx: {start_idx* self.diagonal_blocksize}, end_idx: {end_idx* self.diagonal_blocksize}")
        if self.arrowhead_blocksize > 0:
            self.dist_rhs[: -self.arrowhead_blocksize] = rhs[
                start_idx * self.diagonal_blocksize : end_idx * self.diagonal_blocksize
            ]
        else:
            self.dist_rhs[:] = rhs[
                start_idx * self.diagonal_blocksize : end_idx * self.diagonal_blocksize
            ]
        if sparsity == "bta":
            self.dist_rhs[-self.arrowhead_blocksize :] = rhs[
                -self.arrowhead_blocksize :, :
            ]

    def _gather_rhs(
        self,
        rhs: xp.ndarray,
        sparsity: str,
    ):
        """Gather the right-hand side vector."""
        # Calculate the start and end indices for the local slice of rhs
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])

        if (
            backend_flags["array_module"] == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not backend_flags["nccl_avail"]
        ):
            if self.arrowhead_blocksize > 0:
                self.dist_rhs[: -self.arrowhead_blocksize].flatten().get(
                    out=self.send_rhs[
                        self.remainders[self.rank] * self.diagonal_blocksize :
                    ]
                )
            else:
                self.dist_rhs.flatten().get(
                    out=self.send_rhs[
                        self.remainders[self.rank] * self.diagonal_blocksize :
                    ]
                )
        else:
            if self.arrowhead_blocksize > 0:
                self.send_rhs[
                    self.remainders[self.rank] * self.diagonal_blocksize :
                ] = self.dist_rhs[: -self.arrowhead_blocksize].flatten()
            else:
                self.send_rhs[
                    self.remainders[self.rank] * self.diagonal_blocksize :
                ] = self.dist_rhs.flatten()

        synchronize(comm=self.comm)
        self.comm.Allgather(
            sendbuf=self.send_rhs,
            recvbuf=self.recv_rhs,
        )
        synchronize(comm=self.comm)

        # This part needs a major re-work as the multiple RHS are handled one column at a time
        # but Serinv can handle multiple RHS in one go. The problem is linked to DALIA internal
        # representation (2nd dimension of rhs when only 1 rhs) and the collectives operations.
        if (
            backend_flags["array_module"] == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not backend_flags["nccl_avail"]
        ):
            for i in range(self.comm_size):
                start_idx = int(xp.cumsum(n_idx)[i])
                end_idx = int(xp.cumsum(n_idx)[i + 1])

                # Extract data from pinned host memory
                host_data = self.recv_rhs[
                    (i * self.max_n_locals + self.remainders[i])
                    * self.diagonal_blocksize : (i + 1)
                    * self.max_n_locals
                    * self.diagonal_blocksize
                ]

                # Since we're processing one column at a time, rhs.shape[1] is always 1
                # Reshape the 1D host data to 2D (n_rows, 1)
                n_rows = (
                    end_idx * self.diagonal_blocksize
                    - start_idx * self.diagonal_blocksize
                )
                host_data_2d = host_data[:n_rows].reshape(-1, 1)

                # Transfer to device and assign
                # Here we can't use `.set()` because multidimmensional rhs is not contiguous array.
                rhs[
                    start_idx
                    * self.diagonal_blocksize : end_idx
                    * self.diagonal_blocksize
                ] = xp.asarray(host_data_2d)
        else:
            for i in range(self.comm_size):
                start_idx = int(xp.cumsum(n_idx)[i])
                end_idx = int(xp.cumsum(n_idx)[i + 1])

                rhs[
                    start_idx
                    * self.diagonal_blocksize : end_idx
                    * self.diagonal_blocksize
                ] = self.recv_rhs[
                    (i * self.max_n_locals + self.remainders[i])
                    * self.diagonal_blocksize : (i + 1)
                    * self.max_n_locals
                    * self.diagonal_blocksize
                ].reshape(
                    -1, rhs.shape[1]
                )
        if sparsity == "bta":
            # Map the arrow-tip of self.dist_rhs to the global rhs
            rhs[-self.arrowhead_blocksize :] = self.dist_rhs[
                -self.arrowhead_blocksize :
            ]

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        bytes_pobtars: int = (
            self.buffer.nbytes
            + self.reduced_system["A_diagonal_blocks"].nbytes
            + self.reduced_system["A_lower_diagonal_blocks"].nbytes
            + self.reduced_system["A_lower_arrow_blocks"].nbytes
            + self.reduced_system["A_arrow_tip_block"].nbytes
            + self.reduced_system["B"].nbytes
        )
        bytes_local_system: int = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        self.total_bytes += bytes_pobtars + bytes_local_system

        return self.total_bytes
