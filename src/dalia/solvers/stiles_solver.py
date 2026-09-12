# Copyright 2024-2025 DALIA authors. All rights reserved.
import gc
import math
import os
import time
from warnings import warn

import numpy as np
import psutil
import scipy.sparse as sparse_host

from dalia import NDArray, backend_flags, comm_size, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.utils import get_host, print_msg, synchronize_gpu

# Let the worker threads float inside the process's CPU set. With its default
# pinning policy, libstiles binds a single-threaded process running inside a
# restricted CPU set to the first allowed core,
# so several such processes on one node end up sharing that core.
os.environ.setdefault("STILES_BIND", "0")

try:
    from sTiles import sTiles as STilesHandle
    from sTiles import sTilesError
except ImportError as e:
    warn(f"The sTiles package is required to use the STilesSolver: {e}")


class STilesSolver(Solver):
    """Sparse Cholesky solver built on the sTiles library (CPU, multithreaded)."""

    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initialize the sTiles solver.

        Parameters
        ----------
        config : SolverConfig
            Solver configuration. Relevant fields: `stiles_threads`,
            `stiles_tile_size`, `stiles_tile_mode`.
        **kwargs
            Ignored, accepted for interface compatibility with the other
            solvers.
        """
        super().__init__(config)

        if config.stiles_threads is not None:
            self.n_threads: int = config.stiles_threads
        else:
            # Share the physical cores of a node between the DALIA processes
            # placed on it: one process per node gets all the cores.
            n_cores = psutil.cpu_count(logical=False) or 1
            processes_per_node = max(math.ceil(comm_size / _number_of_nodes()), 1)
            self.n_threads = max(1, n_cores // processes_per_node)
        self.tile_size: int = config.stiles_tile_size
        self.tile_mode: str = config.stiles_tile_mode

        # --- sTiles handle and the pattern it has been analysed on
        self._handle = None
        self.n: int = None
        # Sorted int64 keys `row * n + col` of the analysed lower-triangle pattern,
        # and the same pattern as int32 (row, col) arrays for `update()`.
        self._pattern_keys: np.ndarray = None
        self._pattern_rows: np.ndarray = None
        self._pattern_cols: np.ndarray = None
        # Value buffer in the order of the analysed pattern (reused every call).
        self._values: np.ndarray = None
        # sparsity key -> (keys of the last input pattern, positions in `_values`)
        self._value_maps: dict = {}
        self._factorized: bool = False
        self._selinv_done: bool = False

        # Solver Metrics
        self.n_analyses: int = 0
        self.t_analyze = 0.0
        self.t_factorize = 0.0
        self.t_solve = 0.0

    # ------------------------------------------------------------------ API
    def factorize(
        self, A: sp.sparse.spmatrix, sparsity: str | None = None, **kwargs
    ) -> None:
        """Compute the Cholesky decomposition of a symmetric positive definite matrix.

        Parameters
        ----------
        A : sp.sparse.spmatrix or NDArray
            The matrix to decompose. Only its lower triangle is read.
        sparsity : str, optional
            Name of the sparsity pattern ('bt', 'bta', ...). Used as a cache
            key for the values mapping; any string (or None) works.
        **kwargs
            Ignored, accepted for interface compatibility with the other
            solvers.

        Raises
        ------
        ValueError
            If the matrix is not positive definite.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        n, keys, data = _lower_triangle_keys(_to_host_csr(A))
        positions = self._positions_in_pattern(sparsity, n, keys)

        self._values[:] = 0.0
        self._values[positions] = data
        self._numeric_factorization()

        self._factorized = True
        self._selinv_done = False

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_factorize += toc - tic

    def solve(self, rhs: NDArray, sparsity: str | None = None, **kwargs) -> NDArray:
        """Solve A x = rhs using the Cholesky factor.

        Parameters
        ----------
        rhs : NDArray
            Right-hand side(s), shape (n,) or (n, n_rhs).
        sparsity : str, optional
            Ignored, kept for interface compatibility with the other solvers.
        **kwargs
            Ignored, accepted for interface compatibility with the other
            solvers.

        Returns
        -------
        NDArray
            Solution with the shape and array module of `rhs`.

        Raises
        ------
        ValueError
            If no matrix has been factorized.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        self._require_factorized()
        b = np.ascontiguousarray(get_host(rhs), dtype=np.float64)
        x = self._handle.solve(b.reshape(self.n, -1))
        x = xp.asarray(x.reshape(rhs.shape))

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return x

    def logdet(self, sparsity: str | None = None, **kwargs) -> float:
        """Compute the log determinant of the factorized matrix.

        Returns
        -------
        float
            The log determinant, 2 * sum(log(diag(L))).

        Raises
        ------
        ValueError
            If no matrix has been factorized or the value is NaN.
        """
        self._require_factorized()
        logdet = self._handle.logdet
        if np.isnan(logdet):
            raise ValueError("Logdet is NaN. Check the input matrix.")
        return logdet

    def selected_inversion(self, sparsity: str | None = None, **kwargs) -> None:
        """Compute the selected inverse on the pattern of the Cholesky factor.

        The entries are retrieved with `_structured_to_spmatrix`.

        Raises
        ------
        ValueError
            If no matrix has been factorized.
        """
        self._require_factorized()
        self._handle.selinv()
        self._selinv_done = True

    def _structured_to_spmatrix(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str | None = None,
        symmetrize: bool = True,
        **kwargs,
    ) -> sp.sparse.spmatrix:
        """Extract the selected inverse at the non-zero positions of `A`.

        Parameters
        ----------
        A : sp.sparse.spmatrix
            Matrix whose sparsity pattern selects the entries to extract
            (e.g. the identity for the marginal variances).
        sparsity : str, optional
            Ignored, kept for interface compatibility with the other solvers.
        symmetrize : bool, optional
            Ignored, kept for interface compatibility with the other solvers:
            the selected inverse is symmetric and both triangles are read
            directly.

        Returns
        -------
        sp.sparse.spmatrix
            `csc` matrix, with the same array module as DALIA, holding the
            selected inverse on the pattern of `A`.

        Raises
        ------
        ValueError
            If the selected inversion has not been computed.
        """
        if not self._selinv_done:
            raise ValueError("Selected inversion not computed")

        pattern = _to_host_csr(A)
        pattern.sort_indices()
        indptr, indices = pattern.indptr, pattern.indices
        data = np.empty(pattern.nnz, dtype=np.float64)
        for i in range(pattern.shape[0]):
            start, stop = indptr[i], indptr[i + 1]
            if stop > start:
                data[start:stop] = self._handle.selinv_row(i, indices[start:stop])

        out = sparse_host.csr_matrix((data, indices, indptr), shape=pattern.shape)
        return sp.sparse.csc_matrix(out)

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver.

        Returns
        -------
        int
            Number of bytes: factor and selected inverse held by the library,
            plus the values buffer and pattern arrays.
        """
        if self._handle is None:
            return 0
        itemsize = np.dtype(np.float64).itemsize
        # Factor L and its selected inverse (same pattern), values buffer,
        # pattern keys and the int32 (row, col) arrays owned by the library.
        return (
            2 * self._handle.nnz_factor * itemsize
            + self._values.nbytes
            + self._pattern_keys.nbytes
            + 2 * (self._pattern_rows.nbytes + self._pattern_cols.nbytes)
        )

    def close(self) -> None:
        """Release the sTiles handle.

        Only one handle may be alive per process: close a solver before
        creating another one in the same process.
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._factorized = False
        self._selinv_done = False

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - never raise from a finalizer
            pass

    # ------------------------------------------------------------ internals
    def _require_factorized(self) -> None:
        """Check that a factorization is available.

        Raises
        ------
        ValueError
            If no matrix has been factorized.
        """
        if not self._factorized:
            raise ValueError("Matrix factorization not computed")

    def _positions_in_pattern(
        self, sparsity: str | None, n: int, keys: np.ndarray
    ) -> np.ndarray:
        """Locate the entries of an input matrix in the values buffer.

        Re-analyses on the union pattern when `keys` is not contained in the
        current pattern. The result is cached per `sparsity` key and validated
        against `keys` on every call.

        Parameters
        ----------
        sparsity : str or None
            Cache key naming the sparsity pattern of the input matrix.
        n : int
            Dimension of the matrix.
        keys : np.ndarray
            Linear keys `row * n + col` of the lower-triangle entries of the
            input matrix, in its storage order.

        Returns
        -------
        np.ndarray
            Position in `_values` of each entry of `keys`.
        """
        cached = self._value_maps.get(sparsity)
        if (
            cached is not None
            and cached[0].size == keys.size
            and np.array_equal(cached[0], keys)
        ):
            return cached[1]

        if self._pattern_keys is None or self.n != n:
            self._analyze(n, np.unique(keys))
        else:
            positions = np.searchsorted(self._pattern_keys, keys)
            positions[positions == self._pattern_keys.size] = 0
            if not np.array_equal(self._pattern_keys[positions], keys):
                self._analyze(n, np.union1d(self._pattern_keys, keys))

        positions = np.searchsorted(self._pattern_keys, keys)
        self._value_maps[sparsity] = (keys, positions)
        return positions

    def _numeric_factorization(self) -> None:
        """Run the numeric Cholesky of the values in `_values`.

        The symbolic analysis of the handle is reused.

        Raises
        ------
        ValueError
            If the matrix is not positive definite.
        """
        Q = sparse_host.coo_matrix(
            (self._values, (self._pattern_rows, self._pattern_cols)),
            shape=(self.n, self.n),
        )
        try:
            self._handle.update(Q)
        except sTilesError as e:
            self._factorized = False
            raise ValueError(
                f"The matrix does not appear to be positive definite ({e})"
            ) from e

    def _analyze(self, n: int, pattern_keys: np.ndarray) -> None:
        """Run the symbolic analysis of sTiles on a lower-triangle pattern.

        Replaces the current handle and resets the cached value mappings.

        Parameters
        ----------
        n : int
            Dimension of the matrix.
        pattern_keys : np.ndarray
            Sorted linear keys `row * n + col` of the pattern to analyse.
        """
        tic = time.perf_counter()

        self.close()
        self._value_maps = {}
        self.n = n
        self._pattern_keys = pattern_keys
        self._values = np.zeros(pattern_keys.size, dtype=np.float64)

        self._pattern_rows = (pattern_keys // n).astype(np.int32)
        self._pattern_cols = (pattern_keys % n).astype(np.int32)
        pattern = sparse_host.coo_matrix(
            (np.ones(pattern_keys.size), (self._pattern_rows, self._pattern_cols)),
            shape=(n, n),
        )
        self._handle = self._open_handle(pattern)
        self.n_analyses += 1

        toc = time.perf_counter()
        self.t_analyze += toc - tic
        print_msg(
            f"sTiles symbolic analysis #{self.n_analyses}: n={n}, "
            f"nnz(L)={self._handle.nnz_factor}, {toc - tic:.2f} s, "
            f"{self.n_threads} threads"
        )

    def _open_handle(self, pattern: sparse_host.coo_matrix):
        """Create the sTiles handle (symbolic phase only).

        Parameters
        ----------
        pattern : scipy.sparse.coo_matrix
            Lower-triangle sparsity pattern to analyse.

        Returns
        -------
        sTiles.sTiles
            The analysed, not yet factorized, handle.
        """
        kwargs = {
            "cores": self.n_threads,
            "mode": self.tile_mode,
            "tile_size": self.tile_size,
            "inverse": True,
        }
        try:
            return STilesHandle.analyze(pattern, **kwargs)
        except sTilesError:
            # Another (unreferenced) solver still holds the process-wide handle.
            gc.collect()
            return STilesHandle.analyze(pattern, **kwargs)


# ------------------------------------------------------------- host helpers
def _number_of_nodes() -> int:
    """Count the distinct nodes the MPI processes run on (1 without MPI).

    Returns
    -------
    int
        Number of distinct host names across `MPI.COMM_WORLD`.
    """
    if not backend_flags["mpi_avail"]:
        return 1
    from mpi4py import MPI

    return len(set(MPI.COMM_WORLD.allgather(MPI.Get_processor_name())))


def _to_host_csr(A) -> sparse_host.csr_matrix:
    """Convert a matrix to a host CSR matrix.

    Parameters
    ----------
    A : sparse matrix or ndarray
        Square matrix on the host or on the device, sparse or dense.

    Returns
    -------
    scipy.sparse.csr_matrix
        The matrix in host CSR format.
    """
    if hasattr(A, "get"):  # cupy ndarray or cupyx sparse matrix
        A = A.get()
    return sparse_host.csr_matrix(A)


def _lower_triangle_keys(A: sparse_host.csr_matrix):
    """Extract the lower triangle of `A` as linear keys and values.

    The keys follow the storage order of `A` (not sorted): for a given
    construction path of the matrix, this order is stable across calls.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Square matrix.

    Returns
    -------
    n : int
        Dimension of the matrix.
    keys : np.ndarray
        Linear keys `row * n + col` of the lower-triangle entries.
    data : np.ndarray
        Values of those entries, as contiguous float64.
    """
    n = A.shape[0]
    lower = sparse_host.tril(A, format="coo")
    keys = lower.row.astype(np.int64) * n + lower.col
    return n, keys, np.ascontiguousarray(lower.data, dtype=np.float64)
