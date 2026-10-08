# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as host_xp

from dalia import backend_flags, xp

THREADS_PER_BLOCK = 1024

if backend_flags["cupy_avail"]:
    import cupy as cp

    _compute_coo_block_mask_kernel = cp.RawKernel(
        r"""
            extern "C" __global__
            void _compute_coo_block_mask_kernel(
                int *rows,
                int *cols,
                int row_start,
                int row_stop,
                int col_start,
                int col_stop,
                bool *mask,
                int rows_len
            ){
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < rows_len) {
                    mask[tid] = (
                        (rows[tid] >= row_start)
                        && (rows[tid] < row_stop)
                        && (cols[tid] >= col_start)
                        && (cols[tid] < col_stop)
                    );
                }
            }
        """,
        "_compute_coo_block_mask_kernel",
    )


def compute_block_sort_index(
    coo_rows: xp.ndarray, coo_cols: xp.ndarray, block_sizes: xp.ndarray
) -> xp.ndarray:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    Due to the Python for loop around the kernel, this method will
    perform best for larger block sizes (>500).

    Parameters
    ----------
    coo_rows : NDArray
        The row indices of the matrix in coordinate format.
    coo_cols : NDArray
        The column indices of the matrix in coordinate format.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : NDArray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = host_xp.hstack(
        (host_xp.array([0]), host_xp.cumsum(block_sizes)), dtype=host_xp.int32
    )

    sort_index = cp.zeros(len(coo_cols), dtype=cp.int32)
    mask = cp.zeros(len(coo_cols), dtype=cp.bool_)
    coo_rows = coo_rows.astype(cp.int32)
    coo_cols = coo_cols.astype(cp.int32)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0
    for i, j in cp.ndindex(num_blocks, num_blocks):
        _compute_coo_block_mask_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                host_xp.int32(block_offsets[i]),
                host_xp.int32(block_offsets[i + 1]),
                host_xp.int32(block_offsets[j]),
                host_xp.int32(block_offsets[j + 1]),
                mask,
                host_xp.int32(len(coo_cols)),
            ),
        )

        bnnz = cp.sum(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            offset += bnnz

    return sort_index


def compute_block_slice(
    rows: xp.ndarray,
    cols: xp.ndarray,
    block_offsets: xp.ndarray,
    block_row: int,
    block_col: int,
) -> slice:
    """Computes the slice of the block in the data.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix.
    cols : NDArray
        The column indices of the matrix.
    block_offsets : NDArray
        The offsets of the blocks.
    block_row : int
        The block row to compute the slice for.
    block_col : int
        The block column to compute the slice for.

    Returns
    -------
    start : int
        The start index of the block.
    stop : int
        The stop index of the block.

    """
    mask = cp.zeros(rows.shape[0], dtype=cp.bool_)
    row_start, row_stop = host_xp.int32(block_offsets[block_row]), host_xp.int32(
        block_offsets[block_row + 1]
    )
    col_start, col_stop = host_xp.int32(block_offsets[block_col]), host_xp.int32(
        block_offsets[block_col + 1]
    )

    rows = rows.astype(cp.int32)
    cols = cols.astype(cp.int32)

    blocks_per_grid = (rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _compute_coo_block_mask_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            rows,
            cols,
            row_start,
            row_stop,
            col_start,
            col_stop,
            mask,
            host_xp.int32(rows.shape[0]),
        ),
    )
    if cp.sum(mask) == 0:
        # No data in this block, return an empty slice.
        return None, None

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    inds = cp.nonzero(mask)[0]

    # NOTE: this copies back to the host
    # return int(inds[0]), int(inds[-1] + 1)
    return inds
