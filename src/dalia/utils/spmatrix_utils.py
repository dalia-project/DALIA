from dalia import backend_flags, sp, xp


def bdiag_tiling(
    sparse_blocks: list[sp.sparse.spmatrix],
) -> sp.sparse.spmatrix:
    """Tile in a block-diagonal fashion the given sparse blocks.
    This is equivalent to the scipy.sparse.block_diag function but is needed
    as the latter is not supported in cupyx.scipy.sparse.

    return a COO matrix
    """
    # Get the shape of the blocks
    block_shapes = [block.shape for block in sparse_blocks]

    out_data = []
    out_row = []
    out_col = []

    for i, block in enumerate(sparse_blocks):
        coo_block = block.tocoo()

        out_data.append(coo_block.data)
        out_row.append(
            coo_block.row + sum([block_shape[0] for block_shape in block_shapes[:i]])
        )
        out_col.append(
            coo_block.col + sum([block_shape[1] for block_shape in block_shapes[:i]])
        )

    out_data = xp.concatenate(out_data)
    out_row = xp.concatenate(out_row)
    out_col = xp.concatenate(out_col)

    out = sp.sparse.coo_matrix(
        (out_data, (out_row, out_col)),
        shape=(
            sum([block_shape[0] for block_shape in block_shapes]),
            sum([block_shape[1] for block_shape in block_shapes]),
        ),
    )

    return out


def extract_diagonal(
    a: sp.sparse.spmatrix,
) -> xp.ndarray:
    """Extract the diagonal of a sparse matrix."""

    if a.shape[0] != a.shape[1]:
        raise ValueError("The input matrix must be square.")

    # If scipy.sparse or xp.ndarray .diagonal() exists
    if not backend_flags["cupy_avail"] or isinstance(a, xp.ndarray):
        diagonal = a.diagonal()
    else:
        a = sp.sparse.coo_matrix(a)
        diagonal_mask = a.row == a.col
        return xp.array(a.data[diagonal_mask])

    return diagonal


def memory_footprint(
    sparse_matrix: sp.sparse.spmatrix,
) -> int:
    """Calculate the memory footprint of a sparse matrix in bytes."""
    if not isinstance(sparse_matrix, sp.sparse.spmatrix):
        raise ValueError("Input must be a sparse matrix.")

    A = sparse_matrix.tocsc()

    data_size = A.data.nbytes
    indices_size = A.indices.nbytes
    indptr_size = A.indptr.nbytes

    total_memory_bytes = data_size + indices_size + indptr_size
    total_memory_gb = total_memory_bytes / (1024**3)

    print(f"Total memory footprint of Q_prior: {total_memory_gb:.6f} GB")
