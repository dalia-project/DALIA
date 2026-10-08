# Copyright 2024-2025 DALIA authors. All rights reserved.

import os
from typing import Any, TypeAlias, TypeVar
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from dalia.__about__ import __version__

backend_flags = {
    "array_module": None,
    "cupy_avail": False,
    "mpi_avail": False,
    "mpi_cuda_aware": False,
    "nccl_avail": False,
}

# Allows user to specify the array module via an environment variable.
backend_flags["array_module"] = os.environ.get("ARRAY_MODULE")

if backend_flags["array_module"] is not None:
    if backend_flags["array_module"] == "numpy":
        import numpy as xp
        import scipy as sp

        xp_host = xp

    elif backend_flags["array_module"] == "cupy":
        try:
            import cupy as xp
            import cupyx.scipy as sp
            import numpy as xp_host

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except (ImportError, ImportWarning, ModuleNotFoundError) as e:
            warn(
                f"'CuPy' backend selected but unavailable: defaulting to 'NumPy'. ({e})"
            )
            import numpy as xp
            import scipy as sp

            xp_host = xp
    else:
        warn(
            f"Unrecognized ARRAY_MODULE '{backend_flags['array_module']}', defaulting to 'NumPy'."
        )

        import numpy as xp
        import scipy as sp

        xp_host = xp
else:
    # If the user does not specify the array module, prioritize numpy.
    # LOG: warn("No `ARRAY_MODULE` specified, DALIA.core defaulting to 'NumPy'.")
    import numpy as xp
    import scipy as sp

    xp_host = xp

# In any case, check if CuPy is available.
try:
    import cupy

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cupy.abs(1)

    backend_flags["cupy_avail"] = True
except (ImportError, ImportWarning, ModuleNotFoundError) as e:
    ...
    # LOG: warn(f"No 'CuPy' backend detected. ({e})")


try:
    # Check if mpi4py is available
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.size

    # Create a small GPU array
    array = np.array([comm_rank], dtype=np.float32)

    # Perform an MPI operation to check working
    if comm_size > 1:
        if comm_rank == 0:
            comm.Send([array, MPI.FLOAT], dest=1)
        elif comm_rank == 1:
            comm.Recv([array, MPI.FLOAT], source=0)

    backend_flags["mpi_avail"] = True

    if backend_flags["cupy_avail"] and os.environ.get("MPI_CUDA_AWARE", "0") == "1":
        # If CuPy is available and CUDA-aware MPI is requested, check if it works.
        try:
            cupy_array = cupy.array([comm_rank], dtype=cupy.float32)
            if comm_size > 1:
                if comm_rank == 0:
                    comm.Send([cupy_array.data.ptr, MPI.FLOAT], dest=1)
                elif comm_rank == 1:
                    comm.Recv([cupy_array.data.ptr, MPI.FLOAT], source=0)
                # Only set to True if MPI communication with GPU data succeeded
                backend_flags["mpi_cuda_aware"] = True
            else:
                # For single process, we can't test MPI communication but we can test GPU memory access
                # Just accessing the GPU pointer suggests CUDA-aware MPI support is available
                _ = cupy_array.data.ptr  # This will fail if CUDA context is broken
                backend_flags["mpi_cuda_aware"] = True
        except Exception as e:
            warn(f"CUDA-aware MPI test failed: {e}, CUDA-aware MPI will be disabled.")

    if backend_flags["cupy_avail"] and os.environ.get("USE_NCCL", "0") == "1":
        # If CuPy is available and NCCL is requested, check if NCCL is available.
        try:
            # Check if NCCL is available and functional
            from cupy.cuda import nccl

            nccl_id = nccl.get_unique_id()
            backend_flags["nccl_avail"] = True
        except (ImportError, ImportWarning, ModuleNotFoundError) as e:
            warn(
                f"'NCCL' backend requested but unavailable, NCCL will not be used. ({e})"
            )
        except (RuntimeError, OSError) as e:
            warn(f"NCCL test failed: {e}, NCCL will not be used.")

except (ImportError, ImportWarning, ModuleNotFoundError) as e:
    # LOG: warn(f"No 'MPI' backend detected. ({e})")

    comm_rank = 0
    comm_size = 1


# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]


__all__ = [
    "__version__",
    "xp",
    "xp_host",
    "sp",
    "ArrayLike",
    "NDArray",
    "comm_rank",
    "comm_size",
    "backend_flags",
]
