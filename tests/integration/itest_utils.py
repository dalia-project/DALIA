import os
import runpy
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from dalia import backend_flags, comm_rank

GENERATED_DIR = Path(__file__).resolve().parent / "_generated"


def generate_data(name: str, generator: Callable[[Path], None]) -> Path:
    """Generate the data of a test from scratch in a scratch directory.

    The tests never depend on data files of the repository. Rank 0 calls
    `generator(data_dir)`, the other ranks wait for it.

    Returns
    -------
    Path
        Directory containing the generated inputs and reference outputs.
    """
    data_dir = GENERATED_DIR / name

    if comm_rank == 0:
        shutil.rmtree(data_dir, ignore_errors=True)
        data_dir.mkdir(parents=True)
        generator(data_dir)

    if backend_flags["mpi_avail"]:
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()

    return data_dir


def generate_example_data(example_path: Path, seed: int = 0) -> Path:
    """Generate the data of an example through its own `generate_data.py`.

    Parameters
    ----------
    example_path : Path
        Directory of the example, containing `generate_data.py`.
    seed : int
        Seed of the global NumPy generator, for the scripts not seeding it.
    """

    def run_script(data_dir: Path) -> None:
        # The scripts write either next to themselves or in the working directory
        script = shutil.copy(Path(example_path) / "generate_data.py", data_dir)
        cwd = os.getcwd()
        os.chdir(data_dir)
        try:
            np.random.seed(seed)
            runpy.run_path(script, run_name="__main__")
        finally:
            os.chdir(cwd)

    return generate_data(Path(example_path).name, run_script)
