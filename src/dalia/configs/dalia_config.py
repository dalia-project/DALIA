# Copyright 2024-2025 DALIA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, PositiveInt


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["dense", "scipy", "serinv", "stiles"] = "scipy"

    min_processes: PositiveInt = 1

    # --- sTiles solver options (type="stiles") ---
    # Threads per factorization. Default: physical cores of the node divided by
    # the number of DALIA processes.
    stiles_threads: Optional[PositiveInt] = None
    # Tile size in elements (-1 lets sTiles choose).
    stiles_tile_size: int = 32
    # Tile storage mode ("auto" lets sTiles pick dense/semisparse/sparse tiles).
    stiles_tile_mode: Literal["auto", "dense", "semisparse", "sparse"] = "sparse"


class BFGSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iter: PositiveInt = 100
    jac: bool = True

    maxcor: PositiveInt = (
        10  # maximum number of past gradient vectors to store -> good default: dim(theta)
    )
    maxls: PositiveInt = 20  # maximum number of line search iterations

    gtol: float = 1e-1
    # c1: float = 1e-4  # only relevant for BFGS not for L-BFGS-B
    # c2: float = 0.9  # only relevant for BFGS not for L-BFGS-B
    disp: bool = False


class DaliaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ------------------------------------------------
    solver: SolverConfig = SolverConfig()
    minimize: BFGSConfig = BFGSConfig()

    # exit BFGS early if the reduction in the objective function is less than f_reduction_tol after f_reduction_lag iterations
    f_reduction_lag: int = 3
    f_reduction_tol: float = 1e-4

    # exit BFGS early if the change in theta is less than theta_reduction_tol after theta_reduction_lag iterations
    theta_reduction_lag: int = 3
    theta_reduction_tol: float = 1e-4

    inner_iteration_max_iter: PositiveInt = 50
    eps_inner_iteration: float = 1e-3
    eps_gradient_f: float = 1e-3
    eps_hessian_f: float = 5 * 1e-3

    # --- Directory paths ------------------------------------------------------
    simulation_dir: Path = Path("./dalia/")
    output_dir: Path = Path.joinpath(simulation_dir, "output/")

    # --- Verbosity level ------------------------------------------------------
    verbosity: int = 0  # 0: minimal, 1: more info


def parse_config(config: dict | str) -> DaliaConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    return DaliaConfig(**config)
