# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

N_DIAG_BLOCKS_PER_PROCESS = [
    pytest.param(3, id="n_diag_blocks=3"),
    pytest.param(5, id="n_diag_blocks=5"),
    pytest.param(10, id="n_diag_blocks=10"),
]


@pytest.fixture(params=N_DIAG_BLOCKS_PER_PROCESS, autouse=True)
def n_diag_blocks_per_process(request: pytest.FixtureRequest) -> int:
    return request.param


NON_UNIFORM_PARTITION_SIZES = [
    pytest.param(True, id="non_uniform_partition=True"),
    pytest.param(False, id="non_uniform_partition=False"),
]


@pytest.fixture(params=NON_UNIFORM_PARTITION_SIZES, autouse=True)
def non_uniform_partition(request: pytest.FixtureRequest) -> bool:
    return request.param
