# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

N_DIAG_BLOCKS = [
    pytest.param(1, id="n_diag_blocks=1"),
    pytest.param(2, id="n_diag_blocks=2"),
    pytest.param(3, id="n_diag_blocks=3"),
    pytest.param(4, id="n_diag_blocks=4"),
]


@pytest.fixture(params=N_DIAG_BLOCKS, autouse=True)
def n_diag_blocks(request: pytest.FixtureRequest) -> int:
    return request.param
