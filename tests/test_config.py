# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Dict

RTOLS: Dict[str, float] = {
    "strict": 1e-14,
    "relaxed": 1e-10,
}

ATOLS: Dict[str, float] = {
    "strict": 1e-16,
    "relaxed": 1e-12,
}

RANDOM_SEED = 63

__all__ = [
    "RTOLS",
    "ATOLS",
    "RANDOM_SEED",
]
