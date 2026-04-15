"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(global_seed: int) -> None:
    """Seed all relevant random number generators."""

    random.seed(global_seed)
    np.random.seed(global_seed)
