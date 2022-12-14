"""Dataset generator implementation"""
from typing import Tuple

import gin
import numpy as np

from src.constants import FEATURES_NUMBER


@gin.configurable()
def get_dataset(
        n_samples: int,
        train_test_split: float,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.random(size=(n_samples, FEATURES_NUMBER))
    y = 10 * x[:, 0] * (x[:, 0] + 5 * x[:, 1]) - 1/100 * x[:, 2]

    split = int(n_samples * train_test_split)
    return x[:split, :], y[:split], x[split:, :], y[split:]
