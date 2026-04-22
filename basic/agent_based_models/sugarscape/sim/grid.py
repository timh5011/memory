"""Sugar landscape initialization helpers for Sugarscape."""

import numpy as np


def make_sugar_landscape(width: int, height: int) -> np.ndarray:
    """
    Create a 50x50 sugar-max grid with two Gaussian peaks at (15,15) and (35,35),
    sigma=10, scaled so peak cell has sugar_max=4, rounded and clipped to [0,4].
    """
    xs = np.arange(width)
    ys = np.arange(height)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # shape (width, height)

    sigma = 10.0
    peak1 = np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / (2 * sigma ** 2))
    peak2 = np.exp(-((xx - 35) ** 2 + (yy - 35) ** 2) / (2 * sigma ** 2))

    raw = peak1 + peak2  # peaks have value 1.0 each; combined max is 2.0 at each center
    # Scale so the maximum value maps to 4
    scaled = raw / raw.max() * 4.0
    sugar_max = np.round(scaled).astype(int)
    sugar_max = np.clip(sugar_max, 0, 4)
    return sugar_max
