"""Academic HOG and globally normalized RGB histogram features."""

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

FEATURE_COUNT = 1860
FEATURE_SCHEMA = "hog9-16x16-2x2_rgb32-v1"


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Produce 1,764 HOG and 96 RGB histogram features."""
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.shape != (128, 128, 3)
        or rgb.dtype != np.uint8
    ):
        raise ValueError("Features require a 128 × 128 × 3 uint8 RGB array.")
    gradients = hog(
        rgb2gray(rgb),
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    histogram = np.concatenate(
        [
            np.bincount((rgb[..., channel] // 8).ravel(), minlength=32)
            for channel in range(3)
        ]
    )
    return np.concatenate((gradients, histogram / histogram.sum())).astype(np.float32)
