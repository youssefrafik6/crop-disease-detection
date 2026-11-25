import numpy as np
import pytest
from skimage.color import rgb2gray
from skimage.feature import hog

from crop_disease.features import FEATURE_COUNT, extract_features


def test_academic_feature_dimensions_and_consistency():
    image = np.random.default_rng(18).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    vector = extract_features(image)
    assert vector.shape == (FEATURE_COUNT,)
    np.testing.assert_array_equal(vector, extract_features(image.copy()))
    expected_hog = hog(
        rgb2gray(image),
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    pixels = image.astype(np.float32) / 255
    expected_hist = np.concatenate(
        [np.histogram(pixels[..., c], 32, (0, 1))[0] for c in range(3)]
    )
    np.testing.assert_allclose(vector[:1764], expected_hog, rtol=1e-6)
    np.testing.assert_allclose(
        vector[1764:], expected_hist / expected_hist.sum(), rtol=1e-6
    )


@pytest.mark.parametrize("bad", [None, np.zeros((128, 128)), np.zeros((128, 128, 3))])
def test_invalid_feature_input(bad):
    with pytest.raises(ValueError, match="uint8"):
        extract_features(bad)
