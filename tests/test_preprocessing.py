from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from crop_disease.preprocessing import MAX_BYTES, InvalidImage, decode_rgb, preprocess


@pytest.mark.parametrize(
    "mode,value,expected",
    [
        ("L", 125, [125, 125, 125]),
        ("RGBA", (10, 20, 30, 0), [10, 20, 30]),
    ],
)
def test_converts_and_resizes(mode, value, expected, tmp_path):
    image = Image.new(mode, (19, 25), value)
    destination = tmp_path / "sample.png"
    image.save(destination)
    array = preprocess(destination)
    assert array.shape == (128, 128, 3)
    assert array.dtype == np.uint8
    assert array[0, 0].tolist() == expected
    np.testing.assert_array_equal(array, preprocess(destination.read_bytes()))


@pytest.mark.parametrize("value", [b"", b"invalid", b"0" * (MAX_BYTES + 1)])
def test_invalid_image(value):
    with pytest.raises(InvalidImage):
        decode_rgb(value)


def test_rejects_unsupported_format():
    stream = BytesIO()
    Image.new("RGB", (5, 5)).save(stream, format="GIF")
    with pytest.raises(InvalidImage):
        decode_rgb(stream.getvalue())
