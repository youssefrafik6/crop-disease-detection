"""Bounded RGB decoding shared by training and inference."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

FORMATS = ("JPEG", "PNG", "BMP", "WEBP")
MAX_BYTES = 20 * 1024**2
MAX_PIXELS = 25_000_000
RESAMPLING = Image.Resampling.BICUBIC


class InvalidImage(ValueError):
    """An input cannot be safely decoded as a supported still image."""


def decode_rgb(source: Path | bytes) -> Image.Image:
    """Decode one still image, enforcing size and format limits."""
    try:
        size = len(source) if isinstance(source, bytes) else source.stat().st_size
        if not 0 < size <= MAX_BYTES:
            raise InvalidImage("Image must contain between 1 byte and 20 MiB.")
        stream = BytesIO(source) if isinstance(source, bytes) else source
        with Image.open(stream, formats=FORMATS) as image:
            if image.width * image.height > MAX_PIXELS or getattr(
                image, "is_animated", False
            ):
                raise InvalidImage("Use a still image with at most 25 megapixels.")
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise InvalidImage(
            "Cannot decode image. Use a valid JPEG, PNG, BMP or WebP file."
        ) from exc


def resize_rgb(image: Image.Image, size: int = 128) -> np.ndarray:
    """Return an RGB uint8 array using the original bicubic resize."""
    return np.asarray(image.resize((size, size), RESAMPLING))


def preprocess(source: Path | bytes, size: int = 128) -> np.ndarray:
    """Convert a still image to the training and inference input shape."""
    return resize_rgb(decode_rgb(source), size)
