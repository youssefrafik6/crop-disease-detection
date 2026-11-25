"""Small, deterministic leaf-like test images."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crop_disease.config import Config
from crop_disease.train import fit_run


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic-leaves")
    rng = np.random.default_rng(42)
    for label, name, color, count in (
        (0, "Tomato_healthy", (35, 155, 15), 24),
        (1, "Tomato_Early_blight", (150, 75, 35), 40),
    ):
        folder = root / name
        folder.mkdir()
        for index in range(count):
            noise = rng.integers(-20, 21, size=(32, 38, 3))
            rgb = np.clip(noise + color, 0, 255).astype(np.uint8)
            if label:
                rgb[6:13, 12:18] = (61, 29, 17)
            Image.fromarray(rgb).save(folder / f"{index:02d}.png")
    return root


@pytest.fixture(scope="session")
def trained_run(synthetic_data, tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-artifacts")
    run, metrics, figures = (
        directory / name for name in ("models", "metrics", "figures")
    )
    fit_run(
        Config(cnn_epochs=1),
        synthetic_data,
        run,
        metrics,
        figures,
        deep="cnn",
        device="cpu",
    )
    return run, metrics, figures
