"""Archive validation without a network request."""

import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

EXTRACT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "download_data.py")
)["extract_dataset"]


def test_extract_only_academic_tree(tmp_path):
    archive = tmp_path / "fixture.zip"
    root = tmp_path / "dataset"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("PlantVillage/Test_healthy/leaf.png", b"image")
        zip_file.writestr("plantvillage/PlantVillage/Test_healthy/leaf.png", b"mirror")
    EXTRACT(archive, root)
    assert list(root.rglob("*.png")) == [root / "Test_healthy" / "leaf.png"]
    assert (root / "Test_healthy" / "leaf.png").read_bytes() == b"image"
    with pytest.raises(FileExistsError):
        EXTRACT(archive, root)


def test_extract_rejects_traversal(tmp_path):
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("PlantVillage/../outside.png", b"unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        EXTRACT(archive, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "outside.png").exists()
