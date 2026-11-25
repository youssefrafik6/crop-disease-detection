import shutil
from pathlib import Path

import pytest

from crop_disease.config import Config
from crop_disease.data_loader import near_duplicate_audit, partition, scan_dataset


def test_scan_and_split_are_deterministic(synthetic_data):
    samples, audit = scan_dataset(synthetic_data)
    assert audit["valid_before_dedup"] == {0: 24, 1: 40}
    splits = partition(samples, Config())
    assert splits == partition(scan_dataset(synthetic_data)[0], Config())
    assert [len(part) for part in splits.values()] == [44, 10, 10]
    seen = set()
    for part in splits.values():
        assert {sample.label for sample in part} == {0, 1}
        digests = {sample.digest for sample in part}
        assert seen.isdisjoint(digests)
        seen.update(digests)
    assert "cross_split_equal_dhash_buckets" in near_duplicate_audit(splits)


def test_invalid_duplicate_and_conflicting_images(tmp_path, synthetic_data):
    root = tmp_path / "data"
    shutil.copytree(synthetic_data, root)
    folder = root / "Tomato_healthy"
    shutil.copyfile(folder / "00.png", folder / "copy.png")
    shutil.copyfile(folder / "01.png", root / "Tomato_Early_blight" / "conflict.png")
    (folder / "broken.jpg").write_bytes(b"not an image")
    (folder / "readme.txt").write_bytes(b"notes")
    samples, audit = scan_dataset(root)
    assert len(samples) == 63
    assert audit["duplicates"] == audit["invalid"] == audit["unsupported"] == 1
    assert audit["conflicting_groups"] == 1


def test_missing_dataset(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Dataset directory missing"):
        scan_dataset(tmp_path / "missing")


def test_insufficient_examples():
    with pytest.raises(ValueError, match="Insufficient"):
        partition([], Config())
