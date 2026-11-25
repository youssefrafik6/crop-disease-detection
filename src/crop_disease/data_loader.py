"""PlantVillage discovery, duplicate audit and shared split manifests."""

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import Config
from .preprocessing import InvalidImage, decode_rgb

LOG = logging.getLogger(__name__)
LABELS = {0: "Healthy", 1: "Diseased"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    digest: str
    dhash: str


def rgb_digest(image: Image.Image) -> str:
    """Fingerprint decoded pixels and dimensions."""
    payload = f"{image.width}x{image.height}:".encode() + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def scan_dataset(root: Path) -> tuple[list[Sample], dict]:
    """Validate files and keep one image per exact decoded RGB digest."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory missing: {root}. See data/README.md."
        )
    paths = sorted(root.glob("*/*"))
    groups: dict[str, list[Sample]] = {}
    audit = {"unsupported": 0, "invalid": 0, "duplicates": 0, "conflicting_groups": 0}
    for file in paths:
        if (
            not file.is_file()
            or file.is_symlink()
            or file.parent.is_symlink()
            or file.name.startswith(".")
        ):
            continue
        if file.suffix.lower() not in IMAGE_SUFFIXES:
            audit["unsupported"] += 1
            continue
        path = file.relative_to(root).as_posix()
        try:
            image = decode_rgb(file)
        except InvalidImage as exc:
            audit["invalid"] += 1
            LOG.warning("Skipping %s (%s)", path, exc)
            continue
        digest = rgb_digest(image)
        thumb = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        dhash = np.packbits(thumb[:, 1:] > thumb[:, :-1]).tobytes().hex()
        folder = file.parent.name.lower()
        label = int(folder != "healthy" and not folder.endswith("_healthy"))
        groups.setdefault(digest, []).append(Sample(path, label, digest, dhash))
    samples = []
    audit["valid_before_dedup"] = dict(
        Counter(s.label for g in groups.values() for s in g)
    )
    for group in groups.values():
        if len({s.label for s in group}) > 1:
            audit["conflicting_groups"] += 1
            continue
        audit["duplicates"] += len(group) - 1
        samples.append(group[0])
    if {s.label for s in samples} != {0, 1}:
        raise ValueError("Expected class subfolders with healthy and diseased images.")
    return samples, audit


def partition(samples: list[Sample], cfg: Config) -> dict[str, list[Sample]]:
    """Create stable, stratified train, validation and test partitions."""
    try:
        train, holdout = train_test_split(
            samples,
            test_size=cfg.holdout_fraction,
            stratify=[s.label for s in samples],
            random_state=cfg.seed,
        )
        val, test = train_test_split(
            holdout,
            test_size=cfg.test_share_of_holdout,
            stratify=[s.label for s in holdout],
            random_state=cfg.seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Insufficient samples per class for stratified splitting."
        ) from exc
    splits = {"train": train, "validation": val, "test": test}
    if any({s.label for s in part} != {0, 1} for part in splits.values()):
        raise ValueError("Each partition must contain Healthy and Diseased images.")
    return splits


def near_duplicate_audit(splits: dict[str, list[Sample]]) -> dict:
    """Count cross-split equal 64-bit perceptual hashes as review candidates."""
    hashes: dict[str, Counter] = {}
    for split, samples in splits.items():
        for sample in samples:
            hashes.setdefault(sample.dhash, Counter())[split] += 1
    shared = {digest: counts for digest, counts in hashes.items() if len(counts) > 1}
    return {
        "cross_split_equal_dhash_buckets": len(shared),
        "images_in_those_buckets": sum(
            sum(counts.values()) for counts in shared.values()
        ),
        "example_hashes": sorted(shared)[:10],
    }
