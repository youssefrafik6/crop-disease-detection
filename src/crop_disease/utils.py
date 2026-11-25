"""Run metadata and artifact integrity helpers."""

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: Path, value, *, readable: bool = False) -> None:
    """Publish a JSON document atomically."""
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            if readable:
                temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def versions() -> dict:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "Pillow",
        "scikit-image",
        "scikit-learn",
        "skops",
        "torch",
        "torchvision",
        "safetensors",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return packages


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError(
            "Run and model names may contain only letters, numbers, hyphens and underscores."
        )
    return name
