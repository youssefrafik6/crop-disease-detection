"""Download the academic archive or extract a local copy."""

import argparse
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease"
MAX_ARCHIVE = 2 * 1024**3


def extract_dataset(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp, ZipFile(archive) as dataset:
        chosen = []
        seen = set()
        for item in dataset.infolist():
            path = PurePosixPath(item.filename)
            if len(path.parts) != 3 or path.parts[0] != "PlantVillage" or item.is_dir():
                continue
            if (
                ".." in path.parts
                or "\\" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member.")
            out = Path(tmp, *path.parts)
            key = str(out).casefold()
            if key in seen:
                raise ValueError("Duplicate archive member.")
            seen.add(key)
            chosen.append((item, out))
        if not chosen or sum(item.file_size for item, _ in chosen) > MAX_ARCHIVE:
            raise ValueError("Archive layout or uncompressed size is unexpected.")
        for item, out in chosen:
            out.parent.mkdir(parents=True, exist_ok=True)
            with dataset.open(item) as source, out.open("xb") as target:
                shutil.copyfileobj(source, target)
        Path(tmp, "PlantVillage").rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing Kaggle ZIP.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/PlantVillage"))
    args = parser.parse_args()
    try:
        if args.archive:
            extract_dataset(args.archive, args.output)
        else:
            with TemporaryDirectory() as tmp:
                archive = Path(tmp, "plantdisease.zip")
                with (
                    urlopen(DATASET_URL, timeout=60) as source,
                    archive.open("wb") as target,
                ):
                    total = 0
                    while block := source.read(1024 * 1024):
                        total += len(block)
                        if total > MAX_ARCHIVE:
                            raise ValueError("Download exceeds 2 GiB.")
                        target.write(block)
                extract_dataset(archive, args.output)
    except (OSError, ValueError, BadZipFile) as exc:
        sys.exit(f"Dataset setup failed: {exc}. See data/README.md.")
    print(f"Dataset extracted to {args.output}")


if __name__ == "__main__":
    main()
