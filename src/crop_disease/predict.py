"""Versioned artifact loading and the shared prediction backend."""

from pathlib import Path

import numpy as np
import skops.io as sio

from .data_loader import LABELS
from .features import FEATURE_SCHEMA, extract_features
from .models import DESCRIPTIONS
from .preprocessing import preprocess
from .utils import file_digest, read_json, safe_name, versions

ARTIFACT_SCHEMA = 2
TRUSTED_SKOPS_TYPES = {
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.model_selection._split.StratifiedKFold",
}


def model_path(run_dir: Path, name: str, metadata: dict) -> Path:
    extension = "safetensors" if metadata["kind"] == "torch" else "skops"
    return run_dir / f"{safe_name(name)}.{extension}"


def load_model(run_dir: Path, name: str | None = None):
    """Load a supported artifact from a trusted local training run."""
    run = read_json(run_dir / "run.json")
    if run["schema"] != ARTIFACT_SCHEMA or run["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("Unsupported artifact schema. Regenerate the model.")
    name = name or run["selected_model"]
    if name not in run["models"] or name not in DESCRIPTIONS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {', '.join(run['models'])}."
        )
    metadata = run["models"][name]
    path = model_path(run_dir, name, metadata)
    if file_digest(path) != metadata["sha256"]:
        raise ValueError("Artifact checksum mismatch.")
    if metadata["kind"] == "torch":
        from .cnn import restore_model

        model = restore_model(path, name)
    else:
        installed = versions()
        if installed.get("scikit-learn") != run["versions"]["scikit-learn"]:
            raise ValueError(
                "Install the recorded scikit-learn version or retrain the artifact."
            )
        unknown = set(sio.get_untrusted_types(file=path))
        allowed = TRUSTED_SKOPS_TYPES if name == "linear_svm" else set()
        if unknown - allowed:
            raise ValueError(
                f"Artifact contains unsupported types: {sorted(unknown - allowed)}"
            )
        model = sio.load(path, trusted=list(unknown))
    return name, model, run["config"]["image_size"]


def predict_image(loaded, source: Path | bytes) -> dict:
    """Return predicted label and the model's estimated probability."""
    name, model, size = loaded
    rgb = preprocess(source, size)
    if name in {"cnn", "resnet18"}:
        from .cnn import infer_rgb

        probabilities = infer_rgb(model, rgb, name)
        label = int(np.argmax(probabilities))
    else:
        vector = extract_features(rgb)[None, :]
        label = int(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0]
    return {
        "label": LABELS[label],
        "estimated_probability": (
            None if name == "baseline" else float(probabilities[label])
        ),
    }
