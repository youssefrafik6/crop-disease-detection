"""Train and compare candidates using validation data only."""

import logging
import platform
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZIP_DEFLATED

import numpy as np
import skops.io as sio
from threadpoolctl import threadpool_limits

from .config import Config
from .data_loader import LABELS, near_duplicate_audit, partition, scan_dataset
from .evaluate import score_predictions
from .features import FEATURE_SCHEMA, extract_features
from .models import build_models
from .predict import ARTIFACT_SCHEMA, model_path
from .preprocessing import preprocess
from .utils import file_digest, versions, write_json
from .visualization import plot_comparison, plot_confusion, plot_loss

LOG = logging.getLogger(__name__)


def feature_matrix(root, samples, size):
    return np.stack(
        [extract_features(preprocess(root / s.path, size)) for s in samples]
    )


def fit_run(
    cfg: Config,
    root: Path,
    run_dir: Path,
    metrics_dir: Path,
    figure_dir: Path,
    *,
    deep: str | None = None,
    device: str = "auto",
) -> dict:
    """Create an experiment, including a frozen selection and a shared split manifest."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory missing: {root}. See data/README.md."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Validating dataset and checking for duplicates.")
    samples, audit = scan_dataset(root)
    splits = partition(samples, cfg)
    distribution = {
        name: dict(Counter(LABELS[s.label] for s in items))
        for name, items in splits.items()
    }
    audit.update(near_duplicate_audit(splits))
    write_json(
        run_dir / "splits.json",
        {name: [asdict(s) for s in items] for name, items in splits.items()},
    )

    LOG.info("Extracting handcrafted features for train and validation.")
    train, val = splits["train"], splits["validation"]
    x_train, x_val = (
        feature_matrix(root, part, cfg.image_size) for part in (train, val)
    )
    y_train, y_val = (np.array([s.label for s in part]) for part in (train, val))
    if min(Counter(y_train).values()) < cfg.calibration_folds:
        raise ValueError(
            "Training data has too few minority examples for the configured SVM calibration."
        )
    validation, artifacts = {}, {}
    with threadpool_limits(limits=4):
        for name, model in build_models(cfg).items():
            LOG.info("Fitting %s", name)
            model.fit(x_train, y_train)
            validation[name] = score_predictions(
                y_val, model.predict(x_val), model.predict_proba(x_val)[:, 1]
            )
            meta = {"kind": "sklearn"}
            path = model_path(run_dir, name, meta)
            sio.dump(model, path, compression=ZIP_DEFLATED)
            artifacts[name] = {**meta, "sha256": file_digest(path)}

    deep_models = ("cnn", "resnet18") if deep == "both" else (deep,) if deep else ()
    devices = {}
    for deep in deep_models:
        from .cnn import train_network

        meta = {"kind": "torch"}
        path = model_path(run_dir, deep, meta)
        validation[deep], history, actual_device = train_network(
            root, splits, cfg, deep, device, path
        )
        devices[deep] = actual_device
        artifacts[deep] = {**meta, "sha256": file_digest(path)}
        write_json(metrics_dir / f"{deep}_history.json", history)
        plot_loss(history, figure_dir / f"{deep}_loss.svg")

    return record_run(
        cfg,
        run_dir,
        metrics_dir,
        figure_dir,
        validation=validation,
        artifacts=artifacts,
        audit=audit,
        distribution=distribution,
        devices=devices,
    )


def record_run(
    cfg: Config,
    run_dir: Path,
    metrics_dir: Path,
    figure_dir: Path,
    *,
    validation: dict,
    artifacts: dict,
    audit: dict,
    distribution: dict,
    devices: dict,
) -> dict:
    """Publish the candidate comparison and freeze the validation winner."""
    selected = max(validation, key=lambda name: validation[name][cfg.primary_metric])
    LOG.info("Selected %s by validation %s.", selected, cfg.primary_metric)
    write_json(metrics_dir / "validation.json", validation)
    write_json(
        metrics_dir / "dataset_audit.json",
        {"audit": audit, "partition_distribution": distribution},
    )
    plot_comparison(
        validation,
        figure_dir / "validation_curves.svg",
        "Validation · Diseased is positive",
    )
    for name, values in validation.items():
        plot_confusion(
            values,
            figure_dir / f"validation_{name}_confusion.svg",
            f"Validation · {name}",
        )

    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "labels": LABELS,
        "config": asdict(replace(cfg, data_dir="")),
        "preprocessing": {
            "mode": "RGB",
            "resize": "bicubic",
            "size": cfg.image_size,
            "cnn_normalization": "[-1, 1]",
            "resnet18_normalization": "ImageNet mean/std",
        },
        "versions": versions(),
        "python": platform.python_version(),
        "devices": devices,
        "dataset_audit": audit,
        "partition_distribution": distribution,
        "split_sha256": file_digest(run_dir / "splits.json"),
        "models": artifacts,
        "selected_model": selected,
        "validation_summary": {
            name: {
                key: values[key]
                for key in (
                    "accuracy",
                    "weighted_f1",
                    "macro_f1",
                    "roc_auc",
                    "average_precision",
                )
            }
            for name, values in validation.items()
        },
    }
    write_json(run_dir / "run.json", metadata, readable=True)
    return metadata
