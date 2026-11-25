"""Experiment configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_dir: str = "data/raw/PlantVillage"
    holdout_fraction: float = 0.30
    test_share_of_holdout: float = 0.50
    primary_metric: str = "macro_f1"
    image_size: int = 128
    logistic_c: float = 1.0
    svm_c: float = 1.0
    svm_tol: float = 0.01
    svm_max_iter: int = 2000
    calibration_folds: int = 3
    tree_max_depth: int | None = None
    nb_var_smoothing: float = 1e-9
    cnn_epochs: int = 12
    cnn_batch_size: int = 64
    cnn_patience: int = 3
    cnn_learning_rate: float = 0.001

    def __post_init__(self):
        if not 0 < self.holdout_fraction < 1 or not 0 < self.test_share_of_holdout < 1:
            raise ValueError("Split fractions must be between zero and one.")
        if self.image_size != 128:
            raise ValueError("The academic feature schema requires 128 × 128 images.")
        if self.primary_metric not in {
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "roc_auc",
            "average_precision",
        }:
            raise ValueError("Unsupported primary validation metric.")
        if (
            min(
                self.cnn_epochs,
                self.cnn_batch_size,
                self.cnn_patience,
                self.svm_max_iter,
            )
            < 1
            or self.calibration_folds < 2
        ):
            raise ValueError(
                "Positive CNN settings and at least two calibration folds are required."
            )
        if (
            min(
                self.logistic_c,
                self.svm_c,
                self.svm_tol,
                self.nb_var_smoothing,
                self.cnn_learning_rate,
            )
            <= 0
        ):
            raise ValueError(
                "Regularization, smoothing and learning rate must be positive."
            )


def load_config(path: Path) -> Config:
    """Read a trusted local YAML configuration."""
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Expected a YAML mapping of settings.")
        return Config(**values)
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError(f"Invalid experiment configuration in {path}: {exc}") from exc
