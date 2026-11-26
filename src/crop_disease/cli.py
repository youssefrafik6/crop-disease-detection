"""Command-line entry points for experiments and prediction."""

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .holdout import evaluate_holdout
from .predict import load_model, predict_image
from .train import fit_run
from .utils import safe_name


def _experiment_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run", required=True, help="Unique experiment name.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=os.getenv("CROP_DATA_DIR"))
    return parser


def train_main():
    parser = _experiment_parser("Fit candidates and freeze the validation winner.")
    parser.add_argument(
        "--deep",
        choices=("cnn", "resnet18", "both"),
        help="Include additional deep learning candidates.",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override maximum neural training epochs."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run = safe_name(args.run)
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg = replace(cfg, cnn_epochs=args.epochs)
        root = args.data_dir or Path(cfg.data_dir)
        result = fit_run(
            cfg,
            root,
            Path("models", run),
            Path("results/metrics", run),
            Path("results/figures", run),
            deep=args.deep,
            device=args.device,
        )
        print(f"Validation winner: {result['selected_model']}")
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


def evaluate_main():
    parser = _experiment_parser("Evaluate the already-selected model on held-out data.")
    args = parser.parse_args()
    try:
        run = safe_name(args.run)
        root = args.data_dir or Path(load_config(args.config).data_dir)
        result = evaluate_holdout(
            root,
            Path("models", run),
            Path("results/metrics", run),
            Path("results/figures", run),
        )
        print(
            f"Held-out macro F1 ({result['selected_model']}): {result['metrics']['macro_f1']:.4f}"
        )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


def predict_main():
    parser = argparse.ArgumentParser(
        description="Predict the class of a single leaf image."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.getenv("CROP_MODEL_RUN", "models/plantvillage")),
    )
    parser.add_argument(
        "--model", help="Candidate name. Defaults to the validation winner."
    )
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = predict_image(load_model(args.run_dir, args.model), args.image)
        print(f"Prediction: {result['label']}")
        if result["estimated_probability"] is not None:
            print(f"Model score for the predicted label: {result['estimated_probability']:.2f}")
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
