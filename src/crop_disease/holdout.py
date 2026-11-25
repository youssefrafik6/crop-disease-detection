"""Evaluate the frozen validation winner on the held-out partition."""

import numpy as np

from .data_loader import Sample, rgb_digest
from .evaluate import score_predictions
from .predict import load_model
from .preprocessing import decode_rgb
from .train import feature_matrix
from .utils import file_digest, read_json, write_json
from .visualization import plot_comparison, plot_confidence, plot_confusion, plot_errors


def evaluate_holdout(root, run_dir, metrics_dir, figure_dir):
    """Record one held-out evaluation per training run."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory missing: {root}. See data/README.md."
        )
    output = run_dir / "test.json"
    with (run_dir / ".evaluation.lock").open("x") as lock:
        lock.write("Evaluation in progress\n")
    try:
        if output.exists():
            raise ValueError("This run already has a recorded held-out result.")
        metadata = read_json(run_dir / "run.json")
        manifest = run_dir / "splits.json"
        if metadata["split_sha256"] != file_digest(manifest):
            raise ValueError("The split manifest has changed.")
        samples = [Sample(**row) for row in read_json(manifest)["test"]]
        for sample in samples:
            image = decode_rgb(root / sample.path)
            if rgb_digest(image) != sample.digest:
                raise ValueError(
                    f"Dataset no longer matches manifest at {sample.path}."
                )
        name, model, size = load_model(run_dir)
        if metadata["models"][name]["kind"] == "torch":
            from .cnn import predict_samples

            predicted, scores = predict_samples(model, root, samples, size, name)
        else:
            features = feature_matrix(root, samples, size)
            predicted = model.predict(features)
            scores = model.predict_proba(features)[:, 1]
        labels = np.array([s.label for s in samples])
        metrics = score_predictions(labels, predicted, scores)
        errors = [
            {
                "path": sample.path,
                "actual": int(actual),
                "predicted": int(pred),
                "p_diseased": float(score),
            }
            for sample, actual, pred, score in zip(
                samples, labels, predicted, scores, strict=True
            )
            if actual != pred
        ]
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(
            {name: metrics},
            figure_dir / "test_curves.svg",
            "Held out · Diseased is positive",
        )
        plot_confusion(metrics, figure_dir / "test_confusion.svg", f"Held out · {name}")
        plot_errors(
            root,
            errors,
            figure_dir / "test_error_examples.png",
            metadata["config"]["seed"],
        )
        if name != "baseline":
            plot_confidence(
                labels, predicted, scores, figure_dir / "test_probabilities.svg"
            )
        write_json(metrics_dir / "test_errors.json", errors)
        result = {"selected_model": name, "metrics": metrics}
        write_json(metrics_dir / "test.json", result)
        write_json(output, result)
        return result
    finally:
        (run_dir / ".evaluation.lock").unlink()
