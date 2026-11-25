import pytest
from sklearn.pipeline import Pipeline

from crop_disease.config import Config
from crop_disease.holdout import evaluate_holdout
from crop_disease.models import build_models
from crop_disease.utils import read_json


def test_models_and_calibration_pipeline():
    models = build_models(Config())
    assert set(models) == {"baseline", "logistic", "tree", "naive_bayes", "linear_svm"}
    assert isinstance(models["linear_svm"].estimator, Pipeline)
    assert models["linear_svm"].estimator.steps[0][0] == "standardscaler"


def test_full_synthetic_experiment(synthetic_data, trained_run):
    run, metrics, figures = trained_run
    manifest = read_json(run / "run.json")
    scores = read_json(metrics / "validation.json")
    assert manifest["selected_model"] == max(
        scores, key=lambda name: scores[name]["macro_f1"]
    )
    assert manifest["devices"] == {"cnn": "cpu"}
    assert read_json(metrics / "cnn_history.json")["best_epoch"] == 1
    assert not (run / "test.json").exists()
    result = evaluate_holdout(synthetic_data, run, metrics, figures)
    assert result["selected_model"] == manifest["selected_model"]
    assert result["metrics"]["positive_label"] == 1
    assert result["metrics"]["accuracy"] >= 0.9
    assert (figures / "test_error_examples.png").is_file()
    with pytest.raises(ValueError, match="already"):
        evaluate_holdout(synthetic_data, run, metrics, figures)
