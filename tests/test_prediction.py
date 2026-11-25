from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from crop_disease.predict import load_model, predict_image
from crop_disease.preprocessing import InvalidImage


@pytest.mark.parametrize(
    "name", ["baseline", "logistic", "tree", "naive_bayes", "linear_svm", "cnn"]
)
def test_prediction_roundtrip(trained_run, synthetic_data, name):
    run, _, _ = trained_run
    path = synthetic_data / "Tomato_healthy" / "00.png"
    first = predict_image(load_model(run, name), path)
    second = predict_image(load_model(run, name), path.read_bytes())
    assert first == second
    assert first["label"] in {"Healthy", "Diseased"}
    if name == "baseline":
        assert first["estimated_probability"] is None
    else:
        assert 0 <= first["estimated_probability"] <= 1


def test_bad_input(trained_run):
    run, _, _ = trained_run
    with pytest.raises(InvalidImage):
        predict_image(load_model(run), b"corrupt")


def test_streamlit_uses_model_run(trained_run, monkeypatch):
    monkeypatch.setenv("CROP_MODEL_RUN", str(trained_run[0]))
    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(path)).run()
    assert not app.exception
    assert app.title[0].value == "Crop Disease Detection"
    assert "cnn" in [option.lower() for option in app.selectbox[0].options]
    assert app.button[0].disabled


@pytest.mark.parametrize("valid", [True, False])
def test_streamlit_uploaded_image(valid, trained_run, synthetic_data, monkeypatch):
    """Exercise the upload-to-result path with an in-memory file."""
    from io import BytesIO

    import streamlit as st

    monkeypatch.setenv("CROP_MODEL_RUN", str(trained_run[0]))
    path = synthetic_data / "Tomato_healthy" / "00.png"
    raw = path.read_bytes() if valid else b"not a supported image"
    uploaded = BytesIO(raw)
    uploaded.size = len(raw)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    if valid:
        app.button[0].click().run()
        assert not app.exception
        assert app.metric[0].value == "Healthy"
        assert any(
            "Estimated probability of this label" in line.value for line in app.markdown
        )
    else:
        assert "Cannot decode image" in app.error[0].value
