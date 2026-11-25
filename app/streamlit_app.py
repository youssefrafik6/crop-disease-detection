"""Streamlit front end for locally trained artifacts."""

import os
from pathlib import Path

import streamlit as st

from crop_disease.models import DESCRIPTIONS
from crop_disease.predict import load_model, predict_image
from crop_disease.preprocessing import MAX_BYTES, decode_rgb
from crop_disease.utils import read_json

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿", layout="wide")

st.title("Crop Disease Detection")
st.write("Upload a leaf image to explore how the trained models classify it.")
st.caption(
    "Experimental predictions may not generalize to field images. The models do not identify specific diseases or validate plant health."
)

run_dir = Path(os.getenv("CROP_MODEL_RUN", "models/plantvillage"))


@st.cache_resource(show_spinner=False)
def cached_model(directory: Path, name: str, metadata: bytes):
    """Reload a model if the local run manifest changes."""
    return load_model(directory, name)


try:
    run = read_json(run_dir / "run.json")
except (OSError, ValueError):
    st.info("Train a model run and set CROP_MODEL_RUN to its directory.")
    st.code("python scripts/train_models.py --run plantvillage", language="bash")
    st.stop()

names = list(run["models"])
left, right = st.columns([1, 1])
with left:
    model = st.selectbox(
        "Trained model",
        names,
        index=names.index(run["selected_model"]),
        format_func=lambda name: name.replace("_", " ").title(),
    )
    st.caption(DESCRIPTIONS[model])
    upload = st.file_uploader(
        "Leaf image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Still images up to 20 MiB and 25 megapixels.",
    )
    pressed = st.button("Classify leaf", type="primary", disabled=upload is None)
with right:
    if upload is not None:
        try:
            if upload.size > MAX_BYTES:
                raise ValueError("Use an image no larger than 20 MiB.")
            raw = upload.getvalue()
            preview = decode_rgb(raw)
            preview.thumbnail((640, 450))
            st.image(preview, caption="Uploaded image")
            if pressed:
                with st.spinner("Classifying image…"):
                    loaded = cached_model(
                        run_dir, model, (run_dir / "run.json").read_bytes()
                    )
                    result = predict_image(loaded, raw)
                st.metric("Predicted label", result["label"])
                probability = result["estimated_probability"]
                if probability is not None:
                    st.write(
                        f"Estimated probability of this label: **{probability:.1%}**"
                    )
                    st.caption(
                        "This estimate may be less reliable on images from different growing environments."
                    )
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            st.error(str(exc))

with st.expander("Technical details"):
    st.write(
        "Inputs are converted to RGB and resized to 128 × 128. Traditional classifiers use 1,860 texture and color features. Neural classifiers learn from normalized pixels."
    )
    st.write(
        f"Validation winner **{run['selected_model']}** by **{run['config']['primary_metric']}**."
    )
    st.json(run["validation_summary"][model])
