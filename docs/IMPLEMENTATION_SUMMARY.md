# Implementation summary

The repository contains the original CSC 481 source, report and poster, plus a Python package with a reproducible PlantVillage pipeline, classical model comparison, a majority baseline, PyTorch CNN and optional ResNet18, prediction CLI, Streamlit UI, Docker and GitHub Actions.

[Project audit](PROJECT_AUDIT.md) · [Methodology](METHODOLOGY.md) · [Evaluation](MODEL_EVALUATION.md) · [Quick start](../README.md#quick-start).

## Verified full-dataset experiment

The majority baseline, Logistic Regression, Decision Tree, Gaussian Naive Bayes, calibrated Linear SVM and the custom CNN were trained using the same 14,436 training images and compared on the same 3,094 validation images. The SVM was rerun with a looser, configurable solver tolerance to bound runtime. The CNN ran on CPU and ended after 7 epochs with epoch 4 selected. The classical fitting, SVM rerun and CNN were executed in separate worker invocations against the verified shared manifest, then their candidate results were recorded as one run.

The Custom CNN won by validation macro F1 (0.9844). Its one held-out evaluation scored 0.9913 accuracy, 0.9832 macro F1, 0.9912 weighted F1, 0.9988 ROC-AUC and 0.9998 average precision. The held-out set contained 3,094 images. Two perceptual-hash pairs crossed partition boundaries, so the experiment supports image-level comparison only.

Full run artifacts are available locally in `models/plantvillage`. Machine-readable curves, reports and errors are under `results/metrics/plantvillage`. The plot directory includes the error montage and probability distribution. [Verified results](verified_plantvillage.json) · [Evaluation protocol](MODEL_EVALUATION.md).

## Reproduce the benchmark

From the project root in a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/train_models.py --run plantvillage-reproduction --deep cnn --device cpu
python scripts/evaluate_models.py --run plantvillage-reproduction
CROP_MODEL_RUN=models/plantvillage-reproduction streamlit run app/streamlit_app.py
```

To try the existing local artifacts, install `'.[cnn]'` using pip, then run `streamlit run app/streamlit_app.py` or `python scripts/predict_image.py --run-dir models/plantvillage --image /path/to/leaf.jpg`.

## Verification

- All 28 tests passed on Apple Silicon with Python 3.13 and in the Linux Arm64 Docker environment with Python 3.11. Ruff and Black passed on both. Dependency consistency passed. SciPy logged an L-BFGS-B compatibility warning during the Logistic Regression tests. The final suite had zero failures.
- Synthetic training, model persistence, test-set locking, image decoding, CLI failure handling, upload-to-prediction rendering and temporary archive extraction were exercised. The initial calibrated SVM round-trip failed because it stored the internal `StratifiedKFold` type. The narrowly scoped SVM allowlist fixed the failure.
- The optional ResNet18 used real torchvision ImageNet weights for a one-epoch synthetic smoke test. Full-dataset transfer-learning scores were not measured.
- Docker built successfully, served Streamlit on localhost and loaded a saved CNN as its non-root user. The native Streamlit server also answered its health endpoint. The default CNN-enabled Docker build was verified. The optional classical-only and transfer image variants remain unverified.
- The GitHub workflow passed actionlint and its check commands were exercised in the Python 3.11 container. The hosted GitHub Actions run awaits publication. The repository has no prior commits or configured remote. Common credential and machine-path patterns were scanned in candidate source files and in searchable PDF text and metadata without hits.
- The native computer-use bridge could not start. Streamlit's AppTest exercised the app controls and valid and invalid uploads in combination with the HTTP startup check.

## Before publishing

Publication permission for the original academic materials has been confirmed. The MIT license covers the new portfolio implementation and documentation. Review the files staged in any new Git repository. Exclude local model and dataset directories. [Publication and licensing](PUBLICATION.md).
