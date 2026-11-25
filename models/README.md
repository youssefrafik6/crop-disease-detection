# Model artifacts

Each training command creates a uniquely named directory containing a schema-versioned `run.json`, `splits.json`, one `.skops` file per classical candidate and optional `.safetensors` checkpoints. After held-out evaluation it also contains `test.json`. The run metadata carries checksums, dependencies, label mappings, preprocessing, training settings, validation summaries and the frozen selection. Full curves, metrics and error records are under `results/`; preserve them alongside artifacts if archiving a run.

Regenerate from the original Kaggle archive with the commands in the main README. Model files and full split manifests are ignored by Git. If redistribution is approved and versioned binaries become necessary, Git LFS is an option for large checkpoints.

Load artifacts you generated or obtained from a trusted source. The app does not accept model uploads. Classical artifacts use skops with a narrow allowlist for calibration internals. Neural models load dense tensor state dictionaries through safetensors so prediction does not invoke a pickle loader. Checksums detect unintended file changes and provide integrity only relative to trusted metadata.

[Safetensors format and PyTorch API](https://huggingface.co/docs/safetensors/api/torch) · [skops persistence](https://skops.readthedocs.io/en/stable/persistence.html).
