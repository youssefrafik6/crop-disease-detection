# Dataset setup

Use the academic [PlantVillage archive by emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease).

```bash
python scripts/download_data.py
```

If Kaggle requires browser sign-in, download the ZIP from the dataset page and pass the archive location locally.

```bash
python scripts/download_data.py --archive /path/to/plantdisease.zip
```

The downloader extracts only the top-level `PlantVillage/<class>/<image>` tree. It rejects archive traversal and symlink entries in that tree, enforces an archive size limit and refuses to overwrite an existing destination. The current ZIP also contains a nested mirror. Extracting both trees would duplicate the cohort. The downloaded ZIP was about 672 MiB compressed. Allow approximately 1.1 GiB for the archive, chosen extracted tree and experiment outputs.

The default path is `data/raw/PlantVillage`. Override it with `--data-dir`, the `CROP_DATA_DIR` environment variable, or `data_dir` in YAML, in that order. Supply the directory directly containing the class folders. The folder suffix `_healthy` maps to 0. Other included class folders map to 1. Keep unrelated image folders outside that directory.

The named archive contains 20,639 files in 15 class folders for tomato, potato and bell pepper. The academic report lists 38 class folders, a scope that does not match the downloaded archive. The separate 38-category PlantVillage distribution was excluded from this experiment. Validation and deduplication can also change the usable counts. See the experiment's recorded dataset audit.

Keep images, archives, manifests, credentials and processed data out of Git. The download helper uses a public URL and does not read or write Kaggle credentials.

The archive used for verification had SHA-256 `31503970274c82dd26ddb1decb8cf3e7d49bc0420b4281b23abc0201bffce097`. Kaggle can revise the archive. Compare the hash, class counts and split fingerprint when reproducing the recorded experiment.
