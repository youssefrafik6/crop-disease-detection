# Model evaluation

## Protocol

Diseased (1) is the positive class. Precision and recall outside the class report refer to Diseased. Every candidate is scored on validation using accuracy, precision, recall, weighted and macro F1, ROC-AUC, average precision, per-class reports and a confusion matrix. Curves use the disease probability. The majority baseline's constant score supplies a ranking reference. The frozen validation winner is the only candidate evaluated by the held-out command.

Validation selection uses macro F1 by default. The earliest candidate in the fixed comparison order wins an exact tie. The test command records one result per run. Scores from alternate runs on the same test partition should not be used for additional selection. Keep separate partitions for future threshold tuning or external verification.

## Verified PlantVillage run

The top-level tree had 20,639 files across 15 folders. One unsupported file was excluded before decoding, then 14 exact duplicate images were removed. The remaining 20,624 images contained 3,215 healthy and 17,409 diseased examples.

| Partition | Healthy | Diseased | Total |
|---|---:|---:|---:|
| Train | 2,250 | 12,186 | 14,436 |
| Validation | 482 | 2,612 | 3,094 |
| Held out | 483 | 2,611 | 3,094 |

Two equal difference-hash pairs crossed splits. One crossed train and validation and the other crossed validation and test. Visual inspection found substantial similarity in both pairs, consistent with multiple captures of a single leaf. The benchmark uses image-level splits and may overstate performance on unseen plants or collection settings.

### Validation

| Candidate | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.8442 | 0.4578 | 0.7729 | 0.5000 | 0.8442 |
| Logistic Regression | 0.9240 | 0.8611 | 0.9255 | 0.9573 | 0.9906 |
| Decision Tree | 0.8959 | 0.8015 | 0.8958 | 0.8005 | 0.9320 |
| Gaussian Naive Bayes | 0.8361 | 0.7533 | 0.8517 | 0.8884 | 0.9679 |
| Calibrated Linear SVM | 0.9331 | 0.8606 | 0.9298 | 0.9609 | 0.9914 |
| Custom CNN | 0.9919 | 0.9844 | 0.9919 | 0.9990 | 0.9998 |

The Custom CNN had the highest validation macro F1 (0.9844). The CNN ran 7 of its configured maximum of 12 epochs on CPU. Epoch 4 supplied the saved checkpoint. [Validation curves](assets/validation_curves.svg) · [CNN learning curve](assets/cnn_loss.svg).

### Held out

| Candidate | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 0.9913 | 0.9832 | 0.9912 | 0.9988 | 0.9998 |

Disease precision was 0.9920 and disease recall was 0.9977. Healthy recall was 0.9565. The confusion matrix recorded 21 false positives and 6 false negatives. The classical candidates' comparison scores come from validation.

[Confusion matrix](assets/test_confusion.svg) · [ROC and precision–recall curves](assets/test_curves.svg) · [Verified metrics and run metadata](verified_plantvillage.json).

## Historical academic results

These are the four test-set rows recorded in the academic report.

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.9276 | 0.9289 | 0.9621 |
| Decision Tree | 0.8995 | 0.8981 | 0.7953 |
| Gaussian Naive Bayes | 0.8333 | 0.8494 | 0.8946 |
| Linear SVM | 0.9328 | 0.9285 | 0.9650 |

Exact reproduction would require the original partition membership and library versions. The modern run uses a stable sorted and deduplicated dataset, changes SVM calibration preprocessing, and compares candidates by macro F1.

## Error analysis

A false positive labels a healthy image as diseased. This can cause unnecessary follow-up if a prediction is treated as a screening result. A false negative could let visible symptoms go unreviewed. The error figure samples both directions with the training seed so the majority class cannot fill the entire grid. Confidence plots show estimated probability of the predicted label for correct and incorrect predictions. These histograms describe model outputs and do not measure calibration.

The archive is dominated by diseased images. Report healthy recall alongside disease recall to detect poor minority-class performance. Natural damage and lighting variations can resemble disease, while subtle or small lesions can disappear at 128 × 128. The images and confusion matrix illustrate mistakes. Cause cannot be established from the image alone.

The generated montage includes healthy leaves with strong shadows or folds and diseased leaves with relatively small, localized spots. One sampled false negative received an estimated disease probability of 0.01, and one sampled false positive received 0.94. These examples illustrate why high estimated probabilities still require caution. The error montage and per-image records are available in the local run directory.

## Saved CNN verification

Rescoring the saved CNN on the pinned PlantVillage test partition reproduced 99.13% accuracy and 0.9832 macro F1 across 3,094 images. The confusion matrix, ordered Healthy then Diseased, was `[[462, 21], [6, 2605]]`. Healthy-class precision and recall were 0.9872 and 0.9565. Diseased-class precision and recall were 0.9920 and 0.9977.

The authors’ [Cropped PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset) is a candidate external benchmark. A separate run should map its labels to the same binary task, restrict comparisons to overlapping crops, and check for overlap with PlantVillage before scoring. The PlantVillage metrics above describe the held-out partition of the original image collection.

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


The calibrated SVM rerun used a configurable solver tolerance to bound runtime. Classical fitting and CNN fitting ran in separate worker invocations with the same verified partition manifest. Optional ResNet18 weights were exercised in a synthetic one-epoch smoke test. Its full-dataset performance remains unmeasured.
