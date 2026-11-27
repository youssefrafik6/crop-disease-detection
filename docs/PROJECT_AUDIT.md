# Academic project audit

Reviewed the complete Colab export, the 14-page final report and the presentation poster. Commands and future-work suggestions inside those materials describe the academic work.

| Area | Original implementation and rationale for the change |
|---|---|
| Working design | Binary folder labels, stratified 70/15/15 split, one-image-at-a-time extraction, 128 × 128 RGB, grayscale HOG plus RGB histograms, four classifiers, classification reports, ROC and qualitative error review were sound foundations. |
| Environment | Notebook shell escapes install tqdm (unused), copy a Kaggle credential file, download/unzip, and depend on the working directory and `plant_data/PlantVillage`. The imports require NumPy, Pillow, scikit-image, scikit-learn, Matplotlib and Seaborn, with Kaggle CLI and unzip for acquisition. No dependency versions were recorded. |
| Data integrity | Inner folder listing is unsorted. A fixed seed cannot stabilize membership across different input ordering. Every direct child file is accepted, any folder containing “healthy” is labeled healthy, broad exceptions hide defects, and failures after splitting can alter proportions. There are no duplicate or same-leaf checks. |
| Leakage | The original scaler excludes validation and test images but is fitted before SVM calibration CV, exposing calibration folds to scaling statistics. Copies or alternate captures could cross the image-level partition. No plant or capture-group manifest was saved. |
| Selection | Weighted F1 drives validation selection in code. The poster describes SVM as best by F1 and ROC-AUC, while the report's test weighted F1 is slightly higher for Logistic Regression. The original code evaluates the selected model and then evaluates every model again on test for ROC and comparison. The validation scores identifying the selected model are not recoverable from the table alone. |
| Counts | The report states 38 folders, 20,639 usable images and one skipped input. The code counts files before decode, with no folder or image manifest to resolve exact membership. The wider 54,303-image release referenced on the poster is a different cohort. |
| Efficiency and usability | Plotting is interactive, test scores are recomputed, every model receives scaled arrays and no split, model, configuration or machine-readable metrics are saved. No inference interface, deployment recipe, baseline or automated tests exist. |
| Security | No embedded credential value was observed in the source. The setup copies `kaggle.json` into the user's home directory, global warning suppression hides important diagnostics and image inputs lack explicit resource and format bounds. |

The rebuild validates and deduplicates images before splitting, records a stable manifest, fits scaling within SVM calibration folds, compares all candidates on validation and evaluates the selected candidate on held-out test. It adds versioned artifacts, a CLI, a Streamlit app, automated tests and a CNN experiment on the same partitions.

The verified Kaggle ZIP contains a top-level tree and a nested copy. The top-level tree has 15 class folders and 20,639 files. Two cross-split pairs shared the same 64-bit perceptual hash. Visual inspection found matching leaf outlines and vein patterns with small capture changes in both pairs. The split remains image-level and those pairs are a concrete limitation of the comparison.
