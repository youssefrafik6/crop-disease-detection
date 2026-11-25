# Limitations and next experiments

- PlantVillage images often have controlled backgrounds and lighting. Performance on smartphone photos, multiple leaves or complex field backgrounds is unknown.
- Exact duplicate removal protects against identical decoded pixels. Equal 64-bit difference hashes screen for visually similar images. Hash collisions are possible and crops, rotations or lighting changes can produce different hashes. There is no authoritative leaf or plant group identifier. The split is image-level.
- Plant identity and acquisition background may correlate with disease labels. Some species/disease combinations lack comparable healthy examples. A field dataset and a protocol grouped by plant or capture session would test generalization more directly.
- Healthy leaves are the minority in this archive. Blemishes, shadows, deficiency and insect damage can resemble disease. Small lesions can be missed when images are resized or described by global color counts.
- Output is binary and may assign either label to irrelevant inputs. A photograph cannot identify a pathogen in this app. Estimated probabilities can be overconfident, especially outside the training distribution. Future evaluation should add calibration analysis and a rejection policy using a separate calibration set.
- Any sequence of experiments that informs choices from final test results weakens the holdout. Freeze candidates and thresholds before evaluation. Use a new external holdout for subsequent decisions influenced by published scores.
- The binary label follows the source folders. The positive class includes spider-mite damage alongside diseases, which limits pathogen-level interpretation.
