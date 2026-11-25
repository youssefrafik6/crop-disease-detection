!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# download PlantVillage dataset
!kaggle datasets download -d emmarex/plantdisease

# unzip
!unzip -q plantdisease.zip -d plant_data

# check files
!ls plant_data
!pip install -q tqdm


# Imports
import os
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2gray

import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Imports done!")


# Gather file paths and labels

DATA_DIR = "plant_data/PlantVillage"
IMG_SIZE = 128

all_paths = []
all_labels = []

for folder_name in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    label = 0 if "healthy" in folder_name.lower() else 1
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            all_paths.append(fpath)
            all_labels.append(label)

all_labels = np.array(all_labels)
print(f"Total images found: {len(all_paths)}")
print(f"  Healthy (0): {np.sum(all_labels == 0)}")
print(f"  Diseased (1): {np.sum(all_labels == 1)}")


# Stratified train / val / test split (paths only, still no loading)
paths_train, paths_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=SEED
)
paths_val, paths_test, y_val, y_test = train_test_split(
    paths_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Train: {len(paths_train)}  Val: {len(paths_val)}  Test: {len(paths_test)}")
print(f"Train class counts: {Counter(y_train)}")


# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Images", fontsize=14)

for row, label, name in [(0, 0, "Healthy"), (1, 1, "Diseased")]:
    idxs = np.where(y_train == label)[0]
    picked = np.random.choice(idxs, 10, replace=False)  # grab extra in case some fail
    shown = 0
    for idx in picked:
        if shown >= 5:
            break
        try:
            img = Image.open(paths_train[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            axes[row, shown].imshow(img)
            axes[row, shown].set_title(name, fontsize=10)
            axes[row, shown].axis("off")
            shown += 1
        except Exception:
            pass

plt.tight_layout()
plt.show()


# Feature extraction: HOG + color histogram

# We read one image at a time, extract features, throw away the image.
# This keeps memory low — only the small feature vectors stay in RAM.

def extract_features(img_path):
    """Read one image, return HOG + color histogram feature vector."""
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.uint8)
    arr_f = arr.astype(np.float32) / 255.0

    # HOG on grayscale
    gray = rgb2gray(arr)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)

    # Color histogram — 32 bins per channel
    hr = np.histogram(arr_f[:,:,0], bins=32, range=(0,1))[0]
    hg = np.histogram(arr_f[:,:,1], bins=32, range=(0,1))[0]
    hb = np.histogram(arr_f[:,:,2], bins=32, range=(0,1))[0]
    color_hist = np.concatenate([hr, hg, hb]).astype(np.float64)
    color_hist /= (color_hist.sum() + 1e-8)

    return np.concatenate([hog_feat, color_hist])


def extract_all(paths, labels, name=""):
    """Extract features for a list of image paths, skipping bad files."""
    feats = []
    good_labels = []
    good_paths = []
    skipped = 0
    for i, p in enumerate(paths):
        try:
            feats.append(extract_features(p))
            good_labels.append(labels[i])
            good_paths.append(p)
        except Exception:
            skipped += 1
        if (i + 1) % 2000 == 0:
            print(f"  {name}: {i+1}/{len(paths)}")
    print(f"  {name}: {len(feats)} good, {skipped} skipped")
    return np.array(feats), np.array(good_labels), good_paths


print("Extracting features (takes a few minutes)...\n")
X_train, y_train, paths_train = extract_all(paths_train, y_train, "Train")
X_val,   y_val,   paths_val   = extract_all(paths_val,   y_val,   "Val")
X_test,  y_test,  paths_test  = extract_all(paths_test,  y_test,  "Test")

print(f"\nFeature vector length: {X_train.shape[1]}")


# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Free unscaled arrays
del X_train, X_val, X_test
gc.collect()

print("Scaled with StandardScaler.")


# Train and evaluate models
def evaluate(name, model, X_tr, y_tr, X_ev, y_ev):
    """Train model, print metrics on eval set."""
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ev)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_ev)[:, 1]
        auc = roc_auc_score(y_ev, y_prob)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_ev)
        auc = roc_auc_score(y_ev, y_prob)
    else:
        y_prob = None
        auc = None

    print(f"\n  === {name} ===")
    print(classification_report(y_ev, y_pred,
                                target_names=["Healthy", "Diseased"]))
    if auc is not None:
        print(f"  ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_ev, y_pred)
    print(f"  Confusion Matrix:\n  {cm}")

    return {
        "name": name,
        "model": model,
        "acc": accuracy_score(y_ev, y_pred),
        "prec": precision_score(y_ev, y_pred, average='weighted'),
        "rec": recall_score(y_ev, y_pred, average='weighted'),
        "f1": f1_score(y_ev, y_pred, average='weighted'),
        "auc": auc,
        "y_prob": y_prob,
        "cm": cm
    }


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced', random_state=SEED
    ),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, random_state=SEED), cv=3
    ),
}

results = []
for name, model in models.items():
    res = evaluate(name, model, X_train_s, y_train, X_val_s, y_val)
    results.append(res)


# Comparison bar chart
names = [r["name"] for r in results]
f1s   = [r["f1"] for r in results]
aucs  = [r["auc"] if r["auc"] else 0 for r in results]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, f1s, w, label="F1 (weighted)", color="steelblue")
ax.bar(x + w/2, aucs, w, label="ROC-AUC", color="coral")
ax.set_ylabel("Score")
ax.set_title("Model Comparison (Validation Set)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.show()

best = max(results, key=lambda r: r["f1"])
print(f"\nBest model by F1: {best['name']} (F1={best['f1']:.4f})")


# Confusion matrices side by side
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
for i, r in enumerate(results):
    sns.heatmap(r["cm"], annot=True, fmt='d', cmap='Blues',
                xticklabels=["Healthy", "Diseased"],
                yticklabels=["Healthy", "Diseased"], ax=axes[i])
    axes[i].set_title(r["name"], fontsize=11)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.tight_layout()
plt.show()


# Best model: TEST set evaluation
best_model = best["model"]
y_test_pred = best_model.predict(X_test_s)

print(f"\n{'='*50}")
print(f"  FINAL TEST RESULTS — {best['name']}")
print(f"{'='*50}")
print(classification_report(y_test, y_test_pred,
                            target_names=["Healthy", "Diseased"]))

if hasattr(best_model, "predict_proba"):
    y_test_prob = best_model.predict_proba(X_test_s)[:, 1]
elif hasattr(best_model, "decision_function"):
    y_test_prob = best_model.decision_function(X_test_s)
else:
    y_test_prob = None

if y_test_prob is not None:
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"ROC-AUC: {test_auc:.4f}")

cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Healthy", "Diseased"],
            yticklabels=["Healthy", "Diseased"])
plt.title(f"{best['name']} — Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ROC curves for all models (Test set)
fig, ax = plt.subplots(figsize=(7, 6))

for r in results:
    m = r["model"]
    if hasattr(m, "predict_proba"):
        prob = m.predict_proba(X_test_s)[:, 1]
    elif hasattr(m, "decision_function"):
        prob = m.decision_function(X_test_s)
    else:
        continue
    auc_val = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, label=f"{r['name']} (AUC={auc_val:.3f})", linewidth=2)

ax.plot([0,1], [0,1], 'k--', alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models (Test Set)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Full comparison table (Test set)
print("\n" + "=" * 65)
print("  FINAL COMPARISON — TEST SET")
print("=" * 65)

print(f"\n{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("-" * 58)

for r in results:
    m = r["model"]
    yp = m.predict(X_test_s)
    acc  = accuracy_score(y_test, yp)
    prec = precision_score(y_test, yp, average='weighted')
    rec  = recall_score(y_test, yp, average='weighted')
    f1   = f1_score(y_test, yp, average='weighted')
    if hasattr(m, "predict_proba"):
        auc = roc_auc_score(y_test, m.predict_proba(X_test_s)[:,1])
    elif hasattr(m, "decision_function"):
        auc = roc_auc_score(y_test, m.decision_function(X_test_s))
    else:
        auc = None
    a = f"{auc:.4f}" if auc else "  N/A "
    print(f"{r['name']:<22} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {a:>7}")


# Error analysis: show misclassified images
mis_idxs = np.where(y_test_pred != y_test)[0]
print(f"\nMisclassified: {len(mis_idxs)} out of {len(y_test)}")

n = min(10, len(mis_idxs))
picked = np.random.choice(mis_idxs, n, replace=False)
label_map = {0: "Healthy", 1: "Diseased"}

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle(f"Misclassified Images ({best['name']})", fontsize=14)

for i, idx in enumerate(picked):
    row, col = divmod(i, 5)
    img = Image.open(paths_test[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    axes[row, col].imshow(img)
    axes[row, col].set_title(
        f"True: {label_map[y_test[idx]]}\nPred: {label_map[y_test_pred[idx]]}",
        fontsize=9)
    axes[row, col].axis("off")

for i in range(n, 10):
    row, col = divmod(i, 5)
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()
