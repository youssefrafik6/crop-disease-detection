"""Headless evaluation plots and stratified error examples."""

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from .data_loader import LABELS
from .preprocessing import preprocess

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
    }
)


def save(fig, path):
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")


MODEL_LABELS = {
    "baseline": "Majority baseline",
    "logistic": "Logistic regression",
    "tree": "Decision tree",
    "naive_bayes": "Gaussian Naive Bayes",
    "linear_svm": "Calibrated linear SVM",
    "cnn": "Custom CNN",
    "resnet18": "ResNet18",
}


def plot_comparison(metrics, output, title):
    fig = Figure(figsize=(10, 4.6), layout="constrained")
    roc, pr = fig.subplots(1, 2)
    for name, m in metrics.items():
        label = MODEL_LABELS.get(name, name)
        roc.plot(
            m["roc_curve"]["fpr"],
            m["roc_curve"]["tpr"],
            label=f"{label} ({m['roc_auc']:.4f})",
        )
        pr.step(
            m["precision_recall_curve"]["recall"],
            m["precision_recall_curve"]["precision"],
            where="post",
            label=f"{label} ({m['average_precision']:.4f})",
        )
    dist = next(iter(metrics.values()))["class_distribution"]
    pr.axhline(
        dist["Diseased"] / sum(dist.values()),
        color="0.55",
        linestyle="--",
        label="Prevalence",
    )
    roc.plot([0, 1], [0, 1], "--", color="0.55")
    roc.set(
        xlabel="False positive rate", ylabel="True positive rate", title="ROC · AUC"
    )
    pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–recall · average precision",
    )
    for ax in (roc, pr):
        ax.set(xlim=(0, 1), ylim=(0, 1.04))
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    save(fig, output)


def plot_confusion(metrics, output, title):
    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.subplots()
    ConfusionMatrixDisplay(
        np.array(metrics["confusion_matrix"]), display_labels=list(LABELS.values())
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title(title)
    save(fig, output)


def plot_errors(root, rows, output, seed):
    """Show up to five randomly sampled examples per error direction."""
    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(11, 4.8), layout="constrained")
    axes = fig.subplots(2, 5)
    fig.suptitle("Held-out classification errors")
    for actual, panel in enumerate(axes):
        candidates = [row for row in rows if row["actual"] == actual]
        indices = rng.choice(len(candidates), min(5, len(candidates)), replace=False)
        for ax in panel:
            ax.set_axis_off()
        if not len(indices):
            panel[0].text(
                0.5,
                0.5,
                f"No false {'positives' if actual == 0 else 'negatives'}",
                ha="center",
                va="center",
                wrap=True,
            )
        for ax, idx in zip(panel, indices, strict=False):
            row = candidates[idx]
            ax.imshow(preprocess(root / row["path"]))
            ax.set_title(
                f"{LABELS[actual]} → {LABELS[row['predicted']]}\n"
                f"P(Diseased) = {row['p_diseased']:.2f}",
                fontsize=9,
            )
    save(fig, output)


def plot_confidence(labels, predicted, scores, output):
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    confidence = np.where(predicted == 1, scores, 1 - scores)
    for correct, name in ((True, "Correct"), (False, "Incorrect")):
        values = confidence[(labels == predicted) == correct]
        if len(values):
            ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
    ax.set(
        xlabel="Estimated probability of predicted label",
        ylabel="Images",
        title="Held-out probability distribution",
    )
    ax.legend()
    save(fig, output)


def plot_loss(history, output):
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Training")
    ax.plot(epochs, history["validation_loss"], label="Validation")
    ax.axvline(
        history["best_epoch"], color="0.55", linestyle="--", label="Selected epoch"
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Weighted cross-entropy",
        title="CNN learning curve",
    )
    ax.legend()
    save(fig, output)
