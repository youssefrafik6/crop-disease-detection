"""The academic classifiers and a majority baseline."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import Config

DESCRIPTIONS = {
    "baseline": "Predicts the majority training label.",
    "logistic": "Fits a regularized linear classifier to standardized texture and color features.",
    "tree": "Learns threshold rules over texture and color features.",
    "naive_bayes": "Models feature distributions separately for each class.",
    "linear_svm": "Learns a linear margin and estimates probabilities through training-only calibration.",
    "cnn": "Learns image features through convolutional layers.",
    "resnet18": "Fine-tunes an ImageNet-pretrained ResNet18 on the leaf labels.",
}


def build_models(cfg: Config) -> dict:
    """Put standardization inside each SVM calibration fold."""
    weights = {"class_weight": "balanced", "random_state": cfg.seed}
    svm = make_pipeline(
        StandardScaler(),
        LinearSVC(C=cfg.svm_c, tol=cfg.svm_tol, max_iter=cfg.svm_max_iter, **weights),
    )
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg.logistic_c, max_iter=2000, **weights),
        ),
        "tree": DecisionTreeClassifier(max_depth=cfg.tree_max_depth, **weights),
        "naive_bayes": make_pipeline(
            StandardScaler(), GaussianNB(var_smoothing=cfg.nb_var_smoothing)
        ),
        "linear_svm": CalibratedClassifierCV(
            svm,
            cv=StratifiedKFold(
                cfg.calibration_folds, shuffle=True, random_state=cfg.seed
            ),
        ),
    }
