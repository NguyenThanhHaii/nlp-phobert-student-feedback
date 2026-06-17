"""Common evaluation metrics for classification tasks."""

from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Compute standard classification metrics.

    Macro-F1 is the main metric because class imbalance is expected.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
