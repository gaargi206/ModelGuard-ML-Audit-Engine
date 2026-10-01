
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_classification_model(y_true, y_pred):
    """
    Evaluate a classification model.

    Parameters:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        dict: Model evaluation metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }


def summarize_model_evaluation(evaluation_results):
    """
    Extract key classification metrics.
    """

    report = evaluation_results["classification_report"]

    return {
        "accuracy": evaluation_results["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]
    }
