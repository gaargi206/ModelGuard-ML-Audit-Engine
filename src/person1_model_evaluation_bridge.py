import pandas as pd

from src.model_evaluation import (
    evaluate_classification_model,
    summarize_model_evaluation
)


def run_person1_model_evaluation(
    y_true,
    y_pred
):
    """
    Run Person 1's classification evaluation engine.

    Parameters
    ----------
    y_true : array-like
        Actual test labels.

    y_pred : array-like
        Predicted test labels.

    Returns
    -------
    dict
        Raw evaluation results and summary metrics.
    """

    if y_true is None:
        raise ValueError(
            "y_true cannot be None."
        )

    if y_pred is None:
        raise ValueError(
            "y_pred cannot be None."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain "
            "the same number of observations."
        )

    evaluation_results = (
        evaluate_classification_model(
            y_true,
            y_pred
        )
    )

    summary = (
        summarize_model_evaluation(
            evaluation_results
        )
    )

    return {
        "evaluation": evaluation_results,
        "summary": summary
    }


def confusion_matrix_to_dataframe(
    confusion_matrix,
    labels=None
):
    """
    Convert the confusion matrix returned by
    Person 1's engine into a display-friendly
    pandas DataFrame.
    """

    matrix = pd.DataFrame(
        confusion_matrix
    )

    if labels is not None:

        matrix.index = labels
        matrix.columns = labels

    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"

    return matrix


def classification_report_to_dataframe(
    classification_report
):
    """
    Convert Person 1's classification report
    dictionary into a display-friendly DataFrame.
    """

    rows = []

    for label, metrics in classification_report.items():

        if not isinstance(metrics, dict):
            continue

        rows.append({
            "Class": label,
            "Precision": metrics.get(
                "precision",
                0
            ),
            "Recall": metrics.get(
                "recall",
                0
            ),
            "F1 Score": metrics.get(
                "f1-score",
                0
            ),
            "Support": metrics.get(
                "support",
                0
            )
        })

    return pd.DataFrame(rows)
