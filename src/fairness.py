
import pandas as pd
from sklearn.metrics import accuracy_score


def group_accuracy(
    model,
    X_test,
    y_test,
    group_column
):
    """
    Calculate model accuracy separately for each group.

    Parameters:
        model: Trained classification model/pipeline.
        X_test (pandas.DataFrame): Test features.
        y_test: Actual test targets.
        group_column (str): Column used to define groups.

    Returns:
        pandas.DataFrame: Group-wise accuracy results.
    """

    if group_column not in X_test.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in X_test."
        )

    predictions = model.predict(X_test)

    results = []

    for group_value in X_test[group_column].dropna().unique():
        mask = X_test[group_column] == group_value

        group_accuracy_value = accuracy_score(
            y_test[mask],
            predictions[mask]
        )

        results.append({
            "group": group_value,
            "sample_count": int(mask.sum()),
            "accuracy": round(group_accuracy_value, 4)
        })

    return pd.DataFrame(results)


def group_positive_prediction_rate(
    model,
    X_test,
    group_column,
    positive_class=">50K"
):
    """
    Calculate the model's positive prediction rate for each group.

    Parameters:
        model: Trained classification model/pipeline.
        X_test (pandas.DataFrame): Test features.
        group_column (str): Column used to define groups.
        positive_class: Class treated as the positive outcome.

    Returns:
        pandas.DataFrame: Group-wise positive prediction rates.
    """

    if group_column not in X_test.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in X_test."
        )

    predictions = model.predict(X_test)

    results = []

    for group_value in X_test[group_column].dropna().unique():
        mask = X_test[group_column] == group_value

        group_predictions = predictions[mask]

        positive_count = (
            group_predictions == positive_class
        ).sum()

        total_count = len(group_predictions)

        positive_rate = (
            positive_count / total_count
            if total_count > 0 else 0
        )

        results.append({
            "group": group_value,
            "sample_count": total_count,
            "positive_prediction_rate": round(
                positive_rate, 4
            )
        })

    return pd.DataFrame(results)


def calculate_group_accuracy_disparity(
    group_accuracy_results,
    threshold=0.10
):
    """
    Calculate the difference between the highest and lowest
    group accuracy.

    Parameters:
        group_accuracy_results (pandas.DataFrame):
            Output from group_accuracy().
        threshold (float):
            Difference threshold for flagging a potential disparity.

    Returns:
        dict: Group accuracy disparity information.
    """

    if group_accuracy_results.empty:
        return {
            "max_accuracy": None,
            "min_accuracy": None,
            "accuracy_gap": None,
            "threshold": threshold,
            "disparity_detected": False
        }

    max_accuracy = group_accuracy_results["accuracy"].max()
    min_accuracy = group_accuracy_results["accuracy"].min()

    accuracy_gap = max_accuracy - min_accuracy

    disparity_detected = accuracy_gap >= threshold

    return {
        "max_accuracy": round(max_accuracy, 4),
        "min_accuracy": round(min_accuracy, 4),
        "accuracy_gap": round(accuracy_gap, 4),
        "threshold": threshold,
        "disparity_detected": disparity_detected
    }


def audit_fairness(
    model,
    X_test,
    y_test,
    group_column,
    positive_class=">50K",
    disparity_threshold=0.10
):
    """
    Run the complete fairness/group analysis.

    Parameters:
        model: Trained classification model/pipeline.
        X_test (pandas.DataFrame): Test features.
        y_test: Actual test targets.
        group_column (str): Column used to define groups.
        positive_class: Class treated as the positive outcome.
        disparity_threshold (float): Accuracy-gap threshold.

    Returns:
        dict: Complete fairness audit results.
    """

    accuracy_results = group_accuracy(
        model,
        X_test,
        y_test,
        group_column
    )

    positive_rate_results = group_positive_prediction_rate(
        model,
        X_test,
        group_column,
        positive_class=positive_class
    )

    accuracy_disparity = calculate_group_accuracy_disparity(
        accuracy_results,
        threshold=disparity_threshold
    )

    return {
        "group_accuracy": accuracy_results,
        "positive_prediction_rate": positive_rate_results,
        "accuracy_disparity": accuracy_disparity
    }
