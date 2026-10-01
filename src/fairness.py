import pandas as pd
from sklearn.metrics import accuracy_score


def group_accuracy(
    model,
    X_test,
    y_test,
    group_column
):
    if group_column not in X_test.columns:
        raise ValueError(
            f"Group column '{group_column}' was not found in X_test."
        )

    predictions = model.predict(X_test)

    results = []

    for group_value in X_test[group_column].dropna().unique():
        mask = X_test[group_column] == group_value

        if mask.sum() == 0:
            continue

        accuracy = accuracy_score(
            y_test[mask],
            predictions[mask]
        )

        results.append({
            "group": group_value,
            "count": int(mask.sum()),
            "accuracy": float(accuracy)
        })

    return pd.DataFrame(results)


def group_positive_prediction_rate(
    model,
    X_test,
    group_column,
    positive_class
):
    if group_column not in X_test.columns:
        raise ValueError(
            f"Group column '{group_column}' was not found in X_test."
        )

    predictions = model.predict(X_test)

    results = []

    for group_value in X_test[group_column].dropna().unique():
        mask = X_test[group_column] == group_value

        if mask.sum() == 0:
            continue

        positive_rate = (
            (predictions[mask] == positive_class).mean()
        )

        results.append({
            "group": group_value,
            "count": int(mask.sum()),
            "positive_prediction_rate": float(
                positive_rate
            )
        })

    return pd.DataFrame(results)


def calculate_group_accuracy_disparity(
    group_accuracy_results,
    threshold=0.10
):
    if group_accuracy_results.empty:
        return {
            "max_accuracy": None,
            "min_accuracy": None,
            "accuracy_gap": None,
            "threshold": threshold,
            "disparity_detected": False
        }

    max_accuracy = float(
        group_accuracy_results["accuracy"].max()
    )
    min_accuracy = float(
        group_accuracy_results["accuracy"].min()
    )
    accuracy_gap = max_accuracy - min_accuracy

    return {
        "max_accuracy": max_accuracy,
        "min_accuracy": min_accuracy,
        "accuracy_gap": float(accuracy_gap),
        "threshold": threshold,
        "disparity_detected": bool(
            accuracy_gap > threshold
        )
    }


def audit_fairness(
    model,
    X_test,
    y_test,
    group_column,
    positive_class=">50K",
    disparity_threshold=0.10
):
    group_accuracy_results = group_accuracy(
        model,
        X_test,
        y_test,
        group_column
    )

    positive_prediction_results = (
        group_positive_prediction_rate(
            model,
            X_test,
            group_column,
            positive_class
        )
    )

    disparity_results = (
        calculate_group_accuracy_disparity(
            group_accuracy_results,
            threshold=disparity_threshold
        )
    )

    return {
        "group_accuracy": group_accuracy_results,
        "group_positive_prediction_rate":
            positive_prediction_results,
        "accuracy_disparity": disparity_results
    }


if __name__ == "__main__":
    print("fairness.py loaded successfully.")
