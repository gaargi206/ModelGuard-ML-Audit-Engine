import numpy as np
import pandas as pd


def detect_class_imbalance(
    df,
    target_column,
    threshold=0.80
):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the DataFrame."
        )

    distribution = df[target_column].value_counts(normalize=True)

    if distribution.empty:
        return {
            "risk": False,
            "minority_class_proportion": np.nan,
            "threshold": threshold
        }

    minority_proportion = float(distribution.min())
    majority_proportion = float(distribution.max())

    return {
        "risk": bool(
            minority_proportion <
            (1 - threshold) * majority_proportion
        ),
        "minority_class_proportion": minority_proportion,
        "majority_class_proportion": majority_proportion,
        "threshold": threshold
    }


def detect_missing_value_risk(df, threshold=20.0):
    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    risky_columns = missing_percentage[
        missing_percentage > threshold
    ]

    return {
        "risk": not risky_columns.empty,
        "threshold": threshold,
        "risky_columns": risky_columns.to_dict()
    }


def detect_duplicate_risk(df, threshold=5.0):
    duplicate_percentage = (
        df.duplicated().mean() * 100
        if len(df) else 0.0
    )

    return {
        "risk": bool(duplicate_percentage > threshold),
        "duplicate_percentage": round(
            duplicate_percentage, 2
        ),
        "threshold": threshold
    }


def detect_outlier_risk(
    df,
    exclude_columns=None,
    threshold=5.0
):
    exclude_columns = exclude_columns or []

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numerical_columns = [
        column for column in numerical_columns
        if column not in exclude_columns
    ]

    risky_columns = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_percentage = (
            (
                (series < lower_bound) |
                (series > upper_bound)
            ).mean() * 100
        )

        if outlier_percentage > threshold:
            risky_columns[column] = round(
                outlier_percentage, 2
            )

    return {
        "risk": bool(risky_columns),
        "threshold": threshold,
        "risky_columns": risky_columns
    }


def detect_target_leakage_indicator(
    df,
    target_column,
    feature_columns=None
):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the DataFrame."
        )

    if feature_columns is None:
        feature_columns = [
            column for column in df.columns
            if column != target_column
        ]

    target_in_features = target_column in feature_columns

    suspicious_columns = [
        column for column in feature_columns
        if column.lower() in {
            target_column.lower(),
            f"{target_column.lower()}_target",
            "target",
            "label"
        }
    ]

    leakage_indicator = (
        target_in_features or bool(suspicious_columns)
    )

    return {
        "risk": bool(leakage_indicator),
        "target_in_features": bool(target_in_features),
        "suspicious_columns": suspicious_columns
    }


def audit_model_risks(
    df,
    target_column,
    feature_columns=None,
    exclude_outlier_columns=None,
    imbalance_threshold=0.80,
    missing_threshold=20,
    duplicate_threshold=5,
    outlier_threshold=5
):
    return {
        "class_imbalance": detect_class_imbalance(
            df,
            target_column,
            threshold=imbalance_threshold
        ),
        "missing_value_risk": detect_missing_value_risk(
            df,
            threshold=missing_threshold
        ),
        "duplicate_risk": detect_duplicate_risk(
            df,
            threshold=duplicate_threshold
        ),
        "outlier_risk": detect_outlier_risk(
            df,
            exclude_columns=exclude_outlier_columns,
            threshold=outlier_threshold
        ),
        "target_leakage": detect_target_leakage_indicator(
            df,
            target_column,
            feature_columns=feature_columns
        )
    }


def detect_overfitting_risk(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold=0.10
):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    score_gap = train_score - test_score

    return {
        "risk": bool(score_gap > threshold),
        "train_score": train_score,
        "test_score": test_score,
        "score_gap": score_gap,
        "threshold": threshold
    }


def audit_model_risks_with_model(
    df,
    target_column,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_columns=None,
    exclude_outlier_columns=None,
    imbalance_threshold=0.80,
    missing_threshold=20,
    duplicate_threshold=5,
    outlier_threshold=5,
    overfitting_threshold=0.10
):
    results = audit_model_risks(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        exclude_outlier_columns=exclude_outlier_columns,
        imbalance_threshold=imbalance_threshold,
        missing_threshold=missing_threshold,
        duplicate_threshold=duplicate_threshold,
        outlier_threshold=outlier_threshold
    )

    results["overfitting"] = detect_overfitting_risk(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=overfitting_threshold
    )

    return results


if __name__ == "__main__":
    sample = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": ["A", "A", "B", "B"]
    })
    print(audit_model_risks(sample, "target"))
