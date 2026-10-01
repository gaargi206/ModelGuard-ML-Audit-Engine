
import pandas as pd


def detect_class_imbalance(
    df,
    target_column,
    threshold=0.80
):
    class_counts = df[target_column].value_counts(normalize=True)

    if class_counts.empty:
        return {
            "risk": False,
            "message": "Target column has no values."
        }

    largest_class_ratio = class_counts.max()

    return {
        "risk": bool(largest_class_ratio >= threshold),
        "largest_class_ratio": round(largest_class_ratio, 4),
        "threshold": threshold,
        "class_distribution": class_counts.to_dict()
    }


def detect_missing_value_risk(
    df,
    threshold=20.0
):
    missing_percent = (
        df.isnull().mean() * 100
    )

    risky_columns = missing_percent[
        missing_percent >= threshold
    ]

    return {
        "risk": not risky_columns.empty,
        "threshold": threshold,
        "risky_columns": risky_columns.round(2).to_dict()
    }


def detect_duplicate_risk(
    df,
    threshold=5.0
):
    if len(df) == 0:
        duplicate_percentage = 0
    else:
        duplicate_percentage = (
            df.duplicated().sum() / len(df)
        ) * 100

    return {
        "risk": bool(duplicate_percentage >= threshold),
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
    if exclude_columns is None:
        exclude_columns = []

    numerical_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column not in exclude_columns
    ]

    risky_columns = {}

    for column in numerical_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ).sum()

        outlier_percentage = (
            outlier_count / len(df) * 100
        ) if len(df) > 0 else 0

        if outlier_percentage >= threshold:
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
    if feature_columns is None:
        feature_columns = [
            column
            for column in df.columns
            if column != target_column
        ]

    leakage_indicators = []

    if target_column in feature_columns:
        leakage_indicators.append(
            "Target column included among feature columns."
        )

    return {
        "risk": bool(leakage_indicators),
        "indicators": leakage_indicators
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
        "risk": bool(score_gap >= threshold),
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "score_gap": round(score_gap, 4),
        "threshold": threshold
    }


def audit_model_risks(
    df,
    target_column,
    feature_columns=None,
    exclude_outlier_columns=None,
    imbalance_threshold=0.80,
    missing_threshold=20.0,
    duplicate_threshold=5.0,
    outlier_threshold=5.0
):
    return {
        "class_imbalance": detect_class_imbalance(
            df,
            target_column,
            threshold=imbalance_threshold
        ),
        "missing_values": detect_missing_value_risk(
            df,
            threshold=missing_threshold
        ),
        "duplicates": detect_duplicate_risk(
            df,
            threshold=duplicate_threshold
        ),
        "outliers": detect_outlier_risk(
            df,
            exclude_columns=exclude_outlier_columns,
            threshold=outlier_threshold
        ),
        "target_leakage": detect_target_leakage_indicator(
            df,
            target_column,
            feature_columns
        )
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
    missing_threshold=20.0,
    duplicate_threshold=5.0,
    outlier_threshold=5.0,
    overfitting_threshold=0.10
):
    risks = audit_model_risks(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        exclude_outlier_columns=exclude_outlier_columns,
        imbalance_threshold=imbalance_threshold,
        missing_threshold=missing_threshold,
        duplicate_threshold=duplicate_threshold,
        outlier_threshold=outlier_threshold
    )

    risks["overfitting"] = detect_overfitting_risk(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=overfitting_threshold
    )

    return risks
