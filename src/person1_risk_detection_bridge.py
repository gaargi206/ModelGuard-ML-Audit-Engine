import pandas as pd

from sklearn.model_selection import train_test_split

from src.risk_detection import (
    audit_model_risks_with_model
)


def run_person1_risk_detection(
    df,
    target_column,
    model_result,
    test_size=0.20,
    random_state=42,
    exclude_outlier_columns=None
):
    """
    Run Person 1's Risk Detection engine using
    the existing ModelGuard baseline model.

    The Person 1 engine checks for:
    - class imbalance
    - missing values
    - duplicate rows
    - outliers
    - potential target leakage
    - overfitting
    """

    if df is None:
        raise ValueError(
            "Dataset cannot be None."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    if model_result is None:
        raise ValueError(
            "A completed model evaluation is required "
            "before running Risk Detection."
        )

    model = model_result.get("model")

    if model is None:
        raise ValueError(
            "The evaluated model was not found."
        )

    model_df = df.dropna(
        subset=[target_column]
    ).copy()

    X = model_df.drop(
        columns=[target_column]
    )

    y = model_df[target_column]

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    feature_columns = X.columns.tolist()

    if exclude_outlier_columns is None:
        exclude_outlier_columns = (
            X.select_dtypes(
                exclude="number"
            ).columns.tolist()
        )

    risk_result = audit_model_risks_with_model(
        df=df,
        target_column=target_column,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_columns,
        exclude_outlier_columns=exclude_outlier_columns
    )

    return {
        "raw": risk_result,
        "feature_columns": feature_columns,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0])
    }


def risk_result_to_dataframe(
    risk_result
):
    """
    Convert Person 1's risk results into a
    display-friendly DataFrame.

    Handles dictionary-based risk results
    without assuming a single internal structure.
    """

    if risk_result is None:
        return pd.DataFrame(
            columns=[
                "Risk Category",
                "Details"
            ]
        )

    rows = []

    for category, details in risk_result.items():

        if isinstance(details, pd.DataFrame):

            if details.empty:
                rows.append({
                    "Risk Category": category,
                    "Details": "No findings."
                })

            else:
                for _, row in details.iterrows():
                    rows.append({
                        "Risk Category": category,
                        "Details": " | ".join(
                            f"{key}: {value}"
                            for key, value
                            in row.to_dict().items()
                        )
                    })

        elif isinstance(details, dict):

            for key, value in details.items():

                if isinstance(value, (list, tuple)):
                    value = ", ".join(
                        str(item)
                        for item in value
                    )

                rows.append({
                    "Risk Category": (
                        f"{category} — {key}"
                    ),
                    "Details": value
                })

        elif isinstance(details, (list, tuple)):

            rows.append({
                "Risk Category": category,
                "Details": ", ".join(
                    str(item)
                    for item in details
                )
            })

        else:

            rows.append({
                "Risk Category": category,
                "Details": details
            })

    return pd.DataFrame(rows)
