import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def build_baseline_model(X):
    """
    Build the reproducible Logistic Regression pipeline.
    """

    numeric_features = (
        X.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return model


def evaluate_baseline_model(
    df,
    target_column,
    test_size=0.20,
    random_state=42
):
    """
    Train and evaluate the baseline model.
    """

    model_df = df.dropna(
        subset=[target_column]
    ).copy()

    X = model_df.drop(
        columns=[target_column]
    )

    y = model_df[
        target_column
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    model = build_baseline_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    metrics = {
        "Accuracy": round(
            accuracy_score(
                y_test,
                predictions
            ),
            4
        ),
        "Precision": round(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            4
        ),
        "Recall": round(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            4
        ),
        "F1 Score": round(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            4
        )
    }

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "y_test": y_test,
        "predictions": predictions
    }


def identify_model_risks(
    metrics,
    threshold=0.70
):
    """
    Identify metrics below the configured
    screening threshold.
    """

    risk_flags = []

    for metric_name, value in metrics.items():

        if value < threshold:

            risk_flags.append({
                "Metric": metric_name,
                "Value": value,
                "Risk": (
                    "Below screening threshold"
                )
            })

    return pd.DataFrame(
        risk_flags
    )