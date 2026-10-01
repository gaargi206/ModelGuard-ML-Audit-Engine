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
    f1_score
)


def get_group_columns(
    df,
    target_column,
    min_groups=2,
    max_groups=20
):
    """
    Identify columns suitable for group-level analysis.
    """

    group_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        if (
            min_groups
            <= unique_count
            <= max_groups
        ):

            group_columns.append(
                column
            )

    return group_columns


def build_fairness_model(X):

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


def run_group_performance_audit(
    df,
    target_column,
    group_column,
    test_size=0.20,
    random_state=42
):
    """
    Train a baseline model without using the selected
    group column as a model feature, then evaluate
    performance across groups.
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

    groups = model_df[
        group_column
    ]

    # Remove the selected group attribute
    # from model features.
    X_model = X.drop(
        columns=[group_column]
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        g_train,
        g_test
    ) = train_test_split(
        X_model,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = build_fairness_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    group_results = []

    for group in (
        g_test
        .dropna()
        .unique()
    ):

        mask = (
            g_test == group
        )

        if mask.sum() < 2:
            continue

        actual_group = y_test[
            mask
        ]

        predicted_group = predictions[
            mask
        ]

        group_results.append({

            "Group":
                group,

            "Samples":
                int(mask.sum()),

            "Accuracy":
                round(
                    accuracy_score(
                        actual_group,
                        predicted_group
                    ),
                    3
                ),

            "Precision":
                round(
                    precision_score(
                        actual_group,
                        predicted_group,
                        average="weighted",
                        zero_division=0
                    ),
                    3
                ),

            "Recall":
                round(
                    recall_score(
                        actual_group,
                        predicted_group,
                        average="weighted",
                        zero_division=0
                    ),
                    3
                ),

            "F1":
                round(
                    f1_score(
                        actual_group,
                        predicted_group,
                        average="weighted",
                        zero_division=0
                    ),
                    3
                )
        })

    group_df = pd.DataFrame(
        group_results
    )

    if group_df.empty:

        return {
            "group_results": group_df,
            "gap_results": pd.DataFrame(),
            "max_gap": None
        }

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    gaps = []

    for metric in metrics:

        gap = (
            group_df[metric].max()
            -
            group_df[metric].min()
        )

        gaps.append({

            "Metric":
                metric,

            "Maximum Group Gap":
                round(
                    gap,
                    3
                )
        })

    gap_df = pd.DataFrame(
        gaps
    )

    max_gap = max(
        gap_df["Maximum Group Gap"]
    )

    return {
        "group_results": group_df,
        "gap_results": gap_df,
        "max_gap": max_gap
    }


def interpret_group_gap(
    max_gap,
    threshold=0.20
):
    """
    Interpret group-performance gaps using
    a configurable screening threshold.
    """

    if max_gap is None:

        return (
            "Not enough observations per group "
            "for group-level evaluation."
        )

    if max_gap >= threshold:

        return (
            "Potential group-performance disparity "
            "detected. Further fairness investigation "
            "is recommended."
        )

    return (
        "No large group-performance gap was detected "
        "under the current screening threshold."
    )