import pandas as pd

from src.data_quality import audit_data_quality


def run_person1_data_quality_audit(
    df,
    target_column,
    exclude_outlier_columns=None
):
    """
    Run Person 1's Data Quality Engine.
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

    audit_result = audit_data_quality(
        df=df,
        target_column=target_column,
        exclude_outlier_columns=exclude_outlier_columns
    )

    missing_values = audit_result.get(
        "missing_values",
        pd.DataFrame()
    )

    duplicates = audit_result.get(
        "duplicates",
        {}
    )

    structure = audit_result.get(
        "structure",
        {}
    )

    target_distribution = audit_result.get(
        "target_distribution",
        pd.DataFrame()
    )

    outliers = audit_result.get(
        "outliers",
        {}
    )

    return {
        "raw": audit_result,

        "rows": structure.get(
            "rows",
            int(df.shape[0])
        ),

        "columns": structure.get(
            "columns",
            int(df.shape[1])
        ),

        "numeric_columns": structure.get(
            "numerical_columns",
            []
        ),

        "categorical_columns": structure.get(
            "categorical_columns",
            []
        ),

        "missing_values": missing_values,

        "missing_cells": int(
            df.isna().sum().sum()
        ),

        "duplicates": duplicates,

        "duplicate_count": int(
            duplicates.get(
                "duplicate_count",
                df.duplicated().sum()
            )
        ),

        "duplicate_percentage": float(
            duplicates.get(
                "duplicate_percentage",
                0
            )
        ),

        "target_distribution": target_distribution,

        "outliers": outliers
    }


def outliers_to_dataframe(
    outlier_results
):
    """
    Convert Person 1 outlier results
    into a DataFrame.
    """

    rows = []

    if not outlier_results:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Outliers",
                "Outlier %"
            ]
        )

    for feature, values in outlier_results.items():

        rows.append({
            "Feature": feature,
            "Outliers": int(
                values.get(
                    "outlier_count",
                    0
                )
            ),
            "Outlier %": float(
                values.get(
                    "outlier_percentage",
                    0
                )
            )
        })

    return pd.DataFrame(rows)


def missing_values_to_dataframe(
    missing_values
):
    """
    Convert Person 1 missing-value
    results into a display DataFrame.
    """

    if missing_values is None:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Missing Values",
                "Missing %"
            ]
        )

    if missing_values.empty:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Missing Values",
                "Missing %"
            ]
        )

    result = missing_values.reset_index()

    result.columns = [
        "Feature",
        "Missing Values",
        "Missing %"
    ]

    return result


def target_distribution_to_dataframe(
    target_distribution
):
    """
    Convert Person 1 target distribution
    into a display DataFrame.
    """

    if target_distribution is None:
        return pd.DataFrame(
            columns=[
                "Class",
                "Count",
                "Percentage"
            ]
        )

    if target_distribution.empty:
        return pd.DataFrame(
            columns=[
                "Class",
                "Count",
                "Percentage"
            ]
        )

    result = target_distribution.reset_index()

    result.columns = [
        "Class",
        "Count",
        "Percentage"
    ]

    return result
