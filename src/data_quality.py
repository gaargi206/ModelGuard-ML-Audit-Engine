import numpy as np
import pandas as pd


def analyze_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)

    result = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    return result[result["missing_count"] > 0].sort_values(
        "missing_percentage", ascending=False
    )


def analyze_duplicates(df):
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round(
        duplicate_count / len(df) * 100, 2
    ) if len(df) else 0.0

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage
    }


def analyze_dataset_structure(df):
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns
    }


def analyze_target_distribution(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the DataFrame."
        )

    counts = df[target_column].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(2)

    return pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })


def analyze_outliers(df, exclude_columns=None):
    exclude_columns = exclude_columns or []

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numerical_columns = [
        column for column in numerical_columns
        if column not in exclude_columns
    ]

    outlier_results = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )

        outlier_percentage = round(
            outlier_count / len(series) * 100, 2
        )

        outlier_results[column] = {
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage
        }

    return outlier_results


def audit_data_quality(
    df,
    target_column,
    exclude_outlier_columns=None
):
    missing_results = analyze_missing_values(df)
    duplicate_results = analyze_duplicates(df)
    structure_results = analyze_dataset_structure(df)
    target_results = analyze_target_distribution(
        df, target_column
    )
    outlier_results = analyze_outliers(
        df,
        exclude_columns=exclude_outlier_columns
    )

    return {
        "missing_values": missing_results,
        "duplicates": duplicate_results,
        "structure": structure_results,
        "target_distribution": target_results,
        "outliers": outlier_results
    }


if __name__ == "__main__":
    sample = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": ["A", "A", "B", "B"]
    })
    print(audit_data_quality(sample, "target"))
