
import pandas as pd


def analyze_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    return pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage.round(2)
    })


def analyze_duplicates(df):
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (
        duplicate_count / len(df) * 100
        if len(df) > 0 else 0
    )

    return {
        "duplicate_count": int(duplicate_count),
        "duplicate_percentage": round(duplicate_percentage, 2)
    }


def analyze_dataset_structure(df):
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "numeric_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "numeric_summary": df[numerical_columns].describe().to_dict()
        if numerical_columns else {}
    }


def analyze_target_distribution(df, target_column):
    counts = df[target_column].value_counts()
    percentages = (
        df[target_column]
        .value_counts(normalize=True) * 100
    )

    return pd.DataFrame({
        "class_count": counts,
        "class_percentage": percentages.round(2)
    })


def analyze_outliers(df, exclude_columns=None):
    """
    Detect potential outliers in numerical columns using IQR.

    exclude_columns:
        Optional list of columns that should not be treated
        as continuous numerical variables.
    """

    if exclude_columns is None:
        exclude_columns = []

    numerical_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column not in exclude_columns
    ]

    results = []

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
            if len(df) > 0 else 0
        )

        results.append({
            "column": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(outlier_count),
            "outlier_percentage": round(outlier_percentage, 2)
        })

    return pd.DataFrame(results)


def audit_data_quality(
    df,
    target_column,
    exclude_outlier_columns=None
):
    """
    Run the complete data quality audit.

    exclude_outlier_columns:
        Optional list of columns excluded from IQR
        outlier analysis.
    """

    return {
        "structure": analyze_dataset_structure(df),
        "missing_values": analyze_missing_values(df),
        "duplicates": analyze_duplicates(df),
        "target_distribution": analyze_target_distribution(
            df,
            target_column
        ),
        "outliers": analyze_outliers(
            df,
            exclude_columns=exclude_outlier_columns
        )
    }
