import pandas


def load_csv_file(uploaded_file):
    """
    Load an uploaded CSV file into a pandas DataFrame.
    """

    if uploaded_file is None:

        return None

    try:

        df = pandas.read_csv(
            uploaded_file
        )

        return df

    except Exception as e:

        raise ValueError(
            f"Could not read the uploaded CSV file: {e}"
        )


def clean_dataset(df):
    """
    Perform basic non-destructive cleaning.

    This does not remove rows or columns.
    It only standardizes column names and text values.
    """

    df = df.copy()

    # Clean column names

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Clean object/string columns

    for column in df.select_dtypes(
        include="object"
    ).columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Convert common missing-value markers

    missing_markers = [
        "?",
        "NA",
        "N/A",
        "na",
        "null",
        "NULL",
        "None"
    ]

    df = df.replace(
        missing_markers,
        pandas.NA
    )

    return df


def get_dataset_profile(df):
    """
    Return basic information about a dataset.
    """

    return {
        "Rows": int(
            df.shape[0]
        ),

        "Columns": int(
            df.shape[1]
        ),

        "Numeric Columns": len(
            df.select_dtypes(
                include="number"
            ).columns
        ),

        "Categorical Columns": len(
            df.select_dtypes(
                exclude="number"
            ).columns
        ),

        "Missing Cells": int(
            df.isna().sum().sum()
        ),

        "Duplicate Rows": int(
            df.duplicated().sum()
        )
    }