import pandas


def validate_dataset(
    df,
    target_column
):
    """
    Validate a dataset before running
    the ModelGuard audit pipeline.
    """

    issues = []
    warnings = []

    # =====================================================
    # EMPTY DATASET
    # =====================================================

    if df is None:

        issues.append(
            "No dataset was provided."
        )

        return {
            "valid": False,
            "issues": issues,
            "warnings": warnings
        }


    # =====================================================
    # EMPTY DATAFRAME
    # =====================================================

    if df.empty:

        issues.append(
            "The dataset contains no rows."
        )


    # =====================================================
    # TARGET COLUMN
    # =====================================================

    if target_column not in df.columns:

        issues.append(
            "The selected target column does not exist "
            "in the dataset."
        )

        return {
            "valid": False,
            "issues": issues,
            "warnings": warnings
        }


    # =====================================================
    # TARGET MISSINGNESS
    # =====================================================

    target_missing = int(
        df[target_column]
        .isna()
        .sum()
    )

    if target_missing > 0:

        warnings.append(
            f"The target column contains "
            f"{target_missing} missing value(s)."
        )


    # =====================================================
    # TARGET UNIQUE VALUES
    # =====================================================

    target_unique = (
        df[target_column]
        .nunique(
            dropna=True
        )
    )

    if target_unique < 2:

        issues.append(
            "The target column contains fewer than "
            "two valid classes."
        )


    # =====================================================
    # TARGET CARDINALITY
    # =====================================================

    if target_unique > 50:

        warnings.append(
            "The target column contains more than "
            "50 unique values. Classification may not "
            "be appropriate."
        )


    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        warnings.append(
            f"The dataset contains "
            f"{duplicate_count} duplicate row(s)."
        )


    # =====================================================
    # MISSINGNESS
    # =====================================================

    missing_percentage = (
        df.isna()
        .sum()
        .sum()
        /
        max(
            df.shape[0] * df.shape[1],
            1
        )
        *
        100
    )

    if missing_percentage >= 30:

        warnings.append(
            f"Approximately "
            f"{missing_percentage:.1f}% of dataset cells "
            f"are missing."
        )


    # =====================================================
    # CONSTANT FEATURES
    # =====================================================

    constant_columns = [

        column

        for column in df.columns

        if (
            column != target_column
            and
            df[column]
            .nunique(
                dropna=True
            )
            <= 1
        )
    ]

    if constant_columns:

        warnings.append(
            "Constant feature(s) detected: "
            +
            ", ".join(
                constant_columns
            )
        )


    # =====================================================
    # TARGET CLASS DISTRIBUTION
    # =====================================================

    class_distribution = (
        df[target_column]
        .value_counts(
            normalize=True,
            dropna=True
        )
        *
        100
    )

    if not class_distribution.empty:

        largest_class_percentage = (
            class_distribution.max()
        )

        if largest_class_percentage >= 90:

            warnings.append(
                "The target distribution is highly "
                "imbalanced; the largest class represents "
                f"{largest_class_percentage:.1f}% of valid targets."
            )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "valid":
            len(issues) == 0,

        "issues":
            issues,

        "warnings":
            warnings
    }


def get_validation_summary(
    validation_result
):
    """
    Create a compact validation summary.
    """

    if validation_result["valid"]:

        status = "Valid"

    else:

        status = "Invalid"


    return {

        "Status":
            status,

        "Issues":
            len(
                validation_result[
                    "issues"
                ]
            ),

        "Warnings":
            len(
                validation_result[
                    "warnings"
                ]
            )
    }