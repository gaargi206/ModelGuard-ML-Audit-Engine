import pandas as pd


def get_data_quality_summary(df):
    """
    Generate a complete data-quality summary.
    """

    missing_data = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_table = pd.DataFrame({
        "Feature": missing_data.index,
        "Missing Values": missing_data.values,
        "Missing %": (
            missing_data.values
            / max(len(df), 1)
            * 100
        ).round(2)
    })

    duplicate_count = int(
        df.duplicated().sum()
    )

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=True
        ) <= 1
    ]

    high_cardinality_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=True
        ) > 50
    ]

    return {
        "missing_table": missing_table,
        "duplicate_count": duplicate_count,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns
    }


def get_feature_risk_flags(df):
    """
    Identify potential feature-level data-quality risks.
    """

    feature_risks = []

    for column in df.columns:

        unique_count = (
            df[column]
            .nunique(dropna=True)
        )

        missing_percentage = (
            df[column]
            .isna()
            .mean()
            * 100
        )

        if missing_percentage >= 30:

            feature_risks.append({
                "Feature": column,
                "Risk": "High Missingness",
                "Detail": (
                    f"{missing_percentage:.1f}% missing"
                )
            })

        if unique_count <= 1:

            feature_risks.append({
                "Feature": column,
                "Risk": "Constant Feature",
                "Detail": "Only one unique value"
            })

        if unique_count > 50:

            feature_risks.append({
                "Feature": column,
                "Risk": "High Cardinality",
                "Detail": (
                    f"{unique_count} unique values"
                )
            })

        if (
            unique_count >= 0.95 * len(df)
            and len(df) > 10
        ):

            feature_risks.append({
                "Feature": column,
                "Risk": "Potential ID-Like Feature",
                "Detail": "Very high uniqueness"
            })

    return pd.DataFrame(
        feature_risks
    )


def detect_outliers(df):
    """
    Detect potential numeric outliers using IQR.
    """

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    outlier_results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) <= 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (series < lower_bound)
            |
            (series > upper_bound)
        ).sum()

        outlier_percentage = (
            outliers
            / len(series)
            * 100
        )

        outlier_results.append({
            "Feature": column,
            "Outliers": int(outliers),
            "Outlier %": round(
                outlier_percentage,
                2
            )
        })

    return pd.DataFrame(
        outlier_results
    )


def detect_strong_correlations(
    df,
    threshold=0.80
):
    """
    Identify pairs of numeric features
    with strong correlations.
    """

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if len(numeric_columns) < 2:

        return pd.DataFrame(
            columns=[
                "Feature 1",
                "Feature 2",
                "Correlation"
            ]
        )

    correlation_matrix = df[
        numeric_columns
    ].corr()

    strong_correlations = []

    for i in range(
        len(numeric_columns)
    ):

        for j in range(
            i + 1,
            len(numeric_columns)
        ):

            correlation = (
                correlation_matrix.iloc[
                    i,
                    j
                ]
            )

            if abs(correlation) >= threshold:

                strong_correlations.append({
                    "Feature 1":
                        numeric_columns[i],

                    "Feature 2":
                        numeric_columns[j],

                    "Correlation":
                        round(
                            correlation,
                            3
                        )
                })

    return pd.DataFrame(
        strong_correlations
    )


def calculate_audit_score(df):
    """
    Calculate the preliminary ModelGuard
    data-quality screening score.
    """

    total_cells = (
        df.shape[0]
        *
        df.shape[1]
    )

    if total_cells == 0:
        return 0

    missing_ratio = (
        df.isna().sum().sum()
        /
        total_cells
    )

    duplicate_ratio = (
        df.duplicated().sum()
        /
        max(df.shape[0], 1)
    )

    score = 100

    score -= min(
        missing_ratio * 100,
        40
    )

    score -= min(
        duplicate_ratio * 100,
        20
    )

    return max(
        0,
        round(score, 1)
    )