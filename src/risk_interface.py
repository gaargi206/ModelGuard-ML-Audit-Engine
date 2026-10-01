import pandas


def detect_potential_leakage(
    df,
    target_column,
    correlation_threshold=0.95
):
    """
    Screen for simple indicators of potential
    target leakage.

    These checks do not prove leakage exists.
    """

    leakage_flags = []

    target_name = str(
        target_column
    ).lower()

    target_words = set(
        target_name
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )

    for column in df.columns:

        if column == target_column:
            continue

        column_name = str(
            column
        ).lower()

        column_words = set(
            column_name
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )

        # -----------------------------------------
        # Target-name similarity
        # -----------------------------------------

        if target_words.intersection(
            column_words
        ):

            leakage_flags.append({

                "Feature":
                    column,

                "Indicator":
                    "Target-name similarity",

                "Detail":
                    "Feature name overlaps with target name"
            })

        # -----------------------------------------
        # Very high numeric correlation
        # -----------------------------------------

        if (
            pandas.api.types.is_numeric_dtype(
                df[column]
            )
            and
            pandas.api.types.is_numeric_dtype(
                df[target_column]
            )
        ):

            try:

                correlation = (
                    df[
                        [
                            column,
                            target_column
                        ]
                    ]
                    .corr()
                    .iloc[0, 1]
                )

                if (
                    pandas.notna(
                        correlation
                    )
                    and
                    abs(correlation)
                    >= correlation_threshold
                ):

                    leakage_flags.append({

                        "Feature":
                            column,

                        "Indicator":
                            "Very high target correlation",

                        "Detail":
                            f"Correlation = {correlation:.3f}"
                    })

            except Exception:
                pass

    return pandas.DataFrame(
        leakage_flags
    )


def interpret_leakage_results(
    leakage_df
):
    """
    Provide a simple interpretation of
    potential leakage screening results.
    """

    if leakage_df.empty:

        return (
            "No obvious potential leakage indicators "
            "were detected under the current screening rules."
        )

    return (
        "Potential data leakage indicators were detected. "
        "These results require further investigation and "
        "do not prove that data leakage exists."
    )