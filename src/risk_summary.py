import pandas


def build_risk_summary(
    df,
    quality_summary,
    model_metrics=None,
    leakage_df=None,
    fairness_result=None
):
    """
    Build a centralized list of ModelGuard findings.

    Findings are screening indicators and should be
    investigated further rather than treated as proof.
    """

    findings = []

    # =====================================================
    # DATA QUALITY
    # =====================================================

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        quality_summary[
            "duplicate_count"
        ]
    )

    if missing_cells > 0:

        findings.append({

            "Category":
                "Data Quality",

            "Risk":
                "Missing Values",

            "Severity":
                "Review",

            "Finding":
                f"{missing_cells} missing cells detected.",

            "Recommendation":
                "Review missingness patterns and determine an appropriate treatment."
        })

    if duplicate_rows > 0:

        findings.append({

            "Category":
                "Data Quality",

            "Risk":
                "Duplicate Rows",

            "Severity":
                "Review",

            "Finding":
                f"{duplicate_rows} duplicate rows detected.",

            "Recommendation":
                "Check whether duplicate records are valid observations or data-entry duplicates."
        })


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    if model_metrics:

        for metric, value in model_metrics.items():

            if value < 0.70:

                findings.append({

                    "Category":
                        "Model Performance",

                    "Risk":
                        metric,

                    "Severity":
                        "Review",

                    "Finding":
                        f"{metric} is {value:.3f}, below the screening threshold of 0.70.",

                    "Recommendation":
                        "Investigate model choice, feature quality, class imbalance, and evaluation strategy."
                })


    # =====================================================
    # LEAKAGE
    # =====================================================

    if leakage_df is not None:

        leakage_count = len(
            leakage_df
        )

        if leakage_count > 0:

            findings.append({

                "Category":
                    "Data Leakage",

                "Risk":
                    "Potential Leakage Indicator",

                "Severity":
                    "Potential Risk",

                "Finding":
                    f"{leakage_count} potential leakage indicator(s) detected.",

                "Recommendation":
                    "Inspect the flagged features and verify that they were available before the prediction target."
            })


    # =====================================================
    # FAIRNESS
    # =====================================================

    if fairness_result:

        max_gap = fairness_result.get(
            "max_gap"
        )

        if (
            max_gap is not None
            and max_gap >= 0.20
        ):

            findings.append({

                "Category":
                    "Group Performance",

                "Risk":
                    "Performance Gap",

                "Severity":
                    "Review",

                "Finding":
                    f"Maximum group-performance gap is {max_gap:.3f}.",

                "Recommendation":
                    "Investigate group-level performance, sample sizes, data representation, and model behavior."
            })


    # =====================================================
    # NO FINDINGS
    # =====================================================

    if not findings:

        findings.append({

            "Category":
                "Overall",

            "Risk":
                "No Major Screening Findings",

            "Severity":
                "Low",

            "Finding":
                "No major screening findings were generated.",

            "Recommendation":
                "Continue with deeper validation before deployment."
        })


    return pandas.DataFrame(
        findings
    )