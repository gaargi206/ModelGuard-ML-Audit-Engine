import pandas


def get_risk_status(
    audit_score,
    missing_cells,
    duplicate_rows,
    leakage_count=0,
    fairness_gap=None
):
    """
    Generate screening-level risk statuses.

    These statuses are indicators for investigation,
    not definitive conclusions.
    """

    statuses = {}

    # ---------------------------------------------
    # Data Quality
    # ---------------------------------------------

    if audit_score >= 80:

        statuses["Data Quality"] = {
            "Status": "Low",
            "Reason": "No major missingness or duplication issue detected."
        }

    elif audit_score >= 60:

        statuses["Data Quality"] = {
            "Status": "Review",
            "Reason": "Some data-quality issues may require review."
        }

    else:

        statuses["Data Quality"] = {
            "Status": "Potential Risk",
            "Reason": "Significant data-quality indicators detected."
        }

    # ---------------------------------------------
    # Leakage
    # ---------------------------------------------

    if leakage_count == 0:

        statuses["Leakage"] = {
            "Status": "Low",
            "Reason": "No obvious leakage indicators detected."
        }

    else:

        statuses["Leakage"] = {
            "Status": "Potential Risk",
            "Reason": (
                f"{leakage_count} potential leakage indicator(s) detected."
            )
        }

    # ---------------------------------------------
    # Fairness
    # ---------------------------------------------

    if fairness_gap is None:

        statuses["Fairness"] = {
            "Status": "Not Evaluated",
            "Reason": "No group-performance audit has been run."
        }

    elif fairness_gap >= 0.20:

        statuses["Fairness"] = {
            "Status": "Review",
            "Reason": (
                "A relatively large group-performance gap "
                "was detected under the screening threshold."
            )
        }

    else:

        statuses["Fairness"] = {
            "Status": "Low",
            "Reason": (
                "No large group-performance gap was detected "
                "under the current threshold."
            )
        }

    # ---------------------------------------------
    # Overall
    # ---------------------------------------------

    risk_values = [
        item["Status"]
        for item in statuses.values()
        if item["Status"] != "Not Evaluated"
    ]

    if "Potential Risk" in risk_values:

        overall_status = "Potential Risk"

    elif "Review" in risk_values:

        overall_status = "Review"

    else:

        overall_status = "Low"

    statuses["Overall"] = {
        "Status": overall_status,
        "Reason": (
            "Overall screening status based on "
            "the currently available audit indicators."
        )
    }

    return statuses


def generate_audit_summary(
    df,
    target_column,
    audit_score,
    quality_summary,
    model_metrics=None,
    leakage_df=None,
    fairness_result=None
):
    """
    Combine ModelGuard audit results into a single
    structured summary.
    """

    leakage_count = 0

    if leakage_df is not None:

        leakage_count = len(
            leakage_df
        )

    fairness_gap = None

    if fairness_result:

        fairness_gap = fairness_result.get(
            "max_gap"
        )

    statuses = get_risk_status(
        audit_score=audit_score,
        missing_cells=int(
            df.isna().sum().sum()
        ),
        duplicate_rows=int(
            quality_summary[
                "duplicate_count"
            ]
        ),
        leakage_count=leakage_count,
        fairness_gap=fairness_gap
    )

    summary = {

        "Dataset": {
            "Rows": int(
                df.shape[0]
            ),

            "Columns": int(
                df.shape[1]
            ),

            "Target": target_column
        },

        "Data Quality": {
            "Audit Score": audit_score,

            "Missing Cells": int(
                df.isna().sum().sum()
            ),

            "Duplicate Rows": int(
                quality_summary[
                    "duplicate_count"
                ]
            )
        },

        "Model Performance": {},

        "Leakage": {
            "Potential Indicators":
                leakage_count
        },

        "Fairness": {},

        "Risk Status": {

            section: values["Status"]

            for section, values
            in statuses.items()
        }
    }

    # ---------------------------------------------
    # Model Performance
    # ---------------------------------------------

    if model_metrics:

        summary[
            "Model Performance"
        ] = {

            key: float(value)

            for key, value
            in model_metrics.items()
        }

    # ---------------------------------------------
    # Fairness
    # ---------------------------------------------

    if fairness_gap is not None:

        summary[
            "Fairness"
        ] = {

            "Maximum Group Gap":
                float(fairness_gap)
        }

    return summary


def summary_to_dataframe(
    summary
):
    """
    Convert the structured audit summary
    into a simple table for the UI.
    """

    rows = []

    for section, values in summary.items():

        if isinstance(values, dict):

            for key, value in values.items():

                rows.append({

                    "Section":
                        section,

                    "Metric":
                        key,

                    "Value":
                        value
                })

    return pandas.DataFrame(
        rows
    )