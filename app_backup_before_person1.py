import streamlit as st
import pandas

from src.report import generate_audit_report

from src.data_quality_interface import (
    get_data_quality_summary,
    get_feature_risk_flags,
    detect_outliers,
    detect_strong_correlations,
    calculate_audit_score
)

from src.model_interface import (
    evaluate_baseline_model,
    identify_model_risks
)

from src.fairness_interface import (
    get_group_columns,
    run_group_performance_audit,
    interpret_group_gap
)

from src.risk_interface import (
    detect_potential_leakage,
    interpret_leakage_results
)

from src.reproducibility import (
    get_reproducibility_info,
    get_model_configuration
)

from src.audit_summary import (
    generate_audit_summary,
    summary_to_dataframe
)

from src.risk_summary import (
    build_risk_summary
)

from src.dataset_loader import (
    load_csv_file,
    clean_dataset
)

from src.dataset_validation import (
    validate_dataset,
    get_validation_summary
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ModelGuard — ML Model Risk Auditor",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ ModelGuard")
st.caption(
    "ML Model Risk Auditor — Data Quality • Model Performance • "
    "Risk Detection • Fairness • Leakage • Reproducibility"
)


# ============================================================
# SESSION STATE
# ============================================================

if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = None

if "fairness_result" not in st.session_state:
    st.session_state["fairness_result"] = None

if "leakage_df" not in st.session_state:
    st.session_state["leakage_df"] = None

if "validation_result" not in st.session_state:
    st.session_state["validation_result"] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ ModelGuard")

st.sidebar.subheader("Dataset Source")

dataset_source = st.sidebar.radio(
    "Choose dataset source",
    [
        "Built-in Dataset",
        "Upload Custom CSV"
    ]
)


# ============================================================
# DATASET LOADING
# ============================================================

df = None
dataset_name = None
target_column = None


# ------------------------------------------------------------
# BUILT-IN DATASETS
# ------------------------------------------------------------

if dataset_source == "Built-in Dataset":

    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        [
            "Adult Census Income",
            "Online Shoppers Purchasing Intention",
            "Student Dropout & Academic Success"
        ]
    )

    # --------------------------------------------------------
    # ADULT DATASET
    # --------------------------------------------------------

    if dataset_option == "Adult Census Income":

        dataset_name = "Adult Census Income"

        columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income"
        ]

        try:

            df = pandas.read_csv(
                "data/adult.csv",
                header=None,
                names=columns,
                skipinitialspace=True
            )

            # Remove accidental header row
            if len(df) > 0:

                first_row = (
                    df.iloc[0]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                if first_row["income"] == "income":
                    df = df.iloc[1:].reset_index(drop=True)

            df = df.replace("?", pandas.NA)

            target_column = "income"

        except Exception as e:

            st.error(
                f"Could not load Adult dataset: {e}"
            )
            st.stop()


    # --------------------------------------------------------
    # ONLINE SHOPPERS DATASET
    # --------------------------------------------------------

    elif dataset_option == "Online Shoppers Purchasing Intention":

        dataset_name = "Online Shoppers Purchasing Intention"

        try:

            df = pandas.read_csv(
                "data/online_shoppers_intention.csv"
            )

            target_column = "Revenue"

        except Exception as e:

            st.error(
                f"Could not load Online Shoppers dataset: {e}"
            )
            st.stop()


    # --------------------------------------------------------
    # STUDENT DROPOUT DATASET
    # --------------------------------------------------------

    elif dataset_option == "Student Dropout & Academic Success":

        dataset_name = "Student Dropout & Academic Success"

        try:

            df = pandas.read_csv(
                "data/student_dropout.csv",
                sep=";"
            )

            target_column = "Target"

        except Exception as e:

            st.error(
                f"Could not load Student Dropout dataset: {e}"
            )
            st.stop()


# ------------------------------------------------------------
# CUSTOM CSV
# ------------------------------------------------------------

else:

    dataset_name = "Custom CSV Dataset"

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "👈 Upload a CSV file from the sidebar to begin the audit."
        )

        st.markdown(
            """
            ### What ModelGuard checks

            - 🧹 Data Quality
            - 🤖 Model Performance
            - ⚠️ Risk Indicators
            - ⚖️ Group Performance
            - 🔐 Potential Data Leakage
            - 🔁 Reproducibility
            - 📄 Audit Report
            """
        )

        st.stop()

    try:

        df = load_csv_file(uploaded_file)
        df = clean_dataset(df)

    except Exception as e:

        st.error(str(e))
        st.stop()

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns.tolist()
    )


# ============================================================
# COMMON DATA CLEANING
# ============================================================

if df is None:

    st.error("No dataset could be loaded.")
    st.stop()


# Remove whitespace from column names

df.columns = [
    str(column).strip()
    for column in df.columns
]


# Remove common missing markers

df = df.replace(
    [
        "?",
        "NA",
        "N/A",
        "na",
        "null",
        "NULL",
        "None"
    ],
    pandas.NA
)


# Remove rows with missing target

df = df.dropna(
    subset=[target_column]
).reset_index(drop=True)


# Clean target values

if df[target_column].dtype == "object":

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.strip()
    )

    target_values_to_remove = [
        "income",
        "revenue",
        "target"
    ]

    df = df[
        ~df[target_column]
        .str.lower()
        .isin(target_values_to_remove)
    ].reset_index(drop=True)


# ============================================================
# DATASET VALIDATION
# ============================================================

validation_result = validate_dataset(
    df,
    target_column
)

validation_summary = get_validation_summary(
    validation_result
)

st.session_state["validation_result"] = validation_result


# ============================================================
# SIDEBAR VALIDATION STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("Dataset Validation")

if validation_result["valid"]:

    st.sidebar.success("✅ Dataset Valid")

else:

    st.sidebar.error("❌ Dataset Invalid")


if validation_result["warnings"]:

    st.sidebar.warning(
        f"⚠️ {len(validation_result['warnings'])} warning(s)"
    )


# ============================================================
# INVALID DATASET
# ============================================================

if not validation_result["valid"]:

    st.error(
        "❌ ModelGuard cannot continue because the dataset "
        "failed validation."
    )

    st.subheader("Validation Errors")

    for issue in validation_result["issues"]:

        st.error(
            f"• {issue}"
        )

    if validation_result["warnings"]:

        st.subheader("Warnings")

        for warning in validation_result["warnings"]:

            st.warning(
                f"• {warning}"
            )

    st.stop()


# ============================================================
# VALIDATION WARNINGS
# ============================================================

if validation_result["warnings"]:

    with st.expander(
        "⚠️ Dataset Validation Warnings",
        expanded=False
    ):

        for warning in validation_result["warnings"]:

            st.warning(
                warning
            )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🧠 Audit Summary",
        "🚨 Risk Findings",
        "🧹 Data Quality",
        "🤖 Model Evaluation",
        "⚖️ Fairness",
        "🔐 Leakage",
        "🔁 Reproducibility",
        "📄 Report"
    ]
)


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

quality_summary = get_data_quality_summary(df)

audit_score = calculate_audit_score(df)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Cells",
            int(df.isna().sum().sum())
        )

    with col4:

        st.metric(
            "Audit Score",
            f"{audit_score}/100"
        )

    st.divider()

    st.subheader("Dataset Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    with profile_col1:

        st.write(
            "**Dataset:**",
            dataset_name
        )

        st.write(
            "**Target:**",
            target_column
        )

    with profile_col2:

        st.write(
            "**Numeric Columns:**",
            len(
                df.select_dtypes(
                    include="number"
                ).columns
            )
        )

        st.write(
            "**Categorical Columns:**",
            len(
                df.select_dtypes(
                    exclude="number"
                ).columns
            )
        )

    with profile_col3:

        st.write(
            "**Duplicate Rows:**",
            int(df.duplicated().sum())
        )

        st.write(
            "**Validation Status:**",
            validation_summary["Status"]
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.divider()

    st.subheader("Target Distribution")

    target_distribution = (
        df[target_column]
        .value_counts()
        .reset_index()
    )

    target_distribution.columns = [
        "Class",
        "Count"
    ]

    st.dataframe(
        target_distribution,
        width="stretch"
    )


# ============================================================
# AUDIT SUMMARY
# ============================================================

elif page == "🧠 Audit Summary":

    st.header("🧠 Audit Summary")

    st.write(
        "ModelGuard combines the available audit indicators "
        "into a consolidated screening summary."
    )

    # --------------------------------------------------------
    # Leakage scan
    # --------------------------------------------------------

    try:

        leakage_df = detect_potential_leakage(
            df,
            target_column
        )

        st.session_state["leakage_df"] = leakage_df

    except Exception:

        leakage_df = pandas.DataFrame()

        st.session_state["leakage_df"] = leakage_df


    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = generate_audit_summary(
        df=df,
        target_column=target_column,
        audit_score=audit_score,
        quality_summary=quality_summary,
        model_metrics=st.session_state.get(
            "model_metrics"
        ),
        leakage_df=leakage_df,
        fairness_result=st.session_state.get(
            "fairness_result"
        )
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    risk_status = summary.get(
        "Risk Status",
        {}
    )

    st.subheader("Risk Status")

    status_cols = st.columns(4)

    sections = [
        "Overall",
        "Data Quality",
        "Leakage",
        "Fairness"
    ]

    for index, section in enumerate(sections):

        with status_cols[index]:

            status = risk_status.get(
                section,
                "Not Evaluated"
            )

            st.metric(
                section,
                status
            )

    st.divider()

    # --------------------------------------------------------
    # Key Metrics
    # --------------------------------------------------------

    st.subheader("Key Audit Metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.metric(
            "Audit Score",
            f"{audit_score}/100"
        )

    with metric_col2:

        st.metric(
            "Missing Cells",
            int(df.isna().sum().sum())
        )

    with metric_col3:

        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    with metric_col4:

        st.metric(
            "Potential Leakage Indicators",
            len(leakage_df)
        )

    st.divider()

    # --------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------

    st.subheader("Audit Summary Table")

    summary_df = summary_to_dataframe(
        summary
    )

    st.dataframe(
        summary_df,
        width="stretch"
    )


# ============================================================
# RISK FINDINGS
# ============================================================

elif page == "🚨 Risk Findings":

    st.header("🚨 Risk Findings")

    leakage_df = st.session_state.get(
        "leakage_df"
    )

    if leakage_df is None:

        try:

            leakage_df = detect_potential_leakage(
                df,
                target_column
            )

        except Exception:

            leakage_df = pandas.DataFrame()

        st.session_state["leakage_df"] = leakage_df


    risk_summary = build_risk_summary(
        df=df,
        quality_summary=quality_summary,
        model_metrics=st.session_state.get(
            "model_metrics"
        ),
        leakage_df=leakage_df,
        fairness_result=st.session_state.get(
            "fairness_result"
        )
    )

    risk_count = len(
        risk_summary
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Findings",
            risk_count
        )

    with col2:

        potential_risks = (
            risk_summary[
                risk_summary["Severity"]
                == "Potential Risk"
            ]
            .shape[0]
        )

        st.metric(
            "Potential Risks",
            potential_risks
        )

    with col3:

        review_items = (
            risk_summary[
                risk_summary["Severity"]
                == "Review"
            ]
            .shape[0]
        )

        st.metric(
            "Items for Review",
            review_items
        )

    st.divider()

    st.subheader("Risk Findings")

    st.dataframe(
        risk_summary,
        width="stretch"
    )

    st.info(
        "Risk findings are screening indicators. "
        "They do not establish that a model is unsafe, "
        "biased, or affected by leakage without further investigation."
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "🧹 Data Quality":

    st.header("🧹 Data Quality Audit")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Missing Cells",
            int(df.isna().sum().sum())
        )

    with col2:

        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    with col3:

        st.metric(
            "Audit Score",
            f"{audit_score}/100"
        )

    st.divider()

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    st.subheader("Missing Values")

    missing_table = quality_summary[
        "missing_table"
    ]

    st.dataframe(
        missing_table,
        width="stretch"
    )

    # --------------------------------------------------------
    # Feature Risks
    # --------------------------------------------------------

    st.subheader("Feature Risk Flags")

    feature_risks = get_feature_risk_flags(
        df
    )

    if feature_risks.empty:

        st.success(
            "No feature-level risk flags detected."
        )

    else:

        st.dataframe(
            feature_risks,
            width="stretch"
        )

    # --------------------------------------------------------
    # Outliers
    # --------------------------------------------------------

    st.subheader("Outlier Analysis")

    outliers = detect_outliers(
        df
    )

    if outliers.empty:

        st.info(
            "No numeric columns suitable for outlier analysis."
        )

    else:

        st.dataframe(
            outliers,
            width="stretch"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    st.subheader("Strong Correlations")

    correlations = detect_strong_correlations(
        df
    )

    if correlations.empty:

        st.info(
            "No strong numeric correlations detected "
            "under the current threshold."
        )

    else:

        st.dataframe(
            correlations,
            width="stretch"
        )

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    if len(numeric_columns) >= 2:

        st.subheader("Correlation Matrix")

        correlation_matrix = df[
            numeric_columns
        ].corr()

        st.dataframe(
            correlation_matrix,
            width="stretch"
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "🤖 Model Evaluation":

    st.header("🤖 Model Evaluation")

    st.write(
        "ModelGuard trains a baseline Logistic Regression "
        "model using an automated preprocessing pipeline."
    )

    # --------------------------------------------------------
    # Target Distribution
    # --------------------------------------------------------

    st.subheader("Target Distribution")

    target_counts = (
        df[target_column]
        .value_counts()
    )

    target_percentages = (
        df[target_column]
        .value_counts(
            normalize=True
        )
        * 100
    )

    target_table = pandas.DataFrame({
        "Class": target_counts.index,
        "Count": target_counts.values,
        "Percentage": target_percentages.round(2).values
    })

    st.dataframe(
        target_table,
        width="stretch"
    )

    # --------------------------------------------------------
    # Imbalance Warning
    # --------------------------------------------------------

    largest_class_percentage = (
        target_percentages.max()
        if not target_percentages.empty
        else 0
    )

    if largest_class_percentage >= 70:

        st.warning(
            f"The largest target class represents "
            f"{largest_class_percentage:.1f}% of the dataset. "
            "Accuracy should therefore be interpreted alongside "
            "precision, recall, and F1."
        )

    # --------------------------------------------------------
    # Run Model
    # --------------------------------------------------------

    if st.button(
        "▶️ Run Baseline Model",
        type="primary"
    ):

        try:

            result = evaluate_baseline_model(
                df,
                target_column
            )

            st.session_state[
                "model_metrics"
            ] = result["metrics"]

            st.session_state[
                "model_result"
            ] = result

            st.success(
                "Baseline model evaluation completed."
            )

        except Exception as e:

            st.error(
                f"Model evaluation failed: {e}"
            )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    model_result = st.session_state.get(
        "model_result"
    )

    if model_result is not None:

        metrics = model_result[
            "metrics"
        ]

        st.divider()

        st.subheader("Model Metrics")

        metric_cols = st.columns(
            len(metrics)
        )

        for index, (
            metric_name,
            metric_value
        ) in enumerate(metrics.items()):

            with metric_cols[index]:

                st.metric(
                    metric_name,
                    f"{metric_value:.3f}"
                )

        st.divider()

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        confusion_matrix = model_result[
            "confusion_matrix"
        ]

        st.dataframe(
            pandas.DataFrame(
                confusion_matrix
            ),
            width="stretch"
        )

        # ----------------------------------------------------
        # Model Risks
        # ----------------------------------------------------

        st.subheader(
            "Model Risk Screening"
        )

        model_risks = identify_model_risks(
            metrics
        )

        if model_risks.empty:

            st.success(
                "No model metrics fell below "
                "the current screening threshold."
            )

        else:

            st.dataframe(
                model_risks,
                width="stretch"
            )

    else:

        st.info(
            "Click **Run Baseline Model** to evaluate the dataset."
        )


# ============================================================
# FAIRNESS
# ============================================================

elif page == "⚖️ Fairness":

    st.header("⚖️ Group Performance Audit")

    st.write(
        "This module compares model performance across "
        "selected groups. A detected gap is a screening "
        "indicator and does not by itself establish unfairness."
    )

    group_columns = get_group_columns(
        df,
        target_column
    )

    if not group_columns:

        st.warning(
            "No suitable categorical/group columns were found "
            "for the current screening criteria."
        )

    else:

        selected_group = st.selectbox(
            "Select Group Column",
            group_columns
        )

        if st.button(
            "⚖️ Run Group Performance Audit"
        ):

            try:

                fairness_result = (
                    run_group_performance_audit(
                        df,
                        target_column,
                        selected_group
                    )
                )

                st.session_state[
                    "fairness_result"
                ] = fairness_result

                st.success(
                    "Group performance audit completed."
                )

            except Exception as e:

                st.error(
                    f"Fairness audit failed: {e}"
                )

        fairness_result = st.session_state.get(
            "fairness_result"
        )

        if fairness_result is not None:

            group_results = fairness_result[
                "group_results"
            ]

            gap_results = fairness_result[
                "gap_results"
            ]

            max_gap = fairness_result[
                "max_gap"
            ]

            if not group_results.empty:

                st.subheader(
                    "Group Performance"
                )

                st.dataframe(
                    group_results,
                    width="stretch"
                )

            if not gap_results.empty:

                st.subheader(
                    "Performance Gaps"
                )

                st.dataframe(
                    gap_results,
                    width="stretch"
                )

                st.subheader(
                    "Interpretation"
                )

                st.info(
                    interpret_group_gap(
                        max_gap
                    )
                )


# ============================================================
# LEAKAGE
# ============================================================

elif page == "🔐 Leakage":

    st.header("🔐 Potential Data Leakage")

    st.write(
        "ModelGuard screens for features that may have an "
        "unusually strong relationship with the target. "
        "These are indicators for investigation, not proof "
        "of data leakage."
    )

    if st.button(
        "🔍 Run Leakage Scan"
    ):

        try:

            leakage_df = detect_potential_leakage(
                df,
                target_column
            )

            st.session_state[
                "leakage_df"
            ] = leakage_df

            st.success(
                "Leakage screening completed."
            )

        except Exception as e:

            st.error(
                f"Leakage scan failed: {e}"
            )

    leakage_df = st.session_state.get(
        "leakage_df"
    )

    if leakage_df is not None:

        if leakage_df.empty:

            st.success(
                "No obvious potential leakage indicators "
                "were detected under the current screening criteria."
            )

        else:

            st.dataframe(
                leakage_df,
                width="stretch"
            )

            st.info(
                interpret_leakage_results(
                    leakage_df
                )
            )

    else:

        st.info(
            "Click **Run Leakage Scan** to begin."
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

elif page == "🔁 Reproducibility":

    st.header("🔁 Reproducibility")

    st.write(
        "ModelGuard records environment and model "
        "configuration information that can support "
        "repeated experiments."
    )

    reproducibility_info = (
        get_reproducibility_info(
    dataset_name,
    target_column
)
    )

    st.subheader(
        "Environment Information"
    )

    if isinstance(
        reproducibility_info,
        dict
    ):

        environment_df = pandas.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value
                }
                for key, value
                in reproducibility_info.items()
            ]
        )

        st.dataframe(
            environment_df,
            width="stretch"
        )

    else:

        st.write(
            reproducibility_info
        )

    st.divider()

    st.subheader(
        "Baseline Model Configuration"
    )

    model_configuration = (
        get_model_configuration()
    )

    if isinstance(
        model_configuration,
        dict
    ):

        model_df = pandas.DataFrame(
            [
                {
                    "Parameter": key,
                    "Value": value
                }
                for key, value
                in model_configuration.items()
            ]
        )

        st.dataframe(
            model_df,
            width="stretch"
        )

    else:

        st.write(
            model_configuration
        )


# ============================================================
# REPORT
# ============================================================

elif page == "📄 Report":

    st.header("📄 Audit Report")

    st.write(
        "Generate a standalone HTML report containing "
        "the current dataset audit information."
    )

    reproducibility_info = (
        get_reproducibility_info(
    dataset_name,
    target_column
)
    )

    if st.button(
        "📄 Generate Audit Report",
        type="primary"
    ):

        try:

            report_html = generate_audit_report(
                dataset_name=dataset_name,
                df=df,
                target_column=target_column,
                audit_score=audit_score,
                reproducibility_info=reproducibility_info
            )

            st.success(
                "Audit report generated successfully."
            )

            st.download_button(
                label="⬇️ Download HTML Report",
                data=report_html,
                file_name="modelguard_audit_report.html",
                mime="text/html"
            )

            st.subheader(
                "Report Preview"
            )

            st.components.v1.html(
                report_html,
                height=700,
                scrolling=True
            )

        except Exception as e:

            st.error(
                f"Report generation failed: {e}"
            )
