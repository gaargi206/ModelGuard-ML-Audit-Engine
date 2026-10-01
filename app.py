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



from src.person1_data_quality_bridge import (

    run_person1_data_quality_audit,

    outliers_to_dataframe,

    missing_values_to_dataframe,

    target_distribution_to_dataframe

)

from src.person1_model_evaluation_bridge import (
    run_person1_model_evaluation,
    confusion_matrix_to_dataframe,
    classification_report_to_dataframe
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

# MODEL GUARD UI THEME
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background: #0B0F17; }

    section[data-testid="stSidebar"] {
        background: #151A24;
        border-right: 1px solid #2A3140;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1450px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        font-weight: 750 !important;
    }

    div[data-testid="stMetric"] {
        background: #151A24;
        border: 1px solid #2A3140;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.20);
    }

    div[data-testid="stMetricLabel"] { color: #9CA7B8; }
    div[data-testid="stMetricValue"] { font-weight: 750; }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #394355;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #2A3140;
        border-radius: 12px;
        background: #111620;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 9px;
        padding: 7px 10px;
        transition: 0.2s ease;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #202735;
    }

    hr { border-color: #2A3140; }
    </style>
    """,
    unsafe_allow_html=True
)

# HEADER
# ============================================================

st.title("🛡️ ModelGuard")
st.caption(
    "ML Model Risk Auditor  •  Data Quality  •  Model Performance  •  "
    "Risk Detection  •  Fairness  •  Leakage  •  Reproducibility"
)

# SESSION STATE

# ============================================================



if "model_metrics" not in st.session_state:

    st.session_state["model_metrics"] = None



if "model_result" not in st.session_state:

    st.session_state["model_result"] = None



if "fairness_result" not in st.session_state:

    st.session_state["fairness_result"] = None



if "leakage_df" not in st.session_state:

    st.session_state["leakage_df"] = None



if "validation_result" not in st.session_state:

    st.session_state["validation_result"] = None



if "person1_data_quality" not in st.session_state:

    st.session_state["person1_data_quality"] = None

if "person1_model_evaluation" not in st.session_state:
    st.session_state["person1_model_evaluation"] = None





# ============================================================

# SIDEBAR

# ============================================================



st.sidebar.markdown(
    "<div style=\"padding:8px 0 22px 0;text-align:center;\"><div style=\"font-size:42px;margin-bottom:4px;\">🛡️</div><div style=\"font-size:27px;font-weight:800;letter-spacing:-0.5px;\">ModelGuard</div><div style=\"font-size:11px;color:#8F9BAD;margin-top:5px;letter-spacing:1px;text-transform:uppercase;\">ML Risk Auditor</div></div>",
    unsafe_allow_html=True
)



st.sidebar.markdown(
    "<div style=\"font-size:11px;font-weight:700;color:#8F9BAD;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;\">DATASET</div>",
    unsafe_allow_html=True
)



dataset_source = st.sidebar.radio(

    "Choose dataset source",

    [

        "Built-in Dataset",

        "Upload Custom CSV"

    ]

)





# ============================================================

# DATASET VARIABLES

# ============================================================



df = None

dataset_name = None

target_column = None





# ============================================================

# BUILT-IN DATASETS

# ============================================================



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

    # ADULT

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



            if len(df) > 0:



                first_row = (

                    df.iloc[0]

                    .astype(str)

                    .str.strip()

                    .str.lower()

                )



                if first_row["income"] == "income":



                    df = (

                        df.iloc[1:]

                        .reset_index(drop=True)

                    )



            df = df.replace(

                "?",

                pandas.NA

            )



            target_column = "income"



        except Exception as e:



            st.error(

                f"Could not load Adult dataset: {e}"

            )



            st.stop()





    # --------------------------------------------------------

    # ONLINE SHOPPERS

    # --------------------------------------------------------



    elif dataset_option == "Online Shoppers Purchasing Intention":



        dataset_name = (

            "Online Shoppers Purchasing Intention"

        )



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

    # STUDENT DROPOUT

    # --------------------------------------------------------



    elif dataset_option == "Student Dropout & Academic Success":



        dataset_name = (

            "Student Dropout & Academic Success"

        )



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





# ============================================================

# CUSTOM CSV

# ============================================================



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

            - ⚠️ Risk Detection

            - ⚖️ Group Performance

            - 🔐 Potential Data Leakage

            - 🔁 Reproducibility

            - 📄 Audit Report

            """

        )



        st.stop()



    try:



        df = load_csv_file(

            uploaded_file

        )



        df = clean_dataset(

            df

        )



    except Exception as e:



        st.error(

            str(e)

        )



        st.stop()



    target_column = st.sidebar.selectbox(

        "Select Target Column",

        df.columns.tolist()

    )





# ============================================================

# COMMON CLEANING

# ============================================================



if df is None:



    st.error(

        "No dataset could be loaded."

    )



    st.stop()





df.columns = [

    str(column).strip()

    for column in df.columns

]





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





df = df.dropna(

    subset=[target_column]

).reset_index(drop=True)





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

st.session_state[
    "validation_result"
] = validation_result


# ============================================================
# SIDEBAR VALIDATION STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="
        font-size:11px;
        font-weight:700;
        color:#8F9BAD;
        letter-spacing:1px;
        text-transform:uppercase;
        margin-top:12px;
        margin-bottom:8px;
    ">
        VALIDATION STATUS
    </div>
    """,
    unsafe_allow_html=True
)

if validation_result["valid"]:

    st.sidebar.success(
        "✅ Dataset Valid"
    )

else:

    st.sidebar.error(
        "❌ Dataset Invalid"
    )


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

    st.subheader(
        "Validation Errors"
    )

    for issue in validation_result["issues"]:

        st.error(
            f"• {issue}"
        )

    if validation_result["warnings"]:

        st.subheader(
            "Warnings"
        )

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
# NAVIGATION
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="
        font-size:11px;
        font-weight:700;
        color:#8F9BAD;
        letter-spacing:1px;
        text-transform:uppercase;
        margin-top:12px;
        margin-bottom:8px;
    ">
        AUDIT MODULES
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
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
    ],
    label_visibility="collapsed"
)


# ============================================================
# EXISTING DATA QUALITY SUMMARY
# ============================================================

quality_summary = get_data_quality_summary(
    df
)

audit_score = calculate_audit_score(
    df
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="padding:8px 0 20px 0;">

            <div style="
                font-size:14px;
                color:#8F9BAD;
                letter-spacing:1px;
                text-transform:uppercase;
                font-weight:600;
            ">
                ML MODEL RISK AUDITOR
            </div>

            <div style="
                font-size:34px;
                font-weight:800;
                margin-top:4px;
            ">
                Dataset Overview
            </div>

            <div style="
                font-size:15px;
                color:#8F9BAD;
                margin-top:6px;
            ">
                Review dataset health, structure and audit readiness.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # VALIDATION STATUS
    # --------------------------------------------------------

    if validation_result["valid"]:

        st.success(
            "✓ Dataset validated successfully"
        )

    else:

        st.error(
            "✕ Dataset validation failed"
        )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    rows = int(
        df.shape[0]
    )

    columns = int(
        df.shape[1]
    )

    missing_cells = int(
        df.isna()
        .sum()
        .sum()
    )

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "ROWS",
            f"{rows:,}"
        )


    with col2:

        st.metric(
            "COLUMNS",
            columns
        )


    with col3:

        st.metric(
            "MISSING CELLS",
            f"{missing_cells:,}"
        )


    with col4:

        st.metric(
            "AUDIT SCORE",
            f"{audit_score}/100"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # RISK OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "Risk Overview"
    )


    risk_col1, risk_col2, risk_col3 = st.columns(3)


    with risk_col1:

        if audit_score >= 80:

            st.success(
                "🟢 Data Quality\n\n"
                "Low screening risk"
            )

        elif audit_score >= 60:

            st.warning(
                "🟡 Data Quality\n\n"
                "Review recommended"
            )

        else:

            st.error(
                "🔴 Data Quality\n\n"
                "Potential risk"
            )


    with risk_col2:

        if st.session_state.get(
            "model_metrics"
        ):

            metrics = st.session_state[
                "model_metrics"
            ]

            accuracy = metrics.get(
                "Accuracy"
            )

            if (
                accuracy is not None
                and accuracy >= 0.70
            ):

                st.success(
                    f"🟢 Model Performance\n\n"
                    f"Accuracy: {accuracy:.1%}"
                )

            else:

                st.warning(
                    "🟡 Model Performance\n\n"
                    "Review required"
                )

        else:

            st.info(
                "🔵 Model Performance\n\n"
                "Not evaluated yet"
            )


    with risk_col3:

        fairness_result = st.session_state.get(
            "fairness_result"
        )

        if fairness_result:

            max_gap = fairness_result.get(
                "max_gap"
            )

            if (
                max_gap is not None
                and max_gap >= 0.20
            ):

                st.warning(
                    f"🟡 Fairness\n\n"
                    f"Gap: {max_gap:.1%}"
                )

            else:

                st.success(
                    "🟢 Fairness\n\n"
                    "No large gap detected"
                )

        else:

            st.info(
                "🔵 Fairness\n\n"
                "Not evaluated yet"
            )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Dataset Information"
    )


    info_col1, info_col2 = st.columns(2)


    with info_col1:

        st.markdown(
            f"""
            **Dataset**

            `{dataset_name}`

            **Target**

            `{target_column}`
            """
        )


    with info_col2:

        numeric_count = len(
            df.select_dtypes(
                include="number"
            ).columns
        )

        categorical_count = len(
            df.select_dtypes(
                exclude="number"
            ).columns
        )

        st.markdown(
            f"""
            **Numeric Features**

            `{numeric_count}`

            **Categorical Features**

            `{categorical_count}`

            **Duplicate Rows**

            `{duplicate_rows:,}`
            """
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "📊 Preview Dataset",
        expanded=False
    ):

        st.dataframe(
            df.head(10),
            width="stretch"
        )


# ============================================================
# AUDIT SUMMARY
# ============================================================

elif page == "🧠 Audit Summary":

    st.header(
        "🧠 Audit Summary"
    )

    summary = generate_audit_summary(
        df=df,
        target_column=target_column,
        audit_score=audit_score,
        quality_summary=quality_summary,
        model_metrics=st.session_state.get(
            "model_metrics"
        ),
        leakage_df=st.session_state.get(
            "leakage_df"
        ),
        fairness_result=st.session_state.get(
            "fairness_result"
        )
    )

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

    st.header(
        "🚨 Risk Findings"
    )

    risk_df = build_risk_summary(
        df=df,
        quality_summary=quality_summary,
        model_metrics=st.session_state.get(
            "model_metrics"
        ),
        leakage_df=st.session_state.get(
            "leakage_df"
        ),
        fairness_result=st.session_state.get(
            "fairness_result"
        )
    )

    if risk_df.empty:

        st.success(
            "No screening findings were generated."
        )

    else:

        st.dataframe(
            risk_df,
            width="stretch"
        )# ============================================================
# DATA QUALITY
# ============================================================

elif page == "🧹 Data Quality":

    st.header(
        "🧹 Data Quality Audit"
    )

    st.write(
        "ModelGuard checks missing values, duplicates, "
        "outliers, feature risks and strong correlations."
    )

    # --------------------------------------------------------
    # PERSON 1 DATA QUALITY ENGINE
    # --------------------------------------------------------

    try:

        person1_result = run_person1_data_quality_audit(
            df=df,
            target_column=target_column
        )

        st.session_state[
            "person1_data_quality"
        ] = person1_result

    except Exception as e:

        st.warning(
            f"Person 1 Data Quality Engine could not run: {e}"
        )

        person1_result = None


    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "ROWS",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "COLUMNS",
            len(df.columns)
        )

    with col3:

        st.metric(
            "MISSING CELLS",
            f"{int(df.isna().sum().sum()):,}"
        )

    with col4:

        st.metric(
            "DUPLICATE ROWS",
            f"{int(df.duplicated().sum()):,}"
        )


    st.divider()


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    missing_table = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_df = pandas.DataFrame({
        "Feature": missing_table.index,
        "Missing Values": missing_table.values,
        "Missing %": (
            missing_table.values
            / max(len(df), 1)
            * 100
        ).round(2)
    })

    st.dataframe(
        missing_df,
        width="stretch"
    )


    # --------------------------------------------------------
    # FEATURE RISK FLAGS
    # --------------------------------------------------------

    st.subheader(
        "Feature Risk Flags"
    )

    feature_risks = get_feature_risk_flags(
        df
    )

    if feature_risks.empty:

        st.success(
            "No feature-level screening risks detected."
        )

    else:

        st.dataframe(
            feature_risks,
            width="stretch"
        )


    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    st.subheader(
        "Outlier Detection"
    )

    outlier_df = detect_outliers(
        df
    )

    if outlier_df.empty:

        st.info(
            "No numeric columns were available "
            "for outlier analysis."
        )

    else:

        st.dataframe(
            outlier_df,
            width="stretch"
        )


    # --------------------------------------------------------
    # STRONG CORRELATIONS
    # --------------------------------------------------------

    st.subheader(
        "Strong Correlations"
    )

    correlations = detect_strong_correlations(
        df
    )

    if correlations.empty:

        st.info(
            "No strong numeric correlations detected "
            "under the current screening threshold."
        )

    else:

        st.dataframe(
            correlations,
            width="stretch"
        )


    # --------------------------------------------------------
    # PRELIMINARY SCORE
    # --------------------------------------------------------

    st.subheader(
        "Preliminary Data Quality Score"
    )

    score_col1, score_col2 = st.columns(
        [1, 2]
    )

    with score_col1:

        st.metric(
            "Audit Score",
            f"{audit_score}/100"
        )

    with score_col2:

        if audit_score >= 80:

            st.success(
                "Low screening concern based on "
                "the current preliminary score."
            )

        elif audit_score >= 60:

            st.warning(
                "Some data-quality indicators "
                "should be reviewed."
            )

        else:

            st.error(
                "Significant data-quality indicators "
                "require investigation."
            )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "🤖 Model Evaluation":

    st.header(
        "🤖 Model Evaluation"
    )

    st.write(
        "ModelGuard trains a baseline Logistic Regression "
        "model using an automated preprocessing pipeline."
    )


    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Target Distribution"
    )

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
    # MODEL BUTTON
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


    model_result = (
        st.session_state.get(
            "model_result"
        )
    )


    if model_result is not None:

        metrics = model_result[
            "metrics"
        ]


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Model Metrics"
        )

        metric_cols = st.columns(
            len(metrics)
        )

        for index, (
            metric_name,
            metric_value
        ) in enumerate(
            metrics.items()
        ):

            with metric_cols[index]:

                st.metric(
                    metric_name,
                    f"{metric_value:.3f}"
                )


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.divider()

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
        # MODEL RISK SCREENING
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

    st.header(
        "⚖️ Group Performance Audit"
    )

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


        fairness_result = (
            st.session_state.get(
                "fairness_result"
            )
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

    st.header(
        "🔐 Potential Data Leakage"
    )

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

            leakage_df = (
                detect_potential_leakage(
                    df,
                    target_column
                )
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


    leakage_df = (
        st.session_state.get(
            "leakage_df"
        )
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
        )# ============================================================
# REPRODUCIBILITY
# ============================================================

elif page == "🔁 Reproducibility":

    st.header(
        "🔁 Reproducibility"
    )

    st.write(
        "ModelGuard records the configuration and environment "
        "information needed to help reproduce the audit."
    )

    try:

        reproducibility_info = get_reproducibility_info()

        st.subheader(
            "Environment Information"
        )

        if isinstance(
            reproducibility_info,
            dict
        ):

            reproducibility_df = pandas.DataFrame(
                [
                    {
                        "Property": key,
                        "Value": value
                    }
                    for key, value
                    in reproducibility_info.items()
                ]
            )

            st.dataframe(
                reproducibility_df,
                width="stretch"
            )

        else:

            st.write(
                reproducibility_info
            )

    except Exception as e:

        st.error(
            f"Could not collect reproducibility information: {e}"
        )


    st.divider()

    st.subheader(
        "Model Configuration"
    )

    try:

        model_configuration = get_model_configuration()

        if isinstance(
            model_configuration,
            dict
        ):

            configuration_df = pandas.DataFrame(
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
                configuration_df,
                width="stretch"
            )

        else:

            st.write(
                model_configuration
            )

    except Exception as e:

        st.error(
            f"Could not collect model configuration: {e}"
        )


# ============================================================
# REPORT
# ============================================================

elif page == "📄 Report":

    st.header(
        "📄 ModelGuard Audit Report"
    )

    st.write(
        "Generate a consolidated HTML report containing "
        "the available ModelGuard audit results."
    )


    leakage_df = st.session_state.get(
        "leakage_df"
    )

    fairness_result = st.session_state.get(
        "fairness_result"
    )

    model_metrics = st.session_state.get(
        "model_metrics"
    )


    if leakage_df is None:

        try:

            leakage_df = detect_potential_leakage(
                df,
                target_column
            )

        except Exception:

            leakage_df = pandas.DataFrame()


    risk_summary = build_risk_summary(
        df=df,
        quality_summary=quality_summary,
        model_metrics=model_metrics,
        leakage_df=leakage_df,
        fairness_result=fairness_result
    )


    audit_summary = generate_audit_summary(
        df=df,
        target_column=target_column,
        audit_score=audit_score,
        quality_summary=quality_summary,
        model_metrics=model_metrics,
        leakage_df=leakage_df,
        fairness_result=fairness_result
    )


    st.subheader(
        "Report Contents"
    )

    report_col1, report_col2 = st.columns(2)

    with report_col1:

        st.markdown(
            """
            ✓ Dataset information

            ✓ Data quality findings

            ✓ Model performance

            ✓ Risk indicators
            """
        )

    with report_col2:

        st.markdown(
            """
            ✓ Fairness results

            ✓ Leakage screening

            ✓ Audit summary

            ✓ Risk findings
            """
        )


    st.divider()


    if st.button(
        "📄 Generate HTML Report",
        type="primary"
    ):

        try:

            report_path = generate_audit_report(
                df=df,
                target_column=target_column,
                audit_score=audit_score,
                quality_summary=quality_summary,
                model_metrics=model_metrics,
                leakage_df=leakage_df,
                fairness_result=fairness_result,
                risk_summary=risk_summary,
                audit_summary=audit_summary
            )

            st.success(
                "HTML audit report generated successfully."
            )


            try:

                with open(
                    report_path,
                    "rb"
                ) as report_file:

                    st.download_button(
                        "⬇️ Download Audit Report",
                        data=report_file,
                        file_name="modelguard_audit_report.html",
                        mime="text/html"
                    )

            except Exception as download_error:

                st.warning(
                    f"Report was generated, but the download "
                    f"button could not be prepared: {download_error}"
                )

        except TypeError:

            try:

                report_path = generate_audit_report(
                    df,
                    target_column,
                    audit_score,
                    quality_summary,
                    model_metrics,
                    leakage_df,
                    fairness_result
                )

                st.success(
                    "HTML audit report generated successfully."
                )

                try:

                    with open(
                        report_path,
                        "rb"
                    ) as report_file:

                        st.download_button(
                            "⬇️ Download Audit Report",
                            data=report_file,
                            file_name="modelguard_audit_report.html",
                            mime="text/html"
                        )

                except Exception as download_error:

                    st.warning(
                        f"Report was generated, but the download "
                        f"button could not be prepared: {download_error}"
                    )

            except Exception as e:

                st.error(
                    f"Report generation failed: {e}"
                )

        except Exception as e:

            st.error(
                f"Report generation failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px 0 10px 0;
        color:#6F7B8C;
        font-size:12px;
    ">
        <strong>🛡️ ModelGuard</strong>
        &nbsp;•&nbsp;
        ML Model Risk Auditor
        &nbsp;•&nbsp;
        Open Source Project
    </div>
    """,
    unsafe_allow_html=True
)