from datetime import datetime
import html


def generate_audit_report(
    dataset_name,
    df,
    target_column,
    audit_score,
    reproducibility_info
):
    """
    Generate a standalone HTML audit report for ModelGuard.
    """

    rows = df.shape[0]
    columns = df.shape[1]

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    missing_percentage = (
        missing_cells /
        max(rows * columns, 1)
    ) * 100

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]

    high_cardinality_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) > 50
    ]

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # ---------------------------------------------------------
    # TARGET DISTRIBUTION
    # ---------------------------------------------------------

    target_distribution_html = ""

    if target_column in df.columns:

        target_counts = (
            df[target_column]
            .value_counts(dropna=False)
            .reset_index()
        )

        target_counts.columns = [
            "Class",
            "Count"
        ]

        target_counts["Percentage"] = (
            target_counts["Count"]
            / target_counts["Count"].sum()
            * 100
        ).round(2)

        target_distribution_html = (
            target_counts
            .to_html(
                index=False,
                classes="data-table",
                border=0
            )
        )

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    missing_data = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_table = []

    for column, count in missing_data.items():

        percentage = (
            count /
            max(len(df), 1)
            * 100
        )

        missing_table.append({
            "Feature": column,
            "Missing Values": int(count),
            "Missing %": round(
                percentage,
                2
            )
        })

    missing_df_html = (
        __import__("pandas")
        .DataFrame(missing_table)
        .to_html(
            index=False,
            classes="data-table",
            border=0
        )
    )

    # ---------------------------------------------------------
    # REPRODUCIBILITY
    # ---------------------------------------------------------

    reproducibility_rows = ""

    for item in reproducibility_info:

        check = html.escape(
            str(item.get("Check", ""))
        )

        value = html.escape(
            str(item.get("Value", ""))
        )

        status = html.escape(
            str(item.get("Status", ""))
        )

        reproducibility_rows += f"""
        <tr>
            <td>{check}</td>
            <td>{value}</td>
            <td>{status}</td>
        </tr>
        """

    # ---------------------------------------------------------
    # FEATURE INFORMATION
    # ---------------------------------------------------------

    constant_html = (
        ", ".join(
            map(str, constant_columns)
        )
        if constant_columns
        else "None detected"
    )

    high_cardinality_html = (
        ", ".join(
            map(str, high_cardinality_columns)
        )
        if high_cardinality_columns
        else "None detected"
    )

    # ---------------------------------------------------------
    # DATASET PREVIEW
    # ---------------------------------------------------------

    preview_html = (
        df.head(10)
        .to_html(
            index=False,
            classes="data-table",
            border=0
        )
    )

    # ---------------------------------------------------------
    # GENERATE HTML
    # ---------------------------------------------------------

    generated_time = datetime.now().strftime(
        "%d %B %Y, %H:%M:%S"
    )

    report_html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>ModelGuard Audit Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background-color: #f5f7fa;
    color: #222;
    margin: 0;
    padding: 0;
}}

.container {{
    max-width: 1100px;
    margin: 40px auto;
    background: white;
    padding: 40px;
    border-radius: 12px;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 35px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 8px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

.metric-container {{
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 15px;
    margin: 25px 0;
}}

.metric {{
    background: #f1f3f6;
    padding: 20px;
    border-radius: 10px;
}}

.metric-title {{
    font-size: 13px;
    color: #666;
}}

.metric-value {{
    font-size: 25px;
    font-weight: bold;
    margin-top: 5px;
}}

.score {{
    font-size: 35px;
    font-weight: bold;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

.data-table th {{
    background: #f1f3f6;
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid #ddd;
}}

.data-table td {{
    padding: 9px;
    border-bottom: 1px solid #eee;
}}

.info-box {{
    background: #f1f3f6;
    padding: 18px;
    border-radius: 8px;
    margin-top: 15px;
}}

.warning {{
    background: #fff4d6;
    padding: 15px;
    border-radius: 8px;
}}

.footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    color: #777;
    font-size: 13px;
}}

</style>

</head>


<body>

<div class="container">

<h1>🛡️ ModelGuard</h1>

<div class="subtitle">
ML Model Risk Audit Report
</div>


<div class="info-box">

<strong>Dataset:</strong>
{html.escape(str(dataset_name))}

<br><br>

<strong>Target Variable:</strong>
{html.escape(str(target_column))}

<br><br>

<strong>Report Generated:</strong>
{generated_time}

</div>


<h2>1. Preliminary Audit Score</h2>

<div class="score">
{audit_score}/100
</div>

<p>
This is a preliminary screening score based on
basic dataset quality indicators. It should not
be interpreted as a formal model-risk rating.
</p>


<h2>2. Dataset Overview</h2>

<div class="metric-container">

<div class="metric">

<div class="metric-title">
Rows
</div>

<div class="metric-value">
{rows}
</div>

</div>


<div class="metric">

<div class="metric-title">
Columns
</div>

<div class="metric-value">
{columns}
</div>

</div>


<div class="metric">

<div class="metric-title">
Missing Cells
</div>

<div class="metric-value">
{missing_cells}
</div>

</div>


<div class="metric">

<div class="metric-title">
Duplicate Rows
</div>

<div class="metric-value">
{duplicate_rows}
</div>

</div>

</div>


<h2>3. Feature Profile</h2>

<table class="data-table">

<tr>
<th>Category</th>
<th>Count</th>
</tr>

<tr>
<td>Numeric Features</td>
<td>{len(numeric_columns)}</td>
</tr>

<tr>
<td>Categorical Features</td>
<td>{len(categorical_columns)}</td>
</tr>

<tr>
<td>Constant Features</td>
<td>{len(constant_columns)}</td>
</tr>

<tr>
<td>High Cardinality Features</td>
<td>{len(high_cardinality_columns)}</td>
</tr>

</table>


<div class="warning">

<strong>Constant Features:</strong>

{html.escape(constant_html)}

<br><br>

<strong>High Cardinality Features:</strong>

{html.escape(high_cardinality_html)}

</div>


<h2>4. Missing Value Analysis</h2>

<p>
Overall missing-cell percentage:
<strong>
{missing_percentage:.2f}%
</strong>
</p>

{missing_df_html}


<h2>5. Target Distribution</h2>

{target_distribution_html}


<h2>6. Dataset Preview</h2>

{preview_html}


<h2>7. Reproducibility Information</h2>

<table class="data-table">

<tr>
<th>Check</th>
<th>Value</th>
<th>Status</th>
</tr>

{reproducibility_rows}

</table>


<h2>8. Audit Interpretation</h2>

<div class="info-box">

ModelGuard provides automated screening indicators
for data quality, model risk, potential leakage,
group-performance differences, and reproducibility.

Flagged conditions require human review and should
not automatically be interpreted as evidence of
model failure, unfairness, or data leakage.

</div>


<div class="footer">

Generated by ModelGuard — ML Model Risk Auditor

</div>


</div>

</body>

</html>
"""

    return report_html