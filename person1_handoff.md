# ModelGuard — Person 1 ML/Audit Engine Handoff

## Person 1 Responsibilities

Person 1 is responsible for:

1. Data Quality Auditor
2. Model Training & Evaluation
3. Risk Detection
4. Fairness / Group Analysis

---

## 1. Data Quality

### Module
`src.data_quality`

### Main Function
`audit_data_quality()`

### Input
- `df`: pandas DataFrame
- `target_column`: target column name
- `exclude_outlier_columns`: optional list of categorical/encoded columns excluded from IQR analysis

### Output
Dictionary containing:

- `structure`
- `missing_values`
- `duplicates`
- `target_distribution`
- `outliers`

---

## 2. Model Evaluation

### Module
`src.model_evaluation`

### Main Functions
- `evaluate_classification_model()`
- `summarize_model_evaluation()`

### Input
- `y_true`: actual test labels
- `y_pred`: predicted test labels

### Output
Evaluation information containing:

- accuracy
- classification report
- confusion matrix
- summary metrics

---

## 3. Risk Detection

### Module
`src.risk_detection`

### Main Function
`audit_model_risks_with_model()`

### Input
- `df`
- `target_column`
- `model`
- `X_train`
- `y_train`
- `X_test`
- `y_test`
- `feature_columns`
- `exclude_outlier_columns`

### Output
Risk indicators for:

- class imbalance
- missing values
- duplicates
- outliers
- potential target leakage
- overfitting

These are risk indicators and potential issues, not proof that a problem definitely exists.

---

## 4. Fairness / Group Analysis

### Module
`src.fairness`

### Main Function
`audit_fairness()`

### Input
- `model`
- `X_test`
- `y_test`
- `group_column`
- `positive_class`
- `disparity_threshold`

### Output
Fairness information containing:

- group accuracy
- positive prediction rate
- accuracy disparity

---

## Saved Reports

The following reports have been generated:

- `reports/baseline_results.csv`
- `reports/student_risk_results.csv`
- `reports/student_fairness_results.csv`

---

## Tested Datasets

1. Adult / Census Income
2. Online Shoppers Purchasing Intention
3. Predict Students' Dropout and Academic Success

---

## Integration Status

- Data Quality Auditor: PASS
- Model Evaluation: PASS
- Risk Detection: PASS
- Fairness Analysis: PASS
- Three datasets tested: PASS
- End-to-end integration test: PASS
- Module syntax checks: PASS
- Function availability checks: PASS

