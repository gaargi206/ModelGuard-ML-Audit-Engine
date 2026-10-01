# 🛡️ ModelGuard — ML Model Risk Auditor

ModelGuard is an open-source machine learning model risk auditing tool designed to evaluate datasets and classification models across multiple risk dimensions.

It provides a structured audit covering:

- Data Quality
- Model Performance
- Risk Detection
- Fairness / Group Performance
- Potential Data Leakage
- Reproducibility
- Audit Summary
- Risk Findings
- HTML Audit Reporting

---

## 🎯 Problem Statement

Machine learning models can produce unreliable or risky results because of problems in the underlying data, model performance, class imbalance, potential leakage, group-level performance differences, or lack of reproducibility.

ModelGuard provides a centralized auditing workflow to identify these potential risk indicators before a model is considered for further use.

---

## 🚀 Objectives

The main objectives of ModelGuard are:

1. Validate datasets before auditing.
2. Identify common data-quality issues.
3. Evaluate classification model performance.
4. Detect potential model and data risks.
5. Compare model performance across groups.
6. Screen for potential data leakage.
7. Capture reproducibility information.
8. Generate a consolidated audit summary.
9. Generate a downloadable audit report.
10. Support both built-in datasets and custom CSV uploads.

---

## 🧩 Core Modules

### 1. Dataset Validation

Checks whether a dataset is suitable for the audit pipeline.

Examples include:

- Empty datasets
- Missing target values
- Insufficient target classes
- Duplicate records
- High missingness
- Constant features
- Highly imbalanced targets

---

### 2. Data Quality

The data-quality audit examines:

- Missing values
- Duplicate rows
- Feature characteristics
- Outliers
- Strong correlations
- High-cardinality features
- Potential ID-like features

---

### 3. Model Evaluation

ModelGuard evaluates classification models using performance metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Additional evaluation capabilities are being integrated into the project.

---

### 4. Risk Detection

The risk engine screens for potential issues such as:

- Class imbalance
- Missing values
- Duplicate records
- Outliers
- Potential target leakage
- Overfitting indicators

These findings are screening indicators and require further investigation.

---

### 5. Fairness / Group Performance

ModelGuard can compare model performance across selected groups.

The analysis can include:

- Group accuracy
- Positive prediction rate
- Accuracy disparity
- Group-level performance differences

A detected performance gap is treated as an indicator for further investigation rather than definitive proof of unfairness.

---

### 6. Reproducibility

ModelGuard records relevant information about:

- Python environment
- Operating system
- Model configuration
- Experimental settings

This helps support repeatable analysis.

---

### 7. Audit Summary

The audit summary combines the available audit results into a consolidated view containing:

- Dataset information
- Data-quality indicators
- Model performance
- Leakage indicators
- Group-performance information
- Overall screening status

---

### 8. Risk Findings

ModelGuard converts detected indicators into structured findings containing:

- Category
- Risk
- Severity
- Finding
- Recommendation

---

### 9. Audit Report

The application can generate a downloadable HTML audit report containing key dataset and audit information.

---

## 📊 Datasets

ModelGuard currently supports three built-in datasets:

1. Adult / Census Income
2. Online Shoppers Purchasing Intention
3. Predict Students' Dropout and Academic Success

The application also supports custom CSV datasets.

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │     CSV Dataset     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Dataset Loader    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Validation  │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │ Data Quality│   │    Model    │   │    Risk     │
      │    Audit    │   │ Evaluation  │   │  Detection  │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fairness / Groups   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Audit Summary     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌─────────────┐      ┌─────────────┐
             │Risk Findings│      │Audit Report │
             └─────────────┘      └─────────────┘