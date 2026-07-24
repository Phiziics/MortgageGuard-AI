# MortgageGuard AI

MortgageGuard AI is a production-grade mortgage credit-risk and model-validation platform built with real Freddie Mac loan-level data.

The system predicts whether a mortgage will experience a serious credit event within the next 12 months and supports the complete machine-learning lifecycle:

Business Problem
→ Data Ingestion
→ Data Validation
→ Feature Engineering
→ Model Development
→ Experiment Tracking
→ Independent Model Validation
→ Model Registry
→ Deployment
→ Monitoring
→ Alerts
→ Retraining or Rollback
→ Governance Documentation

## Business Problem

Mortgage lenders need to identify loans with elevated default risk before serious delinquency occurs.

MortgageGuard AI will estimate the probability that a mortgage experiences a serious credit event within the following 12 months.

A serious credit event may include:

- 90 or more days delinquent
- Foreclosure
- Short sale
- Real-estate-owned status
- Charge-off

## Business Objectives

The platform will help a fictional financial institution:

- Identify high-risk mortgages
- Prioritize account reviews
- Estimate expected credit loss
- Monitor portfolio deterioration
- Validate model performance
- Detect data and model drift
- Retrain, recalibrate, restrict, or roll back models

## Data Source

Freddie Mac Single-Family Loan-Level Dataset

Planned vintage years:

- 2020
- 2021
- 2022
- 2023
- 2024

The raw dataset will not be committed to GitHub because of file size and dataset usage restrictions.

## Modeling Strategy

### Baseline

Simple business-rule risk score using:

- Credit score
- Loan-to-value ratio
- Debt-to-income ratio

### Champion Model

Logistic Regression

### Challenger Model

XGBoost

## Evaluation Metrics

### Statistical Metrics

- ROC-AUC
- Precision-Recall AUC
- Gini coefficient
- KS statistic
- Brier score
- Calibration slope
- Calibration intercept
- Recall
- Precision
- F1 score

### Business Metrics

- Default capture rate
- High-risk loan balance
- Expected credit loss
- Top-decile risk capture
- False-negative cost
- Manual review volume

## Time-Based Data Strategy

- Development data: 2020–2021
- Validation data: 2022
- Out-of-time test data: 2023
- Production replay and monitoring: 2024

The exact scoring periods will be finalized after inspecting the available monthly performance history.

## MLOps Architecture

- Python and SQL for processing
- Parquet and DuckDB for local analytical storage
- MLflow for experiment tracking and model registry
- Git for source-code versioning
- DVC for data-version metadata
- FastAPI for online predictions
- Batch scoring for portfolio predictions
- Docker for packaging
- pytest for automated testing
- GitHub Actions for CI/CD
- Evidently and custom metrics for monitoring
- Streamlit for model-monitoring dashboards

## Repository Structure

```text
MortgageGuard-AI/
├── configs/
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── reference/
├── notebooks/
├── sql/
├── src/
├── pipelines/
├── api/
├── dashboard/
├── tests/
├── models/
├── artifacts/
├── reports/
├── deployment/
└── .github/workflows/