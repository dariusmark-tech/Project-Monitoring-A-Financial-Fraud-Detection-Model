# Fraud Detection Model Performance Monitoring

## Overview

This project addresses a critical problem faced by Poundbank, a London-based financial institution: their fraud detection machine learning models are experiencing accuracy degradation due to changing data patterns. Using the `nannyml` library, this analysis monitors model performance and identifies the root causes of drift in production data.

## Problem Statement

Banks increasingly rely on ML models for fraud detection, but evolving transaction patterns can silently erode model effectiveness. Poundbank needs to understand why their fraud detection accuracy is declining and identify which features are driving this degradation.

## Data Description

Two datasets are provided:

- **reference.csv** – Historical/test data used as the baseline for model performance
- **analysis.csv** – Production data where model performance is monitored

### Features

| Column | Description |
|--------|-------------|
| `timestamp` | Date of the transaction |
| `time_since_login_min` | Time since user logged in (minutes) |
| `transaction_amount` | Transaction amount in GBP |
| `transaction_type` | CASH-OUT, PAYMENT, CASH-IN, or TRANSFER |
| `is_first_transaction` | Binary indicator for user's first transaction |
| `user_tenure_months` | Account age in months |
| `is_fraud` | Ground truth label (1 = fraud, 0 = legitimate) |
| `predicted_fraud_proba` | Model's fraud probability score |
| `predicted_fraud` | Model's binary prediction |

## Methodology

### 1. Performance Monitoring (CBPE)
Confidence-Based Performance Estimation (CBPE) estimates model accuracy in production without requiring immediate ground truth labels.

### 2. Realized Performance Calculation
Actual accuracy is calculated where ground truth is available to validate estimates.

### 3. Univariate Drift Detection
- **Numerical features** – Kolmogorov-Smirnov test
- **Categorical features** – Chi-squared test

## Key Findings

### Performance Alerts
Model accuracy triggered alerts during the following months:
- **April 2019**
- **May 2019**  
- **June 2019**

### Root Cause Identification
The feature with the highest drift correlation was **`time_since_login_min`** – the time elapsed since the user logged into the application. This feature showed the most frequent drift alerts across all monitored periods.

### Transaction Amount Anomaly
The average transaction amount peaked in **June 2019** at **£3,069.8**, representing the maximum deviation from the reference baseline.

## Technologies Used

- **Python** – Core analysis language
- **pandas** – Data manipulation and aggregation
- **nannyml** – Model monitoring and drift detection library
- **Jupyter Notebook** – Interactive analysis environment

## Results Summary

```
months_with_performance_alerts = ['april_2019', 'june_2019', 'may_2019']
highest_correlation_feature    = 'time_since_login_min'
alert_avg_transaction_amount   = 3069.8
```

## Business Implications

1. **Model Retraining** – The fraud detection model requires retraining on recent data, particularly focusing on patterns observed from April to June 2019
2. **Feature Engineering** – `time_since_login_min` should be monitored closely, potentially with dynamic thresholds
3. **Transaction Monitoring** – Elevated average transaction amounts may indicate changing fraud patterns or shifts in user behavior

## Getting Started

### Prerequisites

```bash
pip install nannyml pandas
```

### Running the Analysis

```python
import pandas as pd
import nannyml as nml

# Load data
reference = pd.read_csv("reference.csv")
analysis = pd.read_csv("analysis.csv")

# Configure performance monitoring
estimator = nml.CBPE(
    y_pred_proba='predicted_fraud_proba',
    y_pred='predicted_fraud',
    y_true='is_fraud',
    timestamp_column_name='timestamp',
    metrics=['accuracy'],
    chunk_period='M',
    problem_type='classification_binary'
)

# Fit and estimate
estimator.fit(reference)
results = estimator.estimate(analysis)
```

## License

This project is for educational and demonstration purposes.

## Author

Analysis performed as part of Poundbank's fraud detection optimization initiative.
