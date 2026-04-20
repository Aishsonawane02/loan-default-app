# Loan Default Prediction — End-to-End ML Pipeline

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
loan-default-app/
├── app.py                  # Streamlit web application
├── model/
│   ├── loan_model.pkl      # Trained Gradient Boosting model
│   ├── scaler.pkl          # StandardScaler (for numeric features)
│   ├── encoders.pkl        # LabelEncoder objects (one per categorical)
│   ├── feature_names.pkl   # Ordered list of model features
│   └── threshold.pkl       # Optimal classification threshold (0.632)
├── utils/
│   └── preprocessing.py    # Input encoding, risk categorization helpers
├── data/
│   └── sample.csv          # 100 sample rows for testing
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** Kaggle — Loan Default Dataset
- **Rows:** 148,670 loan applications
- **Target:** `Status` (1 = Default, 0 = No Default)
- **Default Rate:** 24.6%

## Key Finding — Data Leakage

Three columns were **removed** due to data leakage:

| Column | Issue |
|---|---|
| `Interest_rate_spread` | 100% null for all defaulters — perfectly reveals target |
| `Upfront_charges` | 99.6% null for defaulters |
| `rate_of_interest` | 99.5% null for defaulters |

**Without removal:** AUC = 1.000 (impossibly perfect — classic leakage sign)  
**After removal:** AUC = 0.889 (realistic, deployable)

## Model Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.731 | 0.501 | 0.679 |
| Random Forest | 0.887 | 0.736 | 0.882 |
| **Gradient Boosting** | **0.889** | **0.749** | 0.873 |

**Final model:** Gradient Boosting with threshold = 0.632

## Risk Tiers

| Default Probability | Risk Level | Decision |
|---|---|---|
| 0% – 25% | Low Risk | Approve |
| 25% – 50% | Medium Risk | Review |
| 50% – 70% | High Risk | Reject |
| 70%+ | Critical | Reject |

## Top Features

1. **property_value** — 29.9%
2. **LTV (Loan-to-Value Ratio)** — 28.6%
3. **dtir1 (Debt-to-Income Ratio)** — 10.8%
4. **income** — 5.3%
5. **lump_sum_payment** — 4.1%
