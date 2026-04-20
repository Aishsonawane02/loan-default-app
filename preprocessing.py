"""
preprocessing.py
Handles all data cleaning and encoding for the Loan Default Predictor.
Used by both the training pipeline and the Streamlit app.
"""
import pandas as pd
import numpy as np


# ── Column groups ─────────────────────────────────────────────
# These 3 columns were removed due to data leakage
# (their missingness perfectly predicted the target in training data)
LEAKY_COLS = ['Interest_rate_spread', 'Upfront_charges', 'rate_of_interest']

NUMERIC_COLS = [
    'loan_amount', 'property_value', 'LTV',
    'income', 'Credit_Score', 'term', 'dtir1'
]

CATEGORICAL_COLS = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial',
    'Neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
    'credit_type', 'co-applicant_credit_type', 'age',
    'submission_of_application', 'Region', 'Security_Type'
]

# Friendly display names for Streamlit UI
FEATURE_LABELS = {
    'loan_amount':               'Loan Amount ($)',
    'property_value':            'Property Value ($)',
    'LTV':                       'Loan-to-Value Ratio (%)',
    'income':                    'Annual Income ($)',
    'Credit_Score':              'Credit Score',
    'term':                      'Loan Term (months)',
    'dtir1':                     'Debt-to-Income Ratio (%)',
    'loan_limit':                'Loan Limit Type',
    'Gender':                    'Gender',
    'approv_in_adv':             'Pre-Approved?',
    'loan_type':                 'Loan Type',
    'loan_purpose':              'Loan Purpose',
    'Credit_Worthiness':         'Credit Worthiness Level',
    'open_credit':               'Open Credit Account?',
    'business_or_commercial':    'Business / Commercial Loan?',
    'Neg_ammortization':         'Negative Amortization?',
    'interest_only':             'Interest Only Payment?',
    'lump_sum_payment':          'Lump Sum Payment?',
    'construction_type':         'Construction Type',
    'occupancy_type':            'Occupancy Type',
    'Secured_by':                'Secured By',
    'total_units':               'Total Units',
    'credit_type':               'Credit Bureau Used',
    'co-applicant_credit_type':  'Co-applicant Credit Bureau',
    'age':                       'Applicant Age Group',
    'submission_of_application': 'Submission Method',
    'Region':                    'Region',
    'Security_Type':             'Security Type',
}

# Allowed values for each categorical (from training data)
CATEGORICAL_OPTIONS = {
    'loan_limit':                ['cf', 'ncf'],
    'Gender':                    ['Male', 'Female', 'Joint', 'Sex Not Available'],
    'approv_in_adv':             ['pre', 'nopre'],
    'loan_type':                 ['type1', 'type2', 'type3'],
    'loan_purpose':              ['p1', 'p2', 'p3', 'p4'],
    'Credit_Worthiness':         ['l1', 'l2'],
    'open_credit':               ['nopc', 'opc'],
    'business_or_commercial':    ['nob/c', 'b/c'],
    'Neg_ammortization':         ['not_neg', 'neg_amm'],
    'interest_only':             ['not_int', 'int_only'],
    'lump_sum_payment':          ['not_lpsm', 'lpsm'],
    'construction_type':         ['sb', 'mh'],
    'occupancy_type':            ['pr', 'sr', 'ir'],
    'Secured_by':                ['home', 'land'],
    'total_units':               ['1U', '2U', '3U', '4U'],
    'credit_type':               ['EXP', 'EQUI', 'CRIF', 'CIB'],
    'co-applicant_credit_type':  ['CIB', 'EXP'],
    'age':                       ['25-34', '35-44', '45-54', '55-64', '65-74'],
    'submission_of_application': ['to_inst', 'not_inst'],
    'Region':                    ['North', 'south', 'central', 'North-East'],
    'Security_Type':             ['direct', 'Indriect'],
}


def preprocess_input(input_dict: dict, le_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Takes a dictionary of raw user inputs (from Streamlit form),
    encodes categorical variables using saved LabelEncoders,
    and returns a DataFrame ready for model prediction.

    Parameters
    ----------
    input_dict    : dict   Raw input from Streamlit widgets
    le_dict       : dict   Saved LabelEncoder objects (one per categorical column)
    feature_names : list   Ordered list of features the model was trained on

    Returns
    -------
    pd.DataFrame  : Single-row DataFrame with all features encoded
    """
    row = {}

    for col in feature_names:
        val = input_dict.get(col, None)

        if col in le_dict:
            # Encode categorical using saved LabelEncoder
            try:
                encoded = le_dict[col].transform([str(val)])[0]
            except ValueError:
                # Unseen category — use 0 as fallback
                encoded = 0
            row[col] = encoded
        else:
            # Numeric column — use directly
            row[col] = float(val) if val is not None else 0.0

    return pd.DataFrame([row])[feature_names]


def get_risk_category(probability: float) -> tuple:
    """
    Converts a default probability into a risk tier and decision.

    Returns
    -------
    (risk_label, decision, color)
    """
    if probability < 0.25:
        return ('LOW RISK',    'APPROVE',     '#4DD9AC')   # green
    elif probability < 0.50:
        return ('MEDIUM RISK', 'REVIEW',      '#F5A623')   # amber
    elif probability < 0.70:
        return ('HIGH RISK',   'REJECT',      '#E05C5C')   # red
    else:
        return ('CRITICAL',    'REJECT',      '#B30000')   # dark red


def compute_ltv(loan_amount: float, property_value: float) -> float:
    """Calculate Loan-to-Value ratio."""
    if property_value > 0:
        return round((loan_amount / property_value) * 100, 2)
    return 0.0
