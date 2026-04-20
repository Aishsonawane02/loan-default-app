"""
app.py — Loan Default Prediction Streamlit App
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Add utils to path ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from preprocessing import (
    preprocess_input, get_risk_category, compute_ltv,
    CATEGORICAL_OPTIONS, FEATURE_LABELS, NUMERIC_COLS, CATEGORICAL_COLS
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #E8EAF0; }
    .stApp { background-color: #E8EAF0; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #E8EAF0; }
    .metric-card {
        background: #131929;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #1E2740;
        margin: 8px 0;
    }
    .approve  { border-left: 5px solid #4DD9AC; }
    .review   { border-left: 5px solid #F5A623; }
    .reject   { border-left: 5px solid #E05C5C; }
    .critical { border-left: 5px solid #B30000; }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model     = joblib.load(os.path.join(base, 'model', 'loan_model.pkl'))
    scaler    = joblib.load(os.path.join(base, 'model', 'scaler.pkl'))
    encoders  = joblib.load(os.path.join(base, 'model', 'encoders.pkl'))
    feat_names= joblib.load(os.path.join(base, 'model', 'feature_names.pkl'))
    threshold = joblib.load(os.path.join(base, 'model', 'threshold.pkl'))
    return model, scaler, encoders, feat_names, threshold

try:
    model, scaler, encoders, feature_names, threshold = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model: {e}")
    model_loaded = False

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🏦 Loan Default Prediction System")
st.markdown("*AI-powered credit risk assessment for lending decisions*")
st.divider()

# ── Sidebar — Applicant Info ──────────────────────────────────
st.sidebar.markdown("## 📋 Applicant Information")
st.sidebar.markdown("Fill in all fields and click **Predict** to get the risk assessment.")

with st.sidebar:
    st.markdown("### 💰 Loan Details")
    loan_amount    = st.number_input("Loan Amount ($)",      min_value=1000,   max_value=5000000, value=250000, step=5000)
    property_value = st.number_input("Property Value ($)",   min_value=1000,   max_value=9999999, value=350000, step=10000)
    term           = st.selectbox("Loan Term (months)",      [60, 120, 180, 240, 300, 360], index=5)
    loan_type      = st.selectbox("Loan Type",               CATEGORICAL_OPTIONS['loan_type'])
    loan_purpose   = st.selectbox("Loan Purpose",            CATEGORICAL_OPTIONS['loan_purpose'])

    # Auto-calculate LTV
    ltv = compute_ltv(loan_amount, property_value)
    st.info(f"📊 Loan-to-Value (LTV): **{ltv:.1f}%**")

    st.markdown("### 👤 Applicant Profile")
    income        = st.number_input("Annual Income ($)",     min_value=0, max_value=9999999, value=75000, step=1000)
    Credit_Score  = st.slider("Credit Score",                min_value=300, max_value=850, value=680)
    age           = st.selectbox("Age Group",                CATEGORICAL_OPTIONS['age'])
    Gender        = st.selectbox("Gender",                   CATEGORICAL_OPTIONS['Gender'])
    dtir1         = st.slider("Debt-to-Income Ratio (%)",    min_value=0.0, max_value=100.0, value=35.0, step=0.5)

    st.markdown("### 📄 Credit & Application")
    Credit_Worthiness = st.selectbox("Credit Worthiness",   CATEGORICAL_OPTIONS['Credit_Worthiness'])
    credit_type       = st.selectbox("Credit Bureau",        CATEGORICAL_OPTIONS['credit_type'])
    open_credit       = st.selectbox("Open Credit?",         CATEGORICAL_OPTIONS['open_credit'])
    approv_in_adv     = st.selectbox("Pre-Approved?",        CATEGORICAL_OPTIONS['approv_in_adv'])

    st.markdown("### 🏠 Property & Loan Structure")
    loan_limit           = st.selectbox("Loan Limit Type",      CATEGORICAL_OPTIONS['loan_limit'])
    occupancy_type       = st.selectbox("Occupancy Type",       CATEGORICAL_OPTIONS['occupancy_type'])
    Secured_by           = st.selectbox("Secured By",           CATEGORICAL_OPTIONS['Secured_by'])
    construction_type    = st.selectbox("Construction Type",    CATEGORICAL_OPTIONS['construction_type'])
    total_units          = st.selectbox("Total Units",          CATEGORICAL_OPTIONS['total_units'])
    Security_Type        = st.selectbox("Security Type",        CATEGORICAL_OPTIONS['Security_Type'])
    Region               = st.selectbox("Region",               CATEGORICAL_OPTIONS['Region'])

    st.markdown("### ⚙️ Loan Features")
    Neg_ammortization    = st.selectbox("Negative Amortization?", CATEGORICAL_OPTIONS['Neg_ammortization'])
    interest_only        = st.selectbox("Interest Only?",         CATEGORICAL_OPTIONS['interest_only'])
    lump_sum_payment     = st.selectbox("Lump Sum Payment?",      CATEGORICAL_OPTIONS['lump_sum_payment'])
    business_or_commercial = st.selectbox("Business/Commercial?", CATEGORICAL_OPTIONS['business_or_commercial'])

    st.markdown("### 👥 Co-applicant")
    co_credit = st.selectbox("Co-applicant Credit Bureau", CATEGORICAL_OPTIONS['co-applicant_credit_type'])
    submission = st.selectbox("Submission Method",         CATEGORICAL_OPTIONS['submission_of_application'])

    predict_btn = st.button("🔍 Predict Default Risk", type="primary", use_container_width=True)

# ── Main Panel ────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("## 📊 Risk Assessment")

    if predict_btn and model_loaded:
        # Build input dict
        input_dict = {
            'loan_limit': loan_limit,
            'Gender': Gender,
            'approv_in_adv': approv_in_adv,
            'loan_type': loan_type,
            'loan_purpose': loan_purpose,
            'Credit_Worthiness': Credit_Worthiness,
            'open_credit': open_credit,
            'business_or_commercial': business_or_commercial,
            'loan_amount': loan_amount,
            'term': term,
            'Neg_ammortization': Neg_ammortization,
            'interest_only': interest_only,
            'lump_sum_payment': lump_sum_payment,
            'property_value': property_value,
            'construction_type': construction_type,
            'occupancy_type': occupancy_type,
            'Secured_by': Secured_by,
            'total_units': total_units,
            'income': income,
            'credit_type': credit_type,
            'Credit_Score': Credit_Score,
            'co-applicant_credit_type': co_credit,
            'age': age,
            'submission_of_application': submission,
            'LTV': ltv,
            'Region': Region,
            'Security_Type': Security_Type,
            'dtir1': dtir1,
        }

        # Preprocess
        X_input = preprocess_input(input_dict, encoders, feature_names)

        # Predict
        prob_default = model.predict_proba(X_input)[0][1]
        risk_label, decision, color = get_risk_category(prob_default)

        # ── Result Cards ─────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Default Probability", f"{prob_default*100:.1f}%")
        with r2:
            st.metric("Risk Category", risk_label)
        with r3:
            st.metric("Decision", decision)

        st.divider()

        # ── Probability Gauge ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 2))
        fig.patch.set_facecolor("#894E2A")
        ax.set_facecolor("#C5C359")
        # Background bar
        ax.barh(['Default Risk'], [100], color='#1E2740', height=0.5, edgecolor='none')
        # Filled bar
        bar_color = color
        ax.barh(['Default Risk'], [prob_default*100], color=bar_color, height=0.5, edgecolor='none')
        # Threshold line
        ax.axvline(threshold*100, color='white', lw=2, ls='--', label=f'Threshold ({threshold*100:.0f}%)')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Default Probability (%)', color='#E8EAF0')
        ax.tick_params(colors='#E8EAF0')
        ax.legend(facecolor="#1C7D14", labelcolor='#E8EAF0', fontsize=9)
        ax.text(prob_default*100 + 1, 0, f'{prob_default*100:.1f}%',
                va='center', color=color, fontweight='bold', fontsize=12)
        for sp in ax.spines.values(): sp.set_edgecolor('#1E2740')
        st.pyplot(fig)
        plt.close()

        # ── Risk breakdown ────────────────────────────────────
        st.markdown("### 🔍 Risk Factor Breakdown")
        risk_factors = []
        if ltv > 80:
            risk_factors.append(("⚠️ High LTV", f"{ltv:.1f}% — above 80% is higher risk"))
        if Credit_Score < 600:
            risk_factors.append(("⚠️ Low Credit Score", f"{Credit_Score} — below 600 indicates risk"))
        if dtir1 > 43:
            risk_factors.append(("⚠️ High DTI Ratio", f"{dtir1:.1f}% — above 43% is lender threshold"))
        if interest_only == 'int_only':
            risk_factors.append(("⚠️ Interest-Only Loan", "Higher risk loan structure"))
        if Neg_ammortization == 'neg_amm':
            risk_factors.append(("⚠️ Negative Amortization", "Balance grows over time — high risk"))
        if loan_type in ['type2', 'type3']:
            risk_factors.append(("⚠️ Non-Standard Loan Type", "Higher risk than type1"))

        if risk_factors:
            for flag, desc in risk_factors:
                st.warning(f"**{flag}:** {desc}")
        else:
            st.success("✅ No major risk flags detected for this applicant.")

        # ── Key metrics ───────────────────────────────────────
        st.markdown("### 📋 Application Summary")
        summary_df = pd.DataFrame({
            'Metric': ['Loan Amount', 'Property Value', 'LTV', 'Annual Income',
                       'Credit Score', 'DTI Ratio', 'Loan Term', 'Decision'],
            'Value': [f'${loan_amount:,.0f}', f'${property_value:,.0f}', f'{ltv:.1f}%',
                      f'${income:,.0f}', str(Credit_Score), f'{dtir1:.1f}%',
                      f'{term} months', f'{decision} ({risk_label})']
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    elif predict_btn and not model_loaded:
        st.error("Model not loaded. Please check the model files in /model/ directory.")
    else:
        st.info("👈 Fill in the applicant details in the sidebar and click **Predict Default Risk**.")
        st.markdown("""
        #### How this works:
        1. **Fill** in all applicant fields in the sidebar
        2. **Click** Predict Default Risk
        3. **Review** the default probability, risk tier, and flags
        4. **Decide** to Approve, Review, or Reject the application

        #### Risk Tiers:
        | Probability | Risk Level | Decision |
        |---|---|---|
        | 0% – 25%  | 🟢 Low Risk  | Approve |
        | 25% – 50% | 🟡 Medium Risk | Review |
        | 50% – 70% | 🔴 High Risk | Reject |
        | 70%+      | 🚨 Critical | Reject |
        """)

with col_right:
    st.markdown("## 📈 Model Information")

    st.markdown("""
    #### Model Used: Gradient Boosting Classifier
    | Property | Value |
    |---|---|
    | Training samples | 118,936 |
    | Test samples | 29,734 |
    | ROC-AUC | **0.889** |
    | F1 Score | **0.749** |
    | Precision | **0.882** |
    | Recall | **0.651** |
    | Default Rate | 24.6% |
    """)

    st.markdown("#### Top Predictive Features")
    features_display = {
        'property_value':            '29.9% importance',
        'LTV (Loan-to-Value)':       '28.6% importance',
        'dtir1 (Debt-to-Income)':    '10.8% importance',
        'income':                    '5.3% importance',
        'lump_sum_payment':          '4.1% importance',
        'Neg_ammortization':         '4.0% importance',
        'credit_type':               '3.6% importance',
    }
    for feat, imp in features_display.items():
        st.markdown(f"- **{feat}**: {imp}")

