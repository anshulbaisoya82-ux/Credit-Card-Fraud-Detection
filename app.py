import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / 'XGBoost.pkl')
scaler = joblib.load(BASE_DIR / 'standard_scaler.pkl')
encoder = joblib.load(BASE_DIR / 'onehot_encoder.pkl')
feature_cols = joblib.load(BASE_DIR / 'feature_columns.pkl')

PAYMENT_TYPES = list(encoder.categories_[0])
EMPLOYMENT_STATUSES = list(encoder.categories_[1])
HOUSING_STATUSES = list(encoder.categories_[2])
SOURCES = list(encoder.categories_[3])
DEVICE_OSS = list(encoder.categories_[4])

st.set_page_config(page_title="Fraud Risk Checker", page_icon="🛡️", layout="wide")
st.title("🛡️ Credit Card Fraud Risk Checker")
st.markdown("Fill in the applicant details below and click **Predict** to assess fraud risk.")

with st.form("prediction_form"):
    st.subheader("📊 Numerical Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Income (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        name_email_similarity = st.number_input("Name / Email Similarity (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        current_address_months_count = st.number_input("Months at Current Address", min_value=0, value=12)
        customer_age = st.number_input("Customer Age", min_value=18, value=30)
        days_since_request = st.number_input("Days Since Request", min_value=0.0, value=1.0, step=0.1)
        intended_balcon_amount = st.number_input("Intended Balance Amount", value=100.0, step=1.0)
        zip_count_4w = st.number_input("Zip Code Count (4 weeks)", min_value=0, value=5)
        velocity_24h = st.number_input("Velocity (24 h)", min_value=0.0, value=200.0)

    with col2:
        bank_branch_count_8w = st.number_input("Bank Branch Count (8 weeks)", min_value=0, value=3)
        date_of_birth_distinct_emails_4w = st.number_input("Distinct Emails linked to DOB (4 weeks)", min_value=0, value=1)
        credit_risk_score = st.number_input("Credit Risk Score", value=150)
        bank_months_count = st.number_input("Months with Bank", min_value=0, value=24)
        proposed_credit_limit = st.number_input("Proposed Credit Limit", min_value=0.0, value=1000.0, step=100.0)
        session_length_in_minutes = st.number_input("Session Length (minutes)", min_value=0.0, value=10.0, step=0.5)
        device_distinct_emails_8w = st.number_input("Distinct Emails on Device (8 weeks)", min_value=0, value=1)
        month = st.number_input("Application Month", min_value=1, max_value=12, value=3)

    with col3:
        st.subheader("✅ Boolean Flags")
        email_is_free = st.checkbox("Free Email Provider?")
        phone_home_valid = st.checkbox("Home Phone Valid?")
        phone_mobile_valid = st.checkbox("Mobile Phone Valid?")
        has_other_cards = st.checkbox("Has Other Cards?")
        foreign_request = st.checkbox("Foreign Request?")
        keep_alive_session = st.checkbox("Keep-Alive Session?")

    st.subheader("🗂️ Categorical Features")
    cat1, cat2, cat3, cat4, cat5 = st.columns(5)
    with cat1:
        payment_type = st.selectbox("Payment Type", PAYMENT_TYPES)
    with cat2:
        employment_status = st.selectbox("Employment Status", EMPLOYMENT_STATUSES)
    with cat3:
        housing_status = st.selectbox("Housing Status", HOUSING_STATUSES)
    with cat4:
        source = st.selectbox("Application Source", SOURCES)
    with cat5:
        device_os = st.selectbox("Device OS", DEVICE_OSS)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

if submitted:
    if income == 0:
        st.warning("⚠️ Income cannot be zero for credit-to-income ratio calculation. Please enter a valid income value.")
        st.stop()

    credit_to_income_ratio = proposed_credit_limit / (income * 100)

    numerical_columns = {
        "income": income,
        "name_email_similarity": name_email_similarity,
        "current_address_months_count": current_address_months_count,
        "customer_age": customer_age,
        "days_since_request": days_since_request,
        "intended_balcon_amount": intended_balcon_amount,
        "zip_count_4w": zip_count_4w,
        "velocity_24h": velocity_24h,
        "bank_branch_count_8w": bank_branch_count_8w,
        "date_of_birth_distinct_emails_4w": date_of_birth_distinct_emails_4w,
        "credit_risk_score": credit_risk_score,
        "email_is_free": int(email_is_free),
        "phone_home_valid": int(phone_home_valid),
        "phone_mobile_valid": int(phone_mobile_valid),
        "bank_months_count": bank_months_count,
        "has_other_cards": int(has_other_cards),
        "proposed_credit_limit": proposed_credit_limit,
        "foreign_request": int(foreign_request),
        "session_length_in_minutes": session_length_in_minutes,
        "keep_alive_session": int(keep_alive_session),
        "device_distinct_emails_8w": device_distinct_emails_8w,
        "month": month,
        "credit_to_income_ratio": credit_to_income_ratio,
    }

    numeric_df = pd.DataFrame([numerical_columns])
    categorical_df = pd.DataFrame(
        [[payment_type, employment_status, housing_status, source, device_os]],
        columns=["payment_type", "employment_status", "housing_status", "source", "device_os"]
    )

    encoded_arr = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out())

    model_df = pd.concat([encoded_df, numeric_df], axis=1)
    model_df = model_df.reindex(columns=feature_cols, fill_value=0)

    scaled_arr = scaler.transform(model_df)
    prediction = model.predict(scaled_arr)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("🚨 **Prediction: FRAUD DETECTED** — This transaction is flagged as high risk.")
    else:
        st.success("✅ **Prediction: NOT FRAUD** — This transaction appears legitimate.")
