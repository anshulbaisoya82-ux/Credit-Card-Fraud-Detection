import joblib
import numpy as np
import pandas as pd
import streamlit as st

## Loading models 
model = joblib.load('XGboost.pkl')
scaler = joblib.load('standard_scaler.pkl')
encoder = joblib.load('onehot_encoder.pkl')
feature = joblib.load('feature_columns.pkl')
dropped_columns = joblib.load('dropped_columns.pkl')

st.title("Fraud Risk Checker")

# Taking inputs

income = st.number_input("Income (0 to 1)")
name_email_similarity = st.number_input("Name/email similarity (0 to 1)")
current_address_months_count = st.number_input("Months at current address")
customer_age = st.number_input("Customer age")
days_since_request = st.number_input("Days since request")
intended_balcon_amount = st.number_input("Intended balance amount")
zip_count_4w = st.number_input("Zip code count (4 weeks)")
velocity_24h = st.number_input("Velocity (24h)")
bank_branch_count_8w = st.number_input("Bank branch count (8 weeks)")
date_of_birth_distinct_emails_4w = st.number_input("Distinct emails linked to DOB (4 weeks)")
credit_risk_score = st.number_input("Credit risk score")
bank_months_count = st.number_input("Months with bank")
proposed_credit_limit = st.number_input("Proposed credit limit")
session_length_in_minutes = st.number_input("Session length (minutes)")
device_distinct_emails_8w = st.number_input("Distinct emails on device (8 weeks)")
month = st.number_input("Application month")

email_is_free = st.checkbox("Free email provider?")
phone_home_valid = st.checkbox("Home phone valid?")
phone_mobile_valid = st.checkbox("Mobile phone valid?")
has_other_cards = st.checkbox("Has other cards?")
foreign_request = st.checkbox("Foreign request?")
keep_alive_session = st.checkbox("Keep-alive session?")

payment_type = st.text_input("Payment type")
employment_status = st.text_input("Employment status")
housing_status = st.text_input("Housing status")
source = st.text_input("Application source")
device_os = st.text_input("Device OS")

if st.button("Predict") :
    credit_to_income_ratio = proposed_credit_limit/(income*100)

    numerical_columns = {
        "income" : income,
        "name_email_similarity" : name_email_similarity,
        "current_address_months_count" : current_address_months_count,
        "customer_age" : customer_age,
        "days_since_request" : days_since_request,
        "intended_balcon_amount" : intended_balcon_amount,
        "zip_count_4w" : zip_count_4w,
        "velocity_24h" : velocity_24h,
        "bank_branch_count_8w" : bank_branch_count_8w,
        "date_of_birth_distinct_emails_4w" : date_of_birth_distinct_emails_4w,
        "credit_risk_score" : credit_risk_score,
        "email_is_free" : int(email_is_free),
        "phone_home_valid" : int(phone_home_valid),
        "phone_mobile_valid" : int(phone_mobile_valid),
        "bank_months_count" : bank_months_count,
        "has_other_cards" : int(has_other_cards),
        "proposed_credit_limit" : proposed_credit_limit,
        "foreign_request" : int(foreign_request),
        "session_length_in_minutes" : session_length_in_minutes,
        "keep_alive_session" : int(keep_alive_session),
        "device_distinct_emails_8w" : device_distinct_emails_8w,
        "month" : month,
        "credit_to_income_ratio" : credit_to_income_ratio
    }
    numeric_df = pd.DataFrame([numerical_columns])
    categorical_df = pd.DataFrame([[payment_type, employment_status, housing_status, source, device_os]],columns=['payment_type','employment_status','housing_status','source','device_os'])
    encoded_colums = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(encoded_colums, columns= encoder.get_feature_names_out())

    model_df = pd.concat([encoded_df, numeric_df], axis=1)

    scaled_df = scaler.transform(model_df)

    prediction = model.predict(scaled_df)

    if prediction == 1:
        st.error("Prediction: FRAUD")
    else:
        st.success("Prediction: NOT FRAUD")


