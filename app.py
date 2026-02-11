import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="FraudSentinels", layout="centered")

if not (os.path.exists("fraud_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("feature_cols.pkl")):
    st.error("❌ Required model files not found. Please ensure fraud_model.pkl, scaler.pkl, and feature_cols.pkl are present.")
    st.stop()

model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    st.markdown("""
    <style>
    .welcome-container {
        background: linear-gradient(135deg, #ff512f, #dd2476, #24c6dc);
        padding: 80px 30px;
        border-radius: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
    }
    .welcome-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .welcome-subtitle {
        font-size: 20px;
        margin-bottom: 28px;
        opacity: 0.95;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">🚨 Welcome to Insurance Fraud Detection</div>
        <div class="welcome-subtitle">
            AI-Powered System to Detect Fraudulent Insurance Claims
        </div>
        <p>This application analyzes claim details using ML to estimate fraud probability.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Get Started"):
        st.session_state.page = "predict"
        st.rerun()
else:
    st.title("🛡️ Insurance Fraud Detection")

    age = st.number_input("Age", 18, 100, 30)
    insured_sex = st.selectbox("Gender", ["MALE", "FEMALE"])
    policy_state = st.selectbox("Policy State", ["OH", "IL", "IN"])
    policy_deductable = st.selectbox("Policy Deductible", [500, 1000, 2000])
    policy_annual_premium = st.number_input("Annual Premium", 200, 10000, 1200)

    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car"])
    incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Major Damage", "Total Loss"])
    number_of_vehicles_involved = st.slider("Number of Vehicles Involved", 1, 4, 1)
    police_report_available = st.selectbox("Police Report Available", ["YES", "NO"])

    total_claim_amount = st.number_input("Total Claim Amount", 0, 200000, 30000)
    injury_claim = st.number_input("Injury Claim Amount", 0, 100000, 10000)
    property_claim = st.number_input("Property Claim Amount", 0, 100000, 5000)
    vehicle_claim = st.number_input("Vehicle Claim Amount", 0, 100000, 15000)

    if st.button("🔍 Predict Fraud"):
        data = {
            'age': age,
            'insured_sex': insured_sex,
            'policy_state': policy_state,
            'policy_deductable': policy_deductable,
            'policy_annual_premium': policy_annual_premium,
            'incident_type': incident_type,
            'incident_severity': incident_severity,
            'number_of_vehicles_involved': number_of_vehicles_involved,
            'police_report_available': police_report_available,
            'total_claim_amount': total_claim_amount,
            'injury_claim': injury_claim,
            'property_claim': property_claim,
            'vehicle_claim': vehicle_claim
        }

        new_df = pd.DataFrame([data])
        new_df['age_group'] = pd.cut(new_df['age'], bins=[18,30,45,60,100], labels=[0,1,2,3]).astype(int)
        new_df['high_claim_flag'] = np.where(new_df['total_claim_amount'] > 20000, 1, 0)
        new_df = pd.get_dummies(new_df)
        new_df = new_df.reindex(columns=feature_cols, fill_value=0)
        new_scaled = scaler.transform(new_df)
        prob = model.predict_proba(new_scaled)[0][1]

        st.metric("Fraud Probability", f"{round(prob*100, 2)} %")
        if prob >= 0.35:
            st.error("⚠️ Fraud Detected")
        else:
            st.success("✅ Genuine Claim")
