import streamlit as st
import pandas as pd
import cloudpickle
import json
import os

# 1. Setup paths and load configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "..", "app_config.json")
bundle_path = os.path.join(BASE_DIR, "..", "models", "deployment_bundle.pkl")
model_path = os.path.join(BASE_DIR, "..", "models", "best_gb_model.pkl")

with open(config_path, "r") as f:
    ui_config = json.load(f)

with open(bundle_path, "rb") as f:
    bundle = cloudpickle.load(f)

with open(model_path, "rb") as f:
    model = cloudpickle.load(f)

scaler_features = [
    'sub_grade', 'term', 'home_ownership', 'fico_range_low', 'total_acc', 
    'pub_rec', 'revol_util', 'annual_inc', 'int_rate', 'dti', 'purpose',
    'mort_acc', 'loan_amnt', 'application_type', 'installment',
    'verification_status', 'pub_rec_bankruptcies', 'addr_state', 'initial_list_status', 
    'fico_range_high', 'revol_bal', 'open_acc', 'emp_length', 
    'time_to_earliest_cr_line', 'issue_year', 'issue_month'
]

# Function to style labels
def style_labels(option):
    return str(option).replace("_", " ").title()

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("🏦 Loan Default Prediction")
st.markdown("Predict the probability of a loan defaulting based on borrower attributes.")

# 2. Create the Input Form
with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=float(1))
        int_rate = st.number_input("Interest Rate (%)", min_value=float(0))
        term_str = st.selectbox("Term", options = ui_config["mappings"]["term"], format_func=style_labels)
        annual_inc = st.number_input("Annual Income", min_value=0.0)

    with col2:
        home_ownership = st.selectbox("Home Ownership", ui_config["categorical"]["home_ownership"], format_func=style_labels)
        purpose = st.selectbox("Loan Purpose", ui_config["categorical"]["purpose"], format_func=style_labels)
        dti = st.number_input("DTI Ratio", min_value=0.0, max_value=60.0)
        init_list_status = st.selectbox("Initial List Status (F -> Fractional, W -> Whole)", ui_config["categorical"]["initial_list_status"], format_func=style_labels)

    with col3:
        ver_status = st.selectbox("Verification Status", ui_config["categorical"]["verification_status"], format_func=style_labels)
        app_type = st.selectbox("Application Type", ui_config["categorical"]["application_type"], format_func=style_labels)
        mort_acc = st.number_input("Mortgage Accounts", min_value=0)
        pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0)

    

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        issue_date = st.date_input("Loan Issue Date")

    with col5:
        fico_low = st.number_input("FICO Range Low", min_value=600, max_value=850, value=700)
    
    with col6:
        fico_high = st.number_input("FICO Range High", min_value=600, max_value=850, value=740)
    
    with col7:
        time_to_cr = st.number_input("Time to Earliest Credit Line", min_value=0)

    submit = st.form_submit_button("Analyse Risk")

    if submit:
        # --- Feature Engineering ---
        calc_issue_year = issue_date.year
        calc_issue_month = issue_date.month
        term_val = 3.0 if "36" in term_str else 5.0


        # --- Create DataFrame ---
        
        input_dict = {
            # The 15 real features
            'int_rate': int_rate, 'term': term_val, 'home_ownership': home_ownership,
            'dti': dti, 'annual_inc': annual_inc,
            'pub_rec_bankruptcies': pub_rec_bankruptcies, 'verification_status': ver_status,
            'initial_list_status': init_list_status, 'mort_acc': mort_acc, 'purpose': purpose,
            'application_type': app_type, 'time_to_earliest_cr_line': time_to_cr,
            'issue_year': calc_issue_year, 'loan_amnt': loan_amnt,
            
            # The 11 "rubbish" features to satisfy the scaler
            'sub_grade': ui_config["categorical"]["sub_grade"][0], 
            'fico_range_low': fico_low, 
            'fico_range_high': fico_high,
            'total_acc': ui_config["numerical"]["total_acc"]["default"], 
            'pub_rec': ui_config["numerical"]["pub_rec"]["default"], 
            'revol_util': ui_config["numerical"]["revol_util"]["default"], 
            'revol_bal': ui_config["numerical"]["revol_bal"]["default"], 
            'open_acc': ui_config["numerical"]["open_acc"]["default"], 
            'emp_length': ui_config["numerical"]["emp_length"]["default"], 
            'addr_state': ui_config["categorical"]["addr_state"][0],
            'issue_month': calc_issue_month,
            'installment': ui_config["numerical"]["installment"]["default"]
        }

        input_df = pd.DataFrame([input_dict])[scaler_features]

        # --- Outlier Handling ---
        for feat, bounds in ui_config["outlier_bounds"].items():
            if feat in bundle['outlier_features'] and bundle['features_selected']:
                input_df[feat] = input_df[feat].clip(lower = bounds['lower'], upper = bounds['upper'])
        
        # --- Outlier clipping for dti ---
        input_df['dti'] = input_df['dti'].clip(upper=60.0)
        
        # --- Encode & Scale ---
        for col, le in bundle['label_encoders'].items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        X_scaled = bundle['scaler'].transform(input_df)

        # --- Feature Selection ---
        X_scaled = pd.DataFrame(X_scaled, columns = scaler_features)
        X_scaled['fico_score_avg'] = (X_scaled['fico_range_low'] + X_scaled['fico_range_high']) / 2
        X_scaled = X_scaled[bundle['features_selected']]

        # --- Prediction ---
        prob = model.predict_proba(X_scaled)[:, 1][0]
        risk_pct = prob * 100

        # Determine Color and Label
        if risk_pct < 30:
            color = "#28a745"  # Green
            status = "Low Risk - Approved"
        elif risk_pct < 60:
            color = "#ff8c00"  # Dark Orange
            status = "Moderate Risk - Manual Review"
        else:
            color = "#dc3545"  # Red
            status = "High Risk - Likely Default"

        # --- Display Results ---
        st.divider()
        st.subheader("Risk Assessment")

        # Custom HTML for the colored box
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-family: sans-serif;
                ">
                <h2 style="margin:0;">{status}</h1>
                <h3 style="margin:0; opacity: 0.9;">Probability of Default: {risk_pct:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

