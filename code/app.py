
import streamlit as st
import pandas as pd
import joblib

# Load saved model
saved = joblib.load("churn_model.joblib")
model = saved["model"]
preprocessor = saved["preprocessor"]
threshold = saved["threshold"]

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate their likelihood of leaving."
)

st.info(f"Current model: {saved['model_name']}")

with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior citizen", ["No", "Yes"]
        )
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        tenure_months = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=2
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        phone_service = st.selectbox(
            "Phone service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )

    with col3:
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )
        paperless_billing = st.selectbox(
            "Paperless billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.subheader("Charges")

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=85.0
        )

    with col5:
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=170.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

if submitted:

    customer = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }

    # Create tenure group using training bins
    customer_df = pd.DataFrame([customer])

    customer_df["tenure_group"] = pd.cut(
        customer_df["tenure_months"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-60 months",
            "60+ months"
        ]
    )

    # Preprocess and predict
    processed = preprocessor.transform(customer_df)

    probability = float(model.predict_proba(processed)[0, 1])
    prediction = probability >= threshold

    st.divider()
    st.subheader("Prediction Result")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Estimated churn probability",
            f"{probability:.2%}"
        )

    with metric_col2:
        st.metric(
            "Decision threshold",
            f"{threshold:.0%}"
        )

    if prediction:
        st.error("⚠️ Likely to churn")
        st.write(
            "Consider reviewing this customer's needs and "
            "whether an appropriate retention offer could help."
        )
    else:
        st.success("✅ Likely to stay")
        st.write(
            "The model predicts a lower churn risk at the "
            "current decision threshold."
        )

    st.caption(
        "This is a machine-learning estimate, not a guarantee. "
        "It should not be the sole basis for customer decisions."
    )
