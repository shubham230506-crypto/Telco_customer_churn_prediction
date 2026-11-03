
import joblib
import pandas as pd

# Load the saved model and preprocessing pipeline
saved = joblib.load("churn_model.joblib")

model = saved["model"]
preprocessor = saved["preprocessor"]
threshold = saved["threshold"]

print("Loaded model:", saved["model_name"])


def predict_churn(customer):
    """
    customer: dictionary containing customer details
    """

    # Convert customer details into a DataFrame
    customer_df = pd.DataFrame([customer])

    # Create tenure group using the same bins as training
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

    # Apply the saved preprocessing pipeline
    processed = preprocessor.transform(customer_df)

    # Get probability of churn
    churn_probability = model.predict_proba(processed)[0, 1]

    # Apply the saved decision threshold
    prediction = int(churn_probability >= threshold)

    if prediction == 1:
        result = "Likely to churn"
    else:
        result = "Likely to stay"

    return {
        "prediction": result,
        "churn_probability": round(float(churn_probability) * 100, 2)
    }


# Example customer
customer = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "tenure_months": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
    "total_charges": 170.0
}

result = predict_churn(customer)

print("\n========== CHURN PREDICTION ==========")
print("Prediction:", result["prediction"])
print("Churn probability:", f'{result["churn_probability"]}%')
