import joblib
import pandas as pd


def main():
    # Load saved pipeline
    model = joblib.load("models/churn_pipeline.pkl")

    # Example one customer input
    sample_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])

    prediction = model.predict(sample_customer)

    if prediction[0] == 1:
        print("Customer is likely to churn.")
    else:
        print("Customer is likely to stay.")


if __name__ == "__main__":
    main()