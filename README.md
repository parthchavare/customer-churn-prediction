# Customer Churn Prediction

This project predicts whether a customer is likely to leave a telecom service using Machine Learning. The system analyzes customer details such as tenure, contract type, internet service, monthly charges, and payment method to identify churn risk.

A complete end-to-end machine learning workflow was implemented, including data cleaning, exploratory data analysis (EDA), preprocessing, model training, evaluation, pipeline saving, and deployment using Streamlit.

---

# Problem Statement

Customer churn is one of the major challenges faced by subscription-based businesses. Retaining existing customers is often more cost-effective than acquiring new ones.

The goal of this project is to build a machine learning model that can predict whether a customer is likely to churn based on their service usage and account information.

---

# Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

Target Column:
- `Churn`
  - Yes → Customer is likely to leave
  - No → Customer is likely to stay

---

# Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# Project Workflow

## 1. Data Loading
The dataset was loaded using Pandas.

## 2. Data Cleaning
The following preprocessing steps were performed:
- Removed `customerID`
- Converted `TotalCharges` to numeric
- Handled missing values

## 3. Exploratory Data Analysis (EDA)
EDA was performed to understand customer behavior and identify patterns related to churn.

Visualizations included:
- Churn distribution
- Monthly charges distribution
- Tenure analysis
- Churn vs contract type
- Correlation heatmap

## 4. Data Preprocessing
The following preprocessing techniques were used:
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- Train-Test Split

## 5. Model Training
The following models were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

## 6. Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 7. Pipeline Creation
A complete machine learning pipeline was created using:
- ColumnTransformer
- Pipeline
- RandomForestClassifier

The final trained pipeline was saved using Joblib.

## 8. Deployment
A Streamlit web application was created for interactive customer churn prediction.

---

# Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── requirements.txt
└── README.md