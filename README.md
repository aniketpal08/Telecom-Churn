# Telco Customer Churn Prediction

A machine learning web application built with **Flask** that predicts whether a telecom customer is likely to churn.

The app supports **two modes**:
1. **Single Customer Prediction** — manually enter customer details and predict churn.
2. **Bulk Prediction** — upload a **CSV or Excel file** containing multiple customers and get churn predictions for all rows.

---

## Dataset
- **Dataset:** Telco Customer Churn Dataset
- **Target Variable:** `Churn`

---

## Features Used

### Numerical Features
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

### Categorical Features
- `InternetService`
- `PaymentMethod`
- `PaperlessBilling`
- `Contract`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `Partner`
- `Dependents`

---

## Model Used
- **Logistic Regression**
- **Preprocessing Pipeline**
  - Median imputation for numerical features
  - Most frequent imputation for categorical features
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features

---

## Project Features
- Single customer churn prediction
- Bulk prediction from **CSV / Excel**
- Automatic preprocessing using saved pipeline
- Column validation for uploaded files
- Output file download with:
  - `Predicted_Churn`
  - `Churn_Probability`

---

## Project Structure

```bash
Telecom-Churn/
│
├── app.py
├── train_model.py
├── model.pkl
├── pipeline.pkl
├── requirements.txt
├── README.md
├── sample_bulk.csv
├── Telco Customer Churn.csv
│
├── templates/
│   ├── home.html
│   ├── single_predict.html
│   └── bulk_upload.html
│
└── static/
    └── style.css