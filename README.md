# Telco Customer Churn Prediction

A Flask-based machine learning web app that predicts whether a telecom customer is likely to churn.

The app supports two modes:
- **Single Prediction** — enter one customer’s details manually
- **Bulk Prediction** — upload a CSV or Excel file to predict churn for multiple customers at once

---

## Dataset
- **Dataset:** Telco Customer Churn
- **Target variable:** `Churn`

---

## Features Used

### Numerical
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

### Categorical
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

## Model
- **Logistic Regression**

### Preprocessing
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling for numerical columns
- One-hot encoding for categorical columns

---

## App Features
- Single customer churn prediction
- Bulk prediction using **CSV / Excel**
- Automatic preprocessing through a saved pipeline
- Uploaded file column validation
- Downloadable output file with:
  - `Predicted_Churn`
  - `Churn_Probability`

---

## Project Structure

```bash
Telecom-Churn/
│
├── app.py
├── train_model.py
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
