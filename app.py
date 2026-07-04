from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
OUTPUT_FILE = "bulk_predictions.csv"

# ---------------- LOAD MODEL + PIPELINE ----------------
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# ---------------- FEATURES USED BY MODEL ----------------
num_attribs = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_attribs = [
    'InternetService', 'PaymentMethod', 'PaperlessBilling', 'Contract',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'Partner', 'Dependents'
]

required_cols = num_attribs + cat_attribs

# ---------------- COLUMN NORMALIZATION MAP ----------------
column_mapping = {
    "tenure": "tenure",

    "monthlycharges": "MonthlyCharges",
    "monthly_charges": "MonthlyCharges",
    "monthly charges": "MonthlyCharges",

    "totalcharges": "TotalCharges",
    "total_charges": "TotalCharges",
    "total charges": "TotalCharges",

    "internetservice": "InternetService",
    "internet_service": "InternetService",
    "internet service": "InternetService",

    "paymentmethod": "PaymentMethod",
    "payment_method": "PaymentMethod",
    "payment method": "PaymentMethod",

    "paperlessbilling": "PaperlessBilling",
    "paperless_billing": "PaperlessBilling",
    "paperless billing": "PaperlessBilling",

    "contract": "Contract",

    "onlinesecurity": "OnlineSecurity",
    "online_security": "OnlineSecurity",
    "online security": "OnlineSecurity",

    "onlinebackup": "OnlineBackup",
    "online_backup": "OnlineBackup",
    "online backup": "OnlineBackup",

    "deviceprotection": "DeviceProtection",
    "device_protection": "DeviceProtection",
    "device protection": "DeviceProtection",

    "techsupport": "TechSupport",
    "tech_support": "TechSupport",
    "tech support": "TechSupport",

    "partner": "Partner",
    "dependents": "Dependents"
}

# ---------------- HELPER FUNCTIONS ----------------
def normalize_columns(df):
    new_cols = []
    for col in df.columns:
        clean_col = col.strip().lower().replace(" ", "").replace("_", "")
        mapped_col = column_mapping.get(clean_col, col)
        new_cols.append(mapped_col)
    df.columns = new_cols
    return df


def validate_columns(df):
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def prepare_bulk_dataframe(df):
    # normalize column names
    df = normalize_columns(df)

    # validate required columns
    missing = validate_columns(df)
    if missing:
        return None, missing

    # keep only required columns
    df = df[required_cols].copy()

    # convert numeric columns
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df, None


# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')


# ---------- SINGLE CUSTOMER PREDICTION ----------
@app.route('/single_predict', methods=['GET', 'POST'])
def single_predict():
    if request.method == 'POST':
        try:
            form_data = {
                "tenure": request.form.get("tenure", ""),
                "MonthlyCharges": request.form.get("MonthlyCharges", ""),
                "TotalCharges": request.form.get("TotalCharges", ""),
                "InternetService": request.form.get("InternetService", ""),
                "PaymentMethod": request.form.get("PaymentMethod", ""),
                "PaperlessBilling": request.form.get("PaperlessBilling", ""),
                "Contract": request.form.get("Contract", ""),
                "OnlineSecurity": request.form.get("OnlineSecurity", ""),
                "OnlineBackup": request.form.get("OnlineBackup", ""),
                "DeviceProtection": request.form.get("DeviceProtection", ""),
                "TechSupport": request.form.get("TechSupport", ""),
                "Partner": request.form.get("Partner", ""),
                "Dependents": request.form.get("Dependents", "")
            }

            input_df = pd.DataFrame([form_data])

            # Convert numeric columns
            input_df["tenure"] = pd.to_numeric(input_df["tenure"], errors="coerce")
            input_df["MonthlyCharges"] = pd.to_numeric(input_df["MonthlyCharges"], errors="coerce")
            input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

            transformed_input = pipeline.transform(input_df)

            prediction = model.predict(transformed_input)[0]

            classes = list(model.classes_)
            yes_index = classes.index("Yes")
            probability = model.predict_proba(transformed_input)[0][yes_index]

            return render_template(
                'single_predict.html',
                form_data=form_data,
                prediction_text=f"Predicted Churn: {prediction}",
                probability_text=f"Churn Probability: {probability:.2%}",
                error=None
            )

        except Exception as e:
            return render_template(
                'single_predict.html',
                form_data=request.form,
                prediction_text=None,
                probability_text=None,
                error=str(e)
            )

    # GET request
    return render_template(
        'single_predict.html',
        form_data={},
        prediction_text=None,
        probability_text=None,
        error=None
    )


# ---------- BULK FILE PREDICTION ----------
@app.route('/bulk')
def bulk():
    return render_template(
        'bulk_upload.html',
        success=None,
        error=None,
        download_ready=False
    )


@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    try:
        if 'file' not in request.files:
            return render_template(
                'bulk_upload.html',
                error="No file uploaded.",
                success=None,
                download_ready=False
            )

        file = request.files['file']

        if file.filename == '':
            return render_template(
                'bulk_upload.html',
                error="Please choose a file.",
                success=None,
                download_ready=False
            )

        filename = file.filename.lower()

        # ---------------- READ CSV / EXCEL SAFELY ----------------
        if filename.endswith('.csv'):
            df = pd.read_csv(file)

            # Case 1: tab-separated read as one column
            if len(df.columns) == 1:
                file.seek(0)
                df = pd.read_csv(file, sep='\t')

            # Case 2: quoted one-column CSV
            if len(df.columns) == 1:
                single_col_header = df.columns[0]
                new_cols = [col.strip().strip('"') for col in single_col_header.split(',')]
                split_rows = df.iloc[:, 0].astype(str).str.strip('"').str.split(',', expand=True)

                if split_rows.shape[1] == len(new_cols):
                    split_rows.columns = new_cols
                    df = split_rows

        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)

        else:
            return render_template(
                'bulk_upload.html',
                error="Unsupported file type. Please upload CSV or Excel.",
                success=None,
                download_ready=False
            )

        # Keep original uploaded file for output
        original_df = df.copy()

        # Prepare and validate dataframe
        prepared_df, missing = prepare_bulk_dataframe(df)

        if missing:
            return render_template(
                'bulk_upload.html',
                error=f"Missing required columns: {', '.join(missing)}",
                success=None,
                download_ready=False
            )

        # Transform using saved pipeline
        transformed_input = pipeline.transform(prepared_df)

        # Predict churn
        predictions = model.predict(transformed_input)

        # Predict churn probability for "Yes"
        classes = list(model.classes_)
        yes_index = classes.index("Yes")
        probabilities = model.predict_proba(transformed_input)[:, yes_index]

        # Add predictions to original dataframe
        original_df["Predicted_Churn"] = predictions
        original_df["Churn_Probability"] = probabilities

        # Save output
        original_df.to_csv(OUTPUT_FILE, index=False)

        return render_template(
            'bulk_upload.html',
            success="Prediction completed successfully!",
            error=None,
            download_ready=True
        )

    except Exception as e:
        return render_template(
            'bulk_upload.html',
            error=str(e),
            success=None,
            download_ready=False
        )


@app.route('/download')
def download_file():
    if os.path.exists(OUTPUT_FILE):
        return send_file(OUTPUT_FILE, as_attachment=True)
    return "No prediction file found."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)