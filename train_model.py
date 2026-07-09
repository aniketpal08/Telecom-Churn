import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    final_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return final_pipeline


# ---------------- LOAD DATA ----------------
df = pd.read_csv("Telco Customer Churn.csv")

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# features used
num_attribs = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_attribs = [
    'InternetService', 'PaymentMethod', 'PaperlessBilling', 'Contract',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'Partner', 'Dependents'
]

# keep only important columns
X = X[num_attribs + cat_attribs]

# stratified split on target
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(X, y):
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

# build pipeline
pipeline = build_pipeline(num_attribs, cat_attribs)
X_train_prepared = pipeline.fit_transform(X_train)

# model
model = LogisticRegression(
    C=100,
    class_weight='balanced',
    penalty='l2',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

model.fit(X_train_prepared, y_train)

# save
joblib.dump(model, MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)

print("Model and pipeline saved successfully!")
