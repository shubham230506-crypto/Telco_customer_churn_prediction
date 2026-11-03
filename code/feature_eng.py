import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL

# Load environment variables
load_dotenv()

# PostgreSQL connection details
connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME")
)

# Create database engine
engine = create_engine(connection_url)

try:
    # Test connection
    with engine.connect() as connection:
        print("PostgreSQL connection successful!")

    # Load your table into Pandas
    df = pd.read_sql(
        "SELECT * FROM public.telco_customer",
        engine
    )

    print("\nDataset loaded successfully!")
    print("Rows and columns:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

except Exception as error:
    print("Error:", error)

finally:
    engine.dispose()

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nDataset shape:")
print(df.shape)

print("\nDataset info:")
print(df.info())

print("\nDataset description:")
print(df.describe())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nChurn distribution:")
print(df['churn_value'].value_counts(dropna=False))

print("\nChurn percentages:")
print(df['churn_value'].value_counts(normalize=True, dropna=False)*100)

import matplotlib.pyplot as plt
import seaborn as sns




missing_rows = df[df['total_charges'].isna()]
print(missing_rows[
    ['customerid','tenure_months','monthly_charges','total_charges','churn_value']
    ])

data = df.copy()

data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')

print("\nMissing values after conversion:")
print(data['total_charges'].isnull().sum())

print(
    data.loc[
        data["total_charges"].isna(),
        ["tenure_months", "monthly_charges", "churn_value"]
    ]
)

# Fill missing total charges with zero only for zero-tenure customers.
# Stop if missing values exist for other customers.
missing = data["total_charges"].isna()
invalid_missing = missing & (data["tenure_months"] != 0)

if invalid_missing.any():
    raise ValueError(
        "Missing total charges found for customers with non-zero "
        "tenure. Investigate these rows before imputing."
    )

data.loc[missing, "total_charges"] = 0

# Create a useful tenure feature
data["tenure_group"] = pd.cut(
    data["tenure_months"],
    bins=[-1, 12, 24, 48, 60, np.inf],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "60+ months"
    ]
)

# Check the results
print("\nTenure group distribution:")
print(data["tenure_group"].value_counts(sort=False))

print("\nRemaining missing values:")
print(data[["total_charges", "tenure_group"]].isna().sum())




from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Work with a copy of the original dataset
data = df.copy()

# Convert total_charges to numeric
data["total_charges"] = pd.to_numeric(
    data["total_charges"], errors="coerce"
)

# Missing charges are expected for zero-tenure customers
missing = data["total_charges"].isna()

if (missing & (data["tenure_months"] != 0)).any():
    raise ValueError(
        "Investigate missing charges for non-zero-tenure customers."
    )

data.loc[missing, "total_charges"] = 0

# Create a new feature
data["tenure_group"] = pd.cut(
    data["tenure_months"],
    bins=[-1, 12, 24, 48, 60, np.inf],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "60+ months"
    ]
)

# Target variable: 0 = stayed, 1 = churned
y = data["churn_value"].astype(int)

# Remove identifiers, target duplicates, and leakage risks
columns_to_drop = [
    "customerid",
    "count",
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "latitude",
    "longitude",
    "churn_label",
    "churn_value",
    "churn_score",
    "churn_reason",
    "cltv"
]

X = data.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessing
X_test_processed = preprocessor.transform(X_test)

# Display results
print("\n--- FINAL DATA PREPARATION ---")
print("Original features:", X.shape[1])
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting churn distribution:")
print(y_test.value_counts(normalize=True).sort_index())

print("\nFeature preparation completed successfully!")



from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import pandas as pd


# 1. Train a simple baseline model

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_processed, y_train)

baseline_predictions = baseline.predict(X_test_processed)

print("\n========== BASELINE MODEL ==========")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Recall:", round(
    recall_score(y_test, baseline_predictions, zero_division=0), 4
))


# 2. Train LightGBM

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

model.fit(X_train_processed, y_train)


# 3. Predict churn

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]


# 4. Evaluate model

print("\n========== LIGHTGBM RESULTS ==========")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Precision:",
      round(precision_score(y_test, y_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, y_pred, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, y_pred, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 5. Show top 15 important features

feature_names = preprocessor.get_feature_names_out()

importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 15 Important Features:")
print(importance.head(15))

# 6. Compare with baseline

print("\n========== BASELINE COMPARISON ==========")
print("Baseline accuracy:",
      round(accuracy_score(y_test, baseline_predictions), 4))
print("LightGBM accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("\nModel training and evaluation completed!")



import shap
import matplotlib.pyplot as plt

# Explain a sample of test customers
X_explain = X_test_processed[:200]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

# Binary LightGBM models can return either a list or an array,
# depending on the SHAP version.
if isinstance(shap_values, list):
    churn_shap_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    churn_shap_values = shap_values[:, :, 1]
else:
    churn_shap_values = shap_values

feature_names = preprocessor.get_feature_names_out()

# Global explanation: which features influence predictions most?
shap.summary_plot(
    churn_shap_values,
    X_explain,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary: Customer Churn")
plt.tight_layout()
plt.show()



# STEP 1: COMPARE MACHINE LEARNING MODELS


import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": model
}

results = []
trained_models = {}

# Train and evaluate each model
for name, current_model in models.items():

    print(f"\nTraining {name}...")

    current_model.fit(X_train_processed, y_train)

    predictions = current_model.predict(X_test_processed)
    probabilities = current_model.predict_proba(X_test_processed)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    trained_models[name] = current_model

# Display comparison
results_df = pd.DataFrame(results).set_index("Model")

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).sort_values("F1 Score", ascending=False))

# Select the model with the highest F1 score
best_model_name = results_df["F1 Score"].idxmax()
best_model = trained_models[best_model_name]

print(f"\nBest model by F1 Score: {best_model_name}")

# Save the model and preprocessing together
joblib.dump(
    {
        "model": best_model,
        "preprocessor": preprocessor,
        "model_name": best_model_name,
        "threshold": 0.50
    },
    "churn_model.joblib"
)

print("\nSaved model and preprocessing to churn_model.joblib")
