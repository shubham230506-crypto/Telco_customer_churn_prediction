# Customer Churn Prediction

A machine learning project that analyzes telecom customer data and
predicts whether a customer is likely to churn (leave the service). The
project uses PostgreSQL for data storage, Python for data preparation
and modeling, and Streamlit for an interactive prediction dashboard.

## Project Overview

Customer churn is an important business problem for telecommunications
companies. Identifying customers who may leave can help businesses
understand retention risks and plan appropriate customer-support or
retention strategies.

This project follows an end-to-end machine learning workflow:

1.  Retrieve customer data from PostgreSQL.
2.  Explore and clean the dataset.
3.  Engineer useful features.
4.  Preprocess numerical and categorical variables.
5.  Train and compare classification models.
6.  Evaluate model performance.
7.  Explain model behavior with SHAP.
8.  Save the trained model and preprocessing pipeline.
9.  Predict churn for new customer records.
10. Display predictions through a Streamlit dashboard.

## Dataset

The project uses a telecom customer churn dataset with **7,043 records
and 33 original columns**.

The target column is `churn_value`:

-   `0` --- Customer stayed.
-   `1` --- Customer churned.

The target distribution in the dataset is approximately:

  Customer status       Records   Percentage
  ----------------- ----------- ------------
  Stayed                  5,174       73.46%
  Churned                 1,869       26.54%
  **Total**           **7,043**     **100%**

### Data preparation

-   Checked data types, missing values, and duplicate records.
-   Found 11 missing `total_charges` values. These belonged to customers
    with zero tenure and were filled with `0`.
-   Excluded `churn_reason`, because it describes churn outcomes and is
    not available for customers who have not churned.
-   Excluded identifiers, location fields, duplicate target
    representations, and potential leakage fields from the model inputs.
-   Created a `tenure_group` feature using customer tenure ranges.

The `tenure_group` categories are:

-   `0-12 months`
-   `13-24 months`
-   `25-48 months`
-   `49-60 months`
-   `60+ months`

## Technology Stack

  Technology     Purpose
  -------------- -----------------------------------------------
  Python         Main programming language
  PostgreSQL     Relational database
  SQLAlchemy     Database connection and SQL access
  Pandas         Data manipulation and analysis
  NumPy          Numerical operations
  Scikit-learn   Preprocessing, model training, and evaluation
  LightGBM       Gradient-boosting classification
  SHAP           Model explainability
  Matplotlib     Data visualization
  Joblib         Saving and loading trained objects
  Streamlit      Interactive web application

## Machine Learning Workflow

### 1. Data loading

Python connects to PostgreSQL and loads the customer table into a Pandas
DataFrame.

### 2. Exploratory data analysis

The dataset is inspected for:

-   Number of rows and columns
-   Data types
-   Missing values
-   Duplicate records
-   Numerical summaries
-   Churn distribution

### 3. Feature selection and engineering

The target is separated from the input features:

-   `X` contains the selected customer features.
-   `y` contains `churn_value`.

A `tenure_group` feature is created from `tenure_months`.

Identifiers, location information, target duplicates, and fields that
could cause data leakage are excluded.

### 4. Train-test split

The data is divided into training and testing sets using:

-   Test size: 20%
-   Random state: 42
-   Stratification: target variable

  Split        Records
  ---------- ---------
  Training       5,634
  Testing        1,409

### 5. Preprocessing

A Scikit-learn `ColumnTransformer` applies separate transformations:

**Numerical features** - Median imputation

**Categorical features** - Most-frequent imputation - One-hot encoding -
Unknown categories are ignored during transformation

The preprocessor is fitted on the training data and then applied to the
test data to avoid test-data leakage.

### 6. Model training and comparison

The models used in the project are:

-   Dummy Classifier --- baseline
-   Logistic Regression
-   Random Forest Classifier
-   LightGBM Classifier

Models are compared using accuracy, precision, recall, F1-score, and
ROC-AUC.

### 7. Model evaluation

The initial LightGBM evaluation produced the following results:

  Metric         Score
  ----------- --------
  Accuracy      79.21%
  Precision     62.02%
  Recall        55.88%
  F1-score      58.79%
  ROC-AUC       83.92%

Initial LightGBM confusion matrix:

  Actual / Predicted     Stayed   Churned
  -------------------- -------- ---------
  Stayed                    907       128
  Churned                   165       209

These scores are from the initial LightGBM run. The final selected model
is determined by the model-comparison code using F1-score at a 0.50
probability threshold. The saved model in `churn_model.joblib`
identifies the selected model.

### 8. Model explainability

SHAP is used to examine how input features influence model predictions.
The SHAP summary plot helps identify features with larger overall
influence and whether their values tend to push predictions toward or
away from churn.

SHAP explanations describe model behavior; they do not establish that a
feature causes customer churn.

### 9. Model persistence

The selected model and fitted preprocessor are saved together in:

`churn_model.joblib`

This allows the prediction script and dashboard to reuse the fitted
transformations and trained model without retraining on every run.

### 10. Customer prediction

`predict_churn.py` loads the saved objects, accepts customer details,
applies the same preprocessing, and calculates the predicted churn
probability.

The default classification threshold is `0.50`:

-   Probability at or above 0.50: likely to churn.
-   Probability below 0.50: likely to stay.

A probability is a model estimate, not a guarantee.

### 11. Streamlit dashboard

`app.py` provides an interface to enter customer details and display:

-   Predicted churn probability
-   Decision threshold
-   Churn prediction

## Project Structure

The main files used by the project are:

``` text
project1/
│
├── feature_eng.py          # Data preparation, training, evaluation, SHAP
├── predict_churn.py        # Prediction for an individual customer
├── app.py                  # Streamlit dashboard
├── churn_model.joblib      # Saved model and preprocessing pipeline
└── README.md               # Project documentation
```

Additional folders or files may be present for datasets, reports, plots,
and experiments.

## Installation

### Prerequisites

-   Python 3.10 or later
-   PostgreSQL
-   pip
-   A PostgreSQL database containing the telecom customer table

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd project1
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install pandas numpy sqlalchemy psycopg2-binary scikit-learn lightgbm shap matplotlib joblib streamlit
```

### 4. Configure PostgreSQL

Update the database connection settings in `feature_eng.py` with your
own PostgreSQL username, password, host, port, and database name.

Do not commit database passwords or other secrets to GitHub. Use
environment variables or a local configuration file excluded through
`.gitignore`.

Ensure the table `public.telco_customer` exists and contains the
expected dataset columns.

## Running the Project

### Train and evaluate models

``` bash
python feature_eng.py
```

This runs the data preparation and model training workflow and saves
`churn_model.joblib`.

### Run an individual prediction

``` bash
python predict_churn.py
```

### Launch the Streamlit dashboard

``` bash
python -m streamlit run app.py
```

Streamlit displays a local URL in the terminal. Open that URL in a
browser to use the dashboard.

## Important Considerations

-   The saved model and preprocessor must be generated from compatible
    versions of the project code and dependencies.
-   New customer input categories should match the categories and
    spelling used in the training dataset.
-   The current decision threshold is 0.50. Threshold tuning and
    probability calibration should be evaluated on validation data
    before operational use.
-   Model performance can change on new data. Monitor performance and
    retrain when appropriate.
-   Predictions should support, not replace, business judgment. Do not
    use churn predictions as the sole basis for decisions affecting
    customers.

## Future Improvements

-   Tune hyperparameters using cross-validation.
-   Tune the classification threshold based on business costs.
-   Evaluate probability calibration.
-   Add model comparison charts and confusion matrices to the dashboard.
-   Add customer-level explanations using SHAP.
-   Add business-focused churn and retention reports.
-   Add automated tests and input validation.
-   Deploy the dashboard to a suitable hosting platform.

## License

Add a license before distributing or reusing this project publicly.
