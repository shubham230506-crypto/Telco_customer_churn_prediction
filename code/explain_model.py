import joblib

# Load the joblib file
data = joblib.load('churn_model.joblib')

# View the object type and its contents
print(type(data))
print(data)
