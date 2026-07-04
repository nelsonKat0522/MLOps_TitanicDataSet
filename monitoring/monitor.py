import os
import pandas as pd
import joblib

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

from sklearn.model_selection import train_test_split

DATA_PATH = "data/titanic.csv"
MODEL_PATH = "models/model.pkl"
REPORT_PATH = "reports/evidently_report.html"

os.makedirs("reports", exist_ok=True)

data = pd.read_csv(DATA_PATH)

data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_reference, X_current, y_reference, y_current = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = joblib.load(MODEL_PATH)

reference_data = X_reference.copy()
current_data = X_current.copy()

reference_data["target"] = y_reference
current_data["target"] = y_current

reference_data["prediction"] = model.predict(X_reference)
current_data["prediction"] = model.predict(X_current)

report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html(REPORT_PATH)

print(f"Evidently report saved to {REPORT_PATH}")
