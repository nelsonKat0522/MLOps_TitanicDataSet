import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference_data = pd.read_csv("titanic.csv")
current_data = pd.read_csv("titanic.csv").sample(frac=0.5, random_state=42)

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html("reports/data_drift_report.html")
