# monitoring.py
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import pandas as pd
from io import StringIO

def generate_drift_report():
    reference = pd.read_csv('src/data/reference.csv')
    current = pd.read_csv('src/data/current.csv')

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    html_io = StringIO()
    report.save_html(html_io)
    html_io.seek(0)

    return html_io.read()
