# monitoring.py
import os
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import pandas as pd
from io import StringIO

def generate_drift_report():

    path_current = 'src/data/current.csv'
    path_reference = 'src/data/reference.csv'
    
    if not os.path.exists(path_current) or not os.path.exists(path_reference):
        return None

    current = pd.read_csv(path_current)
    reference = pd.read_csv(path_reference)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    html_io = StringIO()
    report.save_html(html_io)
    html_io.seek(0)

    return html_io.read()
