import pytest
import pandas as pd
import numpy as np
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core.drift_analyzer import DriftAnalyzer

@pytest.fixture
def analyzer():
    return DriftAnalyzer(p_value_threshold=0.05)

def create_synthetic_data(num_samples=1000, drift=False):
    np.random.seed(42)
    data = {
        "customerID": [f"CUST_{i}" for i in range(num_samples)],
        "tenure": np.random.randint(1, 72, num_samples),
        "MonthlyCharges": np.random.normal(50, 15, num_samples),
        "TotalCharges": np.random.normal(1000, 500, num_samples),
        "Churn": np.random.choice(["Yes", "No"], num_samples, p=[0.2, 0.8])
    }
    
    if drift:
        data["MonthlyCharges"] = np.random.normal(100, 30, num_samples)
        data["TotalCharges"] = np.random.normal(3000, 1000, num_samples)
    
    return pd.DataFrame(data)

def df_to_csv_string(df):
    sio = io.StringIO()
    df.to_csv(sio, index=False)
    return sio.getvalue()

def test_no_drift_detected(analyzer):
    df_base = create_synthetic_data(drift=False)
    df_prod = create_synthetic_data(drift=False)
    
    report = analyzer.detect_drift(df_to_csv_string(df_base), df_to_csv_string(df_prod))
    assert report["summary"]["drifted_features"] == 0
    assert report["summary"]["dataset_health_score"] >= 90.0

def test_ks_and_psi_drift_detection(analyzer):
    df_base = create_synthetic_data(drift=False)
    df_prod = create_synthetic_data(drift=True)
    
    report = analyzer.detect_drift(df_to_csv_string(df_base), df_to_csv_string(df_prod))
    assert report["summary"]["drifted_features"] >= 2
    assert report["summary"]["dataset_health_score"] < 70.0
    
    details_monthly = next((f for f in report["feature_details"] if f["feature"] == "MonthlyCharges"), None)
    assert details_monthly is not None
    assert details_monthly["status"] == "Drift Detected"
    assert details_monthly["psi_severity"] == "Significant"

def test_empty_string_handling(analyzer):
    empty_csv = "feature1,feature2\n"
    report = analyzer.detect_drift(empty_csv, empty_csv)
    assert report["summary"]["total_features"] == 0
    assert report["summary"]["dataset_health_score"] == 100.0

