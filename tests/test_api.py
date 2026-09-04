import sys
import os
import pytest
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.api.main import app

@pytest.mark.anyio
async def test_missing_files_error():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post("/api/v1/drift/scan")
        assert res.status_code == 422

@pytest.mark.anyio
async def test_upload_train_workflow():
    dummy_csv = "customerID,tenure,MonthlyCharges,TotalCharges,Churn\nCUST_1,10,55.5,550.0,No\nCUST_2,20,65.0,1300.0,Yes\nCUST_3,5,20.0,100.0,No\nCUST_4,15,45.0,600.0,No\nCUST_5,25,75.0,1800.0,Yes\nCUST_6,8,30.0,240.0,No\n"
    files = {
        'baseline_csv': ('baseline.csv', dummy_csv, 'text/csv')
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post("/api/v1/train", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "accuracy" in data["data"]

@pytest.mark.anyio
async def test_full_analyze_upload_workflow():
    dummy_csv = "customerID,tenure,MonthlyCharges,TotalCharges,Churn\nCUST_1,10,55.5,550.0,No\nCUST_2,20,65.0,1300.0,Yes\nCUST_3,5,20.0,100.0,No\n"
    files = {
        'baseline_csv': ('baseline.csv', dummy_csv, 'text/csv'),
        'prod_csv': ('production.csv', dummy_csv, 'text/csv'),
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post("/api/v1/drift/scan", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "data" in data
        
        summary = data["data"]["summary"]
        assert summary["drifted_features"] == 0
        assert summary["total_features"] > 0
        assert summary["dataset_health_score"] == 100.0


