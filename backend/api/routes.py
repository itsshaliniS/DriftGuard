from fastapi import APIRouter, File, UploadFile, HTTPException
import subprocess
import sys
import os
import json

router = APIRouter()

# run in separate process to avoid blocking event loop
def run_isolated(cmd, payload):
    script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "isolated_runner.py"))
    proc = subprocess.Popen(
        [sys.executable, script, cmd], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True, 
        encoding='utf-8'
    )
    
    stdout, stderr = proc.communicate(input=payload)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {stderr}")
        
    # extract json output ignoring any warnings
    json_start = stdout.find('{')
    if json_start != -1:
        return json.loads(stdout[json_start:])
    raise ValueError("No JSON output returned")

@router.post("/train")
async def train_model_endpoint(baseline_csv: UploadFile = File(...)):
    content = await baseline_csv.read()
    baseline_str = content.decode("utf-8", errors="ignore")
    metrics = run_isolated("train", baseline_str)
    return {"success": True, "message": "Model trained successfully.", "data": metrics}

@router.post("/model/performance")
async def model_performance_endpoint(prod_csv: UploadFile = File(...)):
    content = await prod_csv.read()
    prod_str = content.decode("utf-8", errors="ignore")
    perf = run_isolated("perf", prod_str)
    if "error" in perf:
        raise HTTPException(status_code=400, detail=perf["error"])
    return {"success": True, "data": perf}

@router.post("/drift/scan")
async def scan_drift(baseline_csv: UploadFile = File(...), prod_csv: UploadFile = File(...)):
    base_content = await baseline_csv.read()
    prod_content = await prod_csv.read()
    
    payload = json.dumps({
        "baseline": base_content.decode("utf-8", errors="ignore"),
        "prod": prod_content.decode("utf-8", errors="ignore")
    })
    
    report = run_isolated("drift", payload)
    return {"success": True, "message": "Drift analysis complete.", "data": report}

@router.post("/predict")
async def predict_endpoint(payload: dict):
    res = run_isolated("predict", json.dumps(payload))
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("detail"))
    return res

