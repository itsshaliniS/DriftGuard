import sys
import json
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def run():
    cmd = sys.argv[1]
    raw = sys.stdin.read()
    
    if cmd == "train":
        from backend.services.report_service import train_churn_model
        res = train_churn_model(raw)
        print(json.dumps(res))
        
    elif cmd == "drift":
        payload = json.loads(raw)
        from backend.services.report_service import generate_drift_report
        res = generate_drift_report(payload["baseline"], payload["prod"])
        print(json.dumps(res))
        
    elif cmd == "perf":
        from backend.services.report_service import evaluate_production_model
        res = evaluate_production_model(raw)
        print(json.dumps(res))
        
    elif cmd == "predict":
        from backend.services.report_service import execute_inference
        res = execute_inference(raw)
        print(json.dumps(res))

if __name__ == "__main__":
    run()

