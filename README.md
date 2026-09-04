# Drift Guard — MLOps Data Drift Detection Platform

## Why I Built This

While looking into what actually happens to ML models after deployment, I kept running into the same problem: training a model is maybe 20% of the job. The real headache is that models decay quietly in production once real-world data drifts away from what they were trained on — and nobody notices until performance has already tanked.

I built Drift Guard to catch that before it becomes a silent failure. The goal wasn't another notebook — it's a small end-to-end pipeline that watches incoming data streams, measures statistical shift, and flags it before it turns into a real problem.

## System Architecture

```
  [Baseline CSV]      [Production Batch]
         |                    |
         v                    v
+------------------------------------------+
|          HTML / CSS / JS UI              |
|        (Chart.js Dashboards)             |
+-------------------+----------------------+
                    | (REST HTTP via Port 8000)
                    v
+------------------------------------------+
|          FastAPI Backend Server          |
|                                          |
|  +----------------+  +----------------+  |
|  | Model Trainer  |  | Drift Analyzer |  |
|  | (RandomForest) |  | (KS/PSI/JS)    |  |
|  +----------------+  +----------------+  |
|          |                  |            |
+----------|------------------|------------+
           v                  v
    [joblib Artifacts]   [JSON Payloads]
```

## How It Works

The analytics engine compares incoming distributions against the baseline to score how much things have degraded. It uses the Kolmogorov-Smirnov (KS) test to check if the shape of the incoming data has drifted from the baseline. Then it calculates the Population Stability Index (PSI) by splitting the distributions into 10 bins to see how much the underlying population has shifted over time. Finally, Jensen-Shannon divergence gives a bounded 0-to-1 score, which avoids some of the scaling issues you run into with something like Wasserstein distance.

## Experiments & Synthetic Drift Tracking

To test this properly, I built a small simulation script (`data/drift_simulator.py`) that deliberately corrupts clean data columns, so I could confirm the pipeline actually catches it and the UI reflects it correctly.

| Drift Type          | Feature Mutated | Target Metric Triggered | Health Score Penalty |
| ------------------- | --------------- | ------------------------ | --------------------- |
| **Covariate Shift**  | `TotalCharges`  | KS Statistic: `~0.65`    | 📉 `81.25%`            |
| **Label Shift**      | `Churn`         | PSI Score: `~0.31`       | 📉 `64.10%`            |
| **Gradual Shift**    | `tenure`        | JS Divergence: `~0.12`   | 📉 `55.40%`            |

## Known Issues

A couple of rough edges still in this build:

MLflow initialization crashes outright on some Windows Python setups (`MINGW-W64`) unless it's run in an isolated terminal — ended up needing an offline string-logging fallback just to keep things stable locally.

If Docker Compose tries to bind port `8000` while an old, hung `Uvicorn` process is still lingering in the background, the container fails to bind. Haven't gotten around to a clean fix for that yet.

## What I Learned

Honestly, this one was a grind but worth it. I burned close to three hours on a Random Forest evaluation crash that turned out to be Pandas' `get_dummies` silently misaligning columns between the baseline and the new batch — fixed it by saving the feature list to `joblib` and forcing `.reindex(fill_value=0)` on every incoming batch.

Also fought with PSI returning `Infinity` whenever a bin ended up with zero counts — fixed that by adding a tiny epsilon (`+ 1e-4`) to the percentage arrays before the log calculation. And getting CORS working cleanly between Nginx and FastAPI across async requests forced me to actually understand `async`/`await` instead of just copy-pasting it.