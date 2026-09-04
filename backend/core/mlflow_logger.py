import mlflow

class MLflowLogger:
    """
    Object-oriented wrapper handling initialization and metric tracking 
    for the enterprise tracking server logic (MLflow).
    """
    def __init__(self, experiment_name: str = "Enterprise_Drift_Monitoring"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_drift_metrics(self, summary: dict) -> str:
        """
        Logs a drift summary payload and returns the active tracking run id.
        """
        with mlflow.start_run() as run:
            # Metrics
            mlflow.log_metric("drifted_features", summary["drifted_features"])
            mlflow.log_metric("total_features_checked", summary["total_features"])
            mlflow.log_metric("dataset_health_score", summary["dataset_health_score"])
            
            # Param
            mlflow.log_param("p_value_threshold", summary["threshold"])
            
            return run.info.run_id
