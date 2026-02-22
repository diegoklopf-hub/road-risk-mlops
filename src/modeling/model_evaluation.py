import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.entity import ModelEvaluationConfig
from src.common_utils import save_json

from prometheus_client import CollectorRegistry, push_to_gateway, Counter, Histogram, generate_latest, Gauge
import time

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        eval_start = time.time()

        registry = CollectorRegistry()

        # Prometheus Metrics
        ml_r2 = Gauge(
            "ml_model_r2_score",
            "Coefficient de détermination R² du modèle ML",
            registry=registry
        )
        ml_rmse = Gauge(
            "ml_model_rmse",
            "Root Mean Squared Error du modèle ML",
            registry=registry
        )
        ml_mae = Gauge(
            "ml_model_mae",
            "Mean Absolute Error du modèle ML",
            registry=registry
        )
        ml_test_set_size = Gauge(
            "ml_model_test_set_size",
            "Nombre d'échantillons dans le jeu de test",
            registry=registry
        )
        ml_evaluation_duration = Gauge(
            "ml_model_evaluation_duration_seconds",
            "Durée de l'évaluation complète",
            registry=registry
        )
        ml_status = Gauge(
            "ml_model_evaluation_success",
            "1 si l'évaluation a réussi, 0 sinon",
            registry=registry
        )

        
        try:
            X_test = pd.read_csv(self.config.X_test_path)
            y_test = pd.read_csv(self.config.y_test_path).iloc[:, -1].astype(float)
            model = joblib.load(self.config.model_path)

            mlflow_uri = getattr(self.config, "mlflow_uri", None)
            if mlflow_uri:
                mlflow.set_registry_uri(mlflow_uri)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(nested=True):
                predicted_qualities = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                # Update Prometheus gauges
                ml_r2.set(r2)
                ml_rmse.set(rmse)
                ml_mae.set(mae)
                ml_test_set_size.set(len(y_test))
                ml_status.set(1)

                # Saving metrics as local
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)

                mlflow.log_params(getattr(self.config, "all_params", {}))

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
               
        except Exception as e:
            ml_status.set(0)
            logger.error(f"Evaluation error: {e}")
            raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")
        
        finally:
            ml_evaluation_duration.set(time.time() - eval_start)  
                        # Push metrics to Pushgateway
            try:
                push_to_gateway(
                    gateway=self.config.pushgateway_url,  # "localhost:9091"
                    job="model_evaluation",
                    grouping_key={"pipeline": "accidents"},  # label used for Grafana filtering
                    registry=registry
                )
            except Exception as push_err:
                # Do not fail the pipeline if Prometheus is down
                print(f"[WARNING] Pushgateway push failed: {push_err}")     
