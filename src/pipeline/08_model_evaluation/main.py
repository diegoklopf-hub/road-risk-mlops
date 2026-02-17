import sys
from pathlib import Path
import mlflow
import os
import json
import time

# MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
mlflow.set_experiment("GLOBAL_PIPELINE")
parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.modeling.model_evaluation import ModelEvaluation
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE

STAGE_NAME = "08 - Model Evaluation stage"


class ModelEvaluationPipeline:

    def run(self):

        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting model evaluation.")

        cm = ConfigurationManager()
        cfg = cm.get_model_evaluation_config()

        evaluator = ModelEvaluation(cfg)

        start = time.time()

        # 🔵 compute + internal logging (then re-log in nested run)
        evaluator.log_into_mlflow()

        duration = time.time() - start

        metric_file = getattr(cfg, "metric_file_name", None)
        metrics_dict = {}

        if metric_file:
            p = Path(metric_file)
            if p.exists():
                metrics_dict = json.loads(p.read_text())
                logger.info(f"Metrics loaded: {metrics_dict}")

        return metrics_dict, duration


    def main(self):

        # 🔵 reconnect parent pipeline or local debug run
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
            # 🟢 nested run visible in UI
            with mlflow.start_run(run_name="08_model_evaluation", nested=True):

                mlflow.log_param("step", "08_model_evaluation")

                metrics, duration = self.run()

                # log metrics in THIS nested run
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

                mlflow.log_metric("evaluation_duration_sec", duration)

                # quality tags
                if "rmse" in metrics and metrics["rmse"] < 25:
                    mlflow.set_tag("model_quality", "good")
                    mlflow.set_tag("candidate_for_production", "true")
                else:
                    mlflow.set_tag("candidate_for_production", "false")

                mlflow.set_tag("stage", "evaluation")
                mlflow.log_param("status", "completed")

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        finally:
            # 🔴 CRUCIAL, otherwise nothing appears in UI
            mlflow.end_run()


if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        ModelEvaluationPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.exception("Evaluation failed")
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
