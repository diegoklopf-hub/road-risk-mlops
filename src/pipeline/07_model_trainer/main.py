import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
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
from src.models.model_trainer import ModelTrainer
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE

STAGE_NAME = "07 - Model Trainer stage"


class ModelTrainerPipeline:

    def run(self):

        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting model training.")

        cm = ConfigurationManager()
        cfg = cm.get_model_trainer_config()

        trainer = ModelTrainer(cfg)

        start = time.time()

        best_model, best_params, features_list = trainer.train()
        trainer.export_model(best_model, best_params, features_list)

        duration = time.time() - start

        return best_model, best_params, features_list, duration


    def main(self):

        # 🔵 reconnect parent pipeline OU debug local
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
            # 🟢 nested run du stage 07
            with mlflow.start_run(run_name="07_model_training", nested=True):

                mlflow.log_param("step", "07_model_training")

                model, params, features, duration = self.run()

                mlflow.log_params(params)
                mlflow.log_metric("training_duration_sec", duration)
                mlflow.log_metric("n_features", len(features))

                # 🔥 log model dans ce nested run
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name="saver_accident_model"
                )

                mlflow.set_tag("stage", "training")
                mlflow.log_param("status", "completed")

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        finally:
            # 🔴 CRUCIAL sinon UI vide
            mlflow.end_run()


if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        ModelTrainerPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.exception("Training failed")
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
