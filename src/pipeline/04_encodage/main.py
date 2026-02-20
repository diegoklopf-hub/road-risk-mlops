import sys
from pathlib import Path
import mlflow
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_encoding import DataEncodage
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE
from src.mlflow_parent import get_or_create_parent_run

STAGE_NAME = "04 - Encodage stage"


class DataEncodagePipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting encodage.")

        cfg = self.config.get_data_encodage_config()
        data_encodage = DataEncodage(config=cfg)

        data_encodage.encode_cyclic_values()
        encoder = data_encodage.encode_categorical_values()
        data_encodage.encode_continue_score_grav()
        data_encodage.validate_data_and_export()
        return encoder

    def main(self):

        # 🔵 Reconnexion au parent run
        parent_run_id = get_or_create_parent_run()
        mlflow.start_run(run_id=parent_run_id)

        try:
            # 🟢 nested run for THIS step
            with mlflow.start_run(run_name="04_encodage", nested=True):

                mlflow.log_param("step", "04_encodage")

                start = time.time()
                encoder = self.run()
                duration = time.time() - start

                mlflow.log_metric("duration_sec", duration)
                mlflow.log_param("status", "completed")

                # 🔥 log model in this nested run
                mlflow.sklearn.log_model(
                    sk_model=encoder,
                    artifact_path="model_encoder",
                    registered_model_name="one_hot_encoder_accident_model"
                )

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        finally:
            # 🔴 CRUCIAL, otherwise MLflow UI stays empty
            mlflow.end_run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataEncodagePipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
