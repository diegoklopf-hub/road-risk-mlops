import sys
from pathlib import Path
import mlflow
import os
import time

# MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
mlflow.set_experiment("GLOBAL_PIPELINE")
parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_encoding import DataEncodage
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE

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
        data_encodage.encode_categorical_values()
        data_encodage.encode_continue_score_grav()
        data_encodage.validate_data_and_export()

    def main(self):

        # 🔵 reconnect parent pipeline OU debug local
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
            # 🟢 nested run pour CE step
            with mlflow.start_run(run_name="04_encodage", nested=True):

                mlflow.log_param("step", "04_encodage")

                start = time.time()
                self.run()
                duration = time.time() - start

                mlflow.log_metric("duration_sec", duration)
                mlflow.log_param("status", "completed")

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        finally:
            # 🔴 CRUCIAL sinon MLflow UI reste vide
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
