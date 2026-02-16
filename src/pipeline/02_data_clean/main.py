import sys
from pathlib import Path
import mlflow
import os
import time

# MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
mlflow.set_experiment("GLOBAL_PIPELINE")   # 🔥 CRUCIAL
parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.common_utils import is_last_status_ok
from src.data_processing.data_clean import DataClean
from src.config import STATUS_FILE

STAGE_NAME = "02 - Data Clean stage"


class DataCleanPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting data clean.")

        cfg = self.config.get_data_clean_config()
        dataclean = DataClean(config=cfg)
        dataclean.clean_data()
        dataclean.check_cleaned_files()

    def main(self):

        # reconnect parent run ou debug
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
            with mlflow.start_run(run_name="02_data_clean", nested=True):

                mlflow.log_param("step", "02_data_clean")

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
            mlflow.end_run()


if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        DataCleanPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception:
        logger.exception("Data clean failed")
        raise
