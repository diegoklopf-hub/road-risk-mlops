import sys
from pathlib import Path
import mlflow
import os
import time

# Fix import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_merge import DataMerge
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE
from src.mlflow_parent import get_or_create_parent_run

STAGE_NAME = "03 - Merge stage"


class DataMergePipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting merge.")

        cfg = self.config.get_data_merge_config()
        dm = DataMerge(cfg)

        dm.merge_by_usager()
        dm.feature_engineering()
        dm.merge_by_accident()
        dm.validate_data_and_export()

    def main(self):

        # 🔵 Reconnexion au parent run
        parent_run_id = get_or_create_parent_run()
        mlflow.start_run(run_id=parent_run_id)

        try:
            # 🟢 nested run for THIS step
            with mlflow.start_run(run_name="03_merge", nested=True):

                mlflow.log_param("step", "03_merge")

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
            # 🔴 REQUIRED, otherwise nothing appears in MLflow UI
            mlflow.end_run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataMergePipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
