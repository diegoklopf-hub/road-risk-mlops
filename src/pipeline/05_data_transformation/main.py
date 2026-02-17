import sys
from pathlib import Path
import mlflow
import os
import time

# Fix imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.data_processing.data_transformation import DataTransformation
from src.custom_logger import logger
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE
from src.mlflow_parent import get_or_create_parent_run

STAGE_NAME = "05 - Data Transformation stage"


class DataTransformationTrainingPipeline:

    def run(self):

        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting data transformation.")

        cm = ConfigurationManager()
        cfg = cm.get_data_transformation_config()

        data_transformation = DataTransformation(config=cfg)

        X_train, X_test, _, _ = data_transformation.train_test_splitting()
        X_train, X_test = data_transformation.normalize(X_train, X_test)
        data_transformation.features_selection(X_train, X_test)
        data_transformation.check_transformation_outputs()

    def main(self):

        # 🔵 Reconnexion au parent run
        parent_run_id = get_or_create_parent_run()
        mlflow.start_run(run_id=parent_run_id)

        try:
            # 🟢 nested run step
            with mlflow.start_run(run_name="05_data_transformation", nested=True):

                mlflow.log_param("step", "05_data_transformation")

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
            # 🔴 required, otherwise UI is empty
            mlflow.end_run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataTransformationTrainingPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
