import sys
from pathlib import Path
import mlflow
import os
import time

# MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
mlflow.set_experiment("GLOBAL_PIPELINE")
parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")

# Fix import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_logger import logger
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE
from src.config_manager import ConfigurationManager
from src.data_processing.data_import import DataImport

STAGE_NAME = "01 - Data Import stage"


class DataImportPipeline:

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting data import.")

        config = ConfigurationManager()
        data_import_config = config.get_data_import_config()

        data_import = DataImport(config=data_import_config)
        data_import.import_csv()
        data_import.check_imported_files()

    def main(self):

    # -------------------------
    # reconnect au parent run
    # -------------------------
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
        # -------------------------
        # nested run réel
        # -------------------------
            with mlflow.start_run(run_name="01_data_import", nested=True):

                mlflow.log_param("step", "01_data_import")

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
        # CRUCIAL : fermer le parent reconnecté
            mlflow.end_run()


if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataImportPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
