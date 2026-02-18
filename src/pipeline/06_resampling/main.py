import sys
from pathlib import Path
import mlflow
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.data_processing.data_resampling import DataResampling
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE
from src.mlflow_parent import get_or_create_parent_run

STAGE_NAME = "06 - Resampling stage"


class DataResamplingPipeline:

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting resampling.")

        cm = ConfigurationManager()
        cfg = cm.get_data_resampling_config()

        resampler = DataResampling(cfg)

        start = time.time()
        resampler.resample_data()
        duration = time.time() - start

        return duration

    def main(self):

        # 🔵 Reconnexion au parent run
        parent_run_id = get_or_create_parent_run()
        mlflow.start_run(run_id=parent_run_id)
        try:
            # 🟢 nested run for this stage
            with mlflow.start_run(run_name="06_resampling", nested=True):

                mlflow.log_param("step", "06_resampling")

                duration = self.run()

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
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        DataResamplingPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.exception("Resampling failed")
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
