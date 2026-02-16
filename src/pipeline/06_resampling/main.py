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

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.data_processing.data_resampling import DataResampling
from src.common_utils import is_last_status_ok
from src.config import STATUS_FILE

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

        # 🔵 reconnect parent pipeline OU debug local
        if parent_run_id:
            mlflow.start_run(run_id=parent_run_id)
        else:
            mlflow.start_run(run_name="debug_parent")

        try:
            # 🟢 nested run du stage
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
            # 🔴 indispensable sinon UI vide
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
