import sys
from pathlib import Path

# --- Make "import src..." work even when running as a file (Mac/Windows/Linux)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.common_utils import append_status, is_last_status_ok
from src.data_processing.data_clean import DataClean
from src.config import STATUS_FILE

STAGE_NAME = "02 - Data Clean stage"


class DataCleanPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        if not is_last_status_ok(STATUS_FILE):
            raise RuntimeError("Previous stage status is not OK. Aborting data clean.")
        data_clean_config = self.config.get_data_clean_config()
        dataclean = DataClean(config=data_clean_config)
        dataclean.clean_data()
        dataclean.check_cleaned_files()

    def main(self):
        self.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataCleanPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
