import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_encoding import DataEncodage

STAGE_NAME = "04 - Encodage stage"


class DataEncodagePipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        data_encodage_config = self.config.get_data_encodage_config()
        data_encodage = DataEncodage(config=data_encodage_config)

        data_encodage.encode_cyclic_values()
        data_encodage.encode_categorical_values()
        data_encodage.encode_continue_score_grav()
        data_encodage.validate_data_and_export()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        DataEncodagePipeline().run()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise


