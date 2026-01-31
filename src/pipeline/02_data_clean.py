import sys
from pathlib import Path

# Add parent directory to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_clean import DataClean

# Define stage name
STAGE_NAME = "02 - Data Clean stage"

class DataCleanPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_clean_config = config.get_data_clean_config()
        data_clean = DataClean(config = data_clean_config)
        data_clean.clean_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataCleanPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
        
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise e

