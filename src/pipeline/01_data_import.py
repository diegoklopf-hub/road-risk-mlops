import sys
from pathlib import Path

# Add parent directory to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_import import DataImport

# Define stage name
STAGE_NAME = "01 - Data Import stage"

class DataImportPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"CWD={Path.cwd()}")
        config = ConfigurationManager()
        data_import_config = config.get_data_import_config()
        data_import = DataImport(config = data_import_config)
        data_import.import_csv()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataImportPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
        
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise 

