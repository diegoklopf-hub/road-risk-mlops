import sys
from pathlib import Path

# Add parent directory to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_merge import DataMerge

# Define stage name
STAGE_NAME = "03 - Merge stage"

class DataMergePipeline:
    def __init__(self): 
        pass

    def main(self):
        config = ConfigurationManager()
        data_merge_config = config.get_data_merge_config()
        data_merge = DataMerge(config = data_merge_config)
        data_merge.merge_by_usager()
        data_merge.feature_engineering()
        data_merge.merge_by_accident()
        data_merge.validate_data_and_export()


if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataMergePipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
        
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise e

