import sys
from pathlib import Path

# Add parent directory to path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.custom_logger import logger
from src.config_manager import ConfigurationManager
from src.data_processing.data_encoding import DataEncodage

# Define stage name
STAGE_NAME = "04 - Encodage stage"

class DataEncodagePipeline:
    def __init__(self): 
        pass

    def main(self):
        config = ConfigurationManager()
        data_encodage_config = config.get_data_encodage_config()
        data_encodage = DataEncodage(config = data_encodage_config)
        data_encodage.encode_cyclic_values()
        data_encodage.encode_categorical_values()
        data_encodage.encode_continue_score_grav()
        data_encodage.validate_data_and_export()
    

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataEncodagePipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
        
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise e
        

