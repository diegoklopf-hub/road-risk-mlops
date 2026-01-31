import sys
from pathlib import Path

# Add parent directory to path
parent_folder = str(Path(__file__).parent.parent.parent)
sys.path.append(parent_folder)

from src.config_manager import ConfigurationManager
from src.data_processing.data_transformation import DataTransformation
from src.custom_logger  import logger

STAGE_NAME = "05 - Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()


            with open(Path(data_transformation_config.STATUS_FILE), 'r') as f:
                status = f.read().split(" ")[-1]
            
            if status == "True":
               
                data_transformation = DataTransformation(config = data_transformation_config)
                X_train, X_test, _, _ = data_transformation.train_test_splitting()
                X_train, X_test = data_transformation.normalize(X_train, X_test)
                data_transformation.features_selection(X_train, X_test)
            else:
                print()
                raise Exception("Your data schema is not valid")
        
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj =  DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise e