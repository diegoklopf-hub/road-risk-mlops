from pathlib import Path
import sys
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.models.model_trainer import ModelTrainer
from src.data_processing.schema_manager import SchemaManager
from src.common_utils import append_status, is_last_status_ok
from src.config import STATUS_FILE

STAGE_NAME = "07 - Model Trainer stage"

class ModelTrainerPipeline:
    def __init__(self):
        self.cm = ConfigurationManager()
        self.status_file = STATUS_FILE

    def run(self):
        if not is_last_status_ok(self.status_file):
            raise RuntimeError("Previous stage status is not OK. Aborting model training.")
        
        cfg = self.cm.get_model_trainer_config()
        trainer = ModelTrainer(cfg)

        best_model, best_params, features_list = trainer.train()
        trainer.export_model(best_model,best_params, features_list)
        return best_model, best_params


if __name__ == '__main__':
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        pipe = ModelTrainerPipeline()
        pipe.run()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise