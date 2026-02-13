from pathlib import Path
import pandas as pd
from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.models.model_evaluation import ModelEvaluation  # adapte si ton chemin est différent
from src.data_processing.schema_manager import SchemaManager
from src.common_utils import append_status, is_last_status_ok
from src.config import STATUS_FILE


STAGE_NAME = "08 - Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        self.cm = ConfigurationManager()
        self.status_file = STATUS_FILE

    def run(self):
        if not is_last_status_ok(self.status_file):
            raise RuntimeError("Previous stage status is not OK. Aborting model evaluation.")
        
        cfg = self.cm.get_model_evaluation_config()
        evaluator = ModelEvaluation(cfg)
        evaluator.log_into_mlflow()

        metric_file = getattr(cfg, "metric_file_name", None)
        if metric_file:
            from pathlib import Path
            import json
            p = Path(metric_file)
            if p.exists():
                scores = json.loads(p.read_text())
                logger.info(f"Metrics: {scores}")

        return True

if __name__ == '__main__':
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        pipe = ModelEvaluationPipeline()
        pipe.run()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise