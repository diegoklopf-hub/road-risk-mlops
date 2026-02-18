import sys
from pathlib import Path
import mlflow
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.modeling.shap_explicability import ShapExplicability


from src.mlflow_parent import get_or_create_parent_run

STAGE_NAME = "09 - SHAP Explicability stage"

class ShapExplicabilityPipeline:

    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):

        cfg = self.config.get_shap_explicability_config()
        shap_exp = ShapExplicability(cfg)

        start = time.time()

        # 🔵 compute + internal logging (then re-log in nested run)
        shap_exp.train_explainer()

        duration = time.time() - start


        return duration


    def main(self):
        
        # 🔵 Reconnexion au parent run
        parent_run_id = get_or_create_parent_run()
        mlflow.start_run(run_id=parent_run_id)

        try:
            # 🟢 nested run visible in UI
            with mlflow.start_run(run_name="09_shap_explicability", nested=True):

                mlflow.log_param("step", "09_shap_explicability")

                duration = self.run()

                mlflow.log_metric("evaluation_duration_sec", duration)

                mlflow.set_tag("stage", "evaluation")
                mlflow.log_param("status", "completed")

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        finally:
            # 🔴 CRUCIAL, otherwise nothing appears in UI
            mlflow.end_run()


if __name__ == "__main__":
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    try:
        ShapExplicabilityPipeline().main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=======x")
    except Exception:
        logger.exception("Evaluation failed")
        logger.error(f">>>>> stage {STAGE_NAME} failed <<<<<\n\nx=======x")
        raise
