import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from src.custom_logger import logger
from src.entity import ShapExplicabilityConfig


class ShapExplicability:
    def __init__(self, config: ShapExplicabilityConfig):
        self.config = config

    def train_explainer(self):
        logger.info("Loading model pipeline from path: %s", self.config.model_path)
        pipeline = joblib.load(self.config.model_path)
        
        # Extract the model because SHAP does not support the full Pipeline object
        model = pipeline.steps[-1][1]
        logger.info("Building SHAP TreeExplainer for model type: %s", type(model))

        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                explainer = shap.TreeExplainer(booster)
            else:
                explainer = shap.TreeExplainer(model)
        except Exception:
            logger.exception("Failed to build SHAP TreeExplainer for model type: %s", type(model))
            raise

        output_path = Path(self.config.shap_explainer_path)
        logger.info("Saving SHAP explainer to path: %s", output_path)
        try:
            joblib.dump(explainer, output_path)
        except Exception:
            logger.exception("Failed to save SHAP explainer to %s", output_path)
            raise

        logger.info("SHAP explainer saved successfully (%d bytes)", output_path.stat().st_size)
        return explainer