import sys
import json
import joblib
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def make_features_numeric(X_train: pd.DataFrame):
        # Create copies so the caller's dataframes are not mutated.
        X_train = X_train.copy()

        # Identify string-like columns that require conversion.
        str_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            return X_train

        for col in str_cols:
            # First try a numeric coercion for values that are actually numbers stored as strings.
            tr_num = pd.to_numeric(X_train[col], errors="coerce")

            # Measure the share of valid numeric values to decide conversion strategy.
            tr_ok = tr_num.notna().mean()

            if tr_ok > 0.99:
                # Mostly numeric -> keep as numeric for model training.
                X_train[col] = tr_num
            else:
                # Use integer category codes so the model only sees numeric inputs.
                X_train[col] = X_train[col].astype("category").cat.codes.astype("int32")

        return X_train

    def train(self):

        # Load training and test data from configured paths.
        X_train = pd.read_csv(self.config.X_train_path)
        y_train = pd.read_csv(self.config.y_train_path).iloc[:, -1].astype(float)

        # Defensive alignment in case feature and target lengths differ.
        if len(X_train) != len(y_train):
            raise ValueError(f"Length mismatch: X_train has {len(X_train)} rows but y_train has {len(y_train)} rows.")

        # Ensure all features are numeric before fitting the model.
        X_train = self.make_features_numeric(X_train)

        bad_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if bad_cols:
            # Fail fast if any non-numeric columns remain after conversion.
            raise TypeError(f"Non-numeric columns remaining: {bad_cols}")

        # Model pipeline with a single XGBoost regressor step.
        pipe = Pipeline(
            steps=[
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=42,
                    ),
                )
            ]
        )

        # Hyperparameter grid kept intentionally small for faster search.
        param_grid = self.config.param_grid
        logger.info(f"Starting GridSearchCV with param_grid: {param_grid}")
        
        # KFold CV to evaluate candidates with shuffling for robustness.
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )

        grid.fit(X_train, y_train)

        # Persist the original feature order for downstream inference consistency.
        features_list = X_train.columns.tolist()

        return grid.best_estimator_, grid.best_params_, features_list
    
    def export_model(self, model, params, features_list):
        # Save the trained model and the feature list alongside the config paths.
        model_path = Path(getattr(self.config, "model_path"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

        features_path = Path(getattr(self.config, "features_path"))
        features_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(features_list, features_path)
        logger.info(f"Features list saved to: {features_path}")

        params_path = features_path.parent / "best_params.json"
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        logger.info(f"Best params saved to: {params_path}")

        # Log tuned hyperparameters for traceability.
        logger.info(f"Best params: {params}")


def main():
    # Entrypoint used by CLI or direct execution.
    logger.info(">>>>> Model Trainer started <<<<<")
    cm = ConfigurationManager()
    cfg = cm.get_model_trainer_config()
    trainer = ModelTrainer(cfg)
    model, params, features_list = trainer.train()
    trainer.export_model(model, params, features_list)
    logger.info(">>>>> Model Trainer completed <<<<<")


if __name__ == "__main__":
    main()
