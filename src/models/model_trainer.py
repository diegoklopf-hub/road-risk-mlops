import os
import sys
from pathlib import Path
import logging

import joblib
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.entity import ModelTrainerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def make_features_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame):
        X_train = X_train.copy()
        X_test = X_test.copy()

        str_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            return X_train, X_test

        for col in str_cols:
            tr_num = pd.to_numeric(X_train[col], errors="coerce")
            te_num = pd.to_numeric(X_test[col], errors="coerce")

            tr_ok = tr_num.notna().mean()
            te_ok = te_num.notna().mean()

            if tr_ok > 0.99 and te_ok > 0.99:
                X_train[col] = tr_num
                X_test[col] = te_num
            else:
                all_vals = pd.concat(
                    [X_train[col].astype("string"), X_test[col].astype("string")],
                    axis=0
                ).astype("category")

                X_train[col] = all_vals.iloc[:len(X_train)].cat.codes.astype("int32")
                X_test[col] = all_vals.iloc[len(X_train):].cat.codes.astype("int32")

        return X_train, X_test

    def train(self):
        X_train = pd.read_csv(self.config.X_train_path)
        y_train = pd.read_csv(self.config.y_train_path).iloc[:, -1].astype(float)

        X_test = pd.read_csv(self.config.X_test_path)
        y_test = pd.read_csv(self.config.y_test_path).iloc[:, -1].astype(float)

        X_train, X_test = self.make_features_numeric(X_train, X_test)

        bad_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if bad_cols:
            raise TypeError(f"Non-numeric columns remaining: {bad_cols}")

        sw = getattr(self.config, "sample_weight_train_path", None)
        if sw is None:
            raise FileNotFoundError("sample_weight_train_path missing in config")

        w_path = Path(sw)
        if not w_path.exists():
            raise FileNotFoundError(f"sample_weight not found: {w_path}")

        w_train = pd.read_csv(w_path).iloc[:, -1].astype(float)
        if len(w_train) != len(X_train):
            raise ValueError(f"sample_weight length mismatch: {len(w_train)} vs {len(X_train)}")

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

        param_grid = {
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }

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

        grid.fit(X_train, y_train, model__sample_weight=w_train)

        return grid.best_estimator_, grid.best_params_


def main():
    logger.info(">>>>> Model Trainer started <<<<<")
    cm = ConfigurationManager()
    cfg = cm.get_model_trainer_config()
    trainer = ModelTrainer(cfg)
    trainer.train()
    logger.info(">>>>> Model Trainer completed <<<<<")


if __name__ == "__main__":
    main()
