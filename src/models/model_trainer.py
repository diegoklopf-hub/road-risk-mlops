import sys
import joblib
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from src.config_manager import ConfigurationManager
from src.custom_logger import logger
from src.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def make_features_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame):
        # Create copies so the caller's dataframes are not mutated.
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Identify string-like columns that require conversion.
        str_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            return X_train, X_test

        for col in str_cols:
            # First try a numeric coercion for values that are actually numbers stored as strings.
            tr_num = pd.to_numeric(X_train[col], errors="coerce")
            te_num = pd.to_numeric(X_test[col], errors="coerce")

            # Measure the share of valid numeric values to decide conversion strategy.
            tr_ok = tr_num.notna().mean()
            te_ok = te_num.notna().mean()

            if tr_ok > 0.99 and te_ok > 0.99:
                # Mostly numeric -> keep as numeric for model training.
                X_train[col] = tr_num
                X_test[col] = te_num
            else:
                # Otherwise, encode categories consistently across train and test.
                all_vals = pd.concat(
                    [X_train[col].astype("string"), X_test[col].astype("string")],
                    axis=0
                ).astype("category")

                # Use integer category codes so the model only sees numeric inputs.
                X_train[col] = all_vals.iloc[:len(X_train)].cat.codes.astype("int32")
                X_test[col] = all_vals.iloc[len(X_train):].cat.codes.astype("int32")

        return X_train, X_test

    def train(self):

        # Load training and test data from configured paths.
        X_train = pd.read_csv(self.config.X_train_path)
        y_train = pd.read_csv(self.config.y_train_path).iloc[:, -1].astype(float)
        X_test = pd.read_csv(self.config.X_test_path)

        # Defensive alignment in case feature and target lengths differ.
        if len(X_train) != len(y_train):
            raise ValueError(f"Length mismatch: X_train has {len(X_train)} rows but y_train has {len(y_train)} rows.")

        # Ensure all features are numeric before fitting the model.
        X_train, X_test = self.make_features_numeric(X_train, X_test)

        bad_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
        if bad_cols:
            # Fail fast if any non-numeric columns remain after conversion.
            raise TypeError(f"Non-numeric columns remaining: {bad_cols}")

        # Build sample weights inversely proportional to target-bin frequency
        # to reduce the influence of over-represented target ranges.
        w_train = None
        if (w_train is None) or (len(w_train) != len(X_train)):
            y_bins = pd.qcut(y_train, q=10, duplicates="drop")
            freq = y_bins.value_counts()
            inv = (1.0 / freq).to_dict()
            w_train = y_bins.map(inv).astype(float)

        w_train = w_train.iloc[:len(X_train)].to_numpy()

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
        
        # KFold CV to evaluate candidates with shuffling for robustness.
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        # Run two searches: with and without sample weights, then keep the best.
        best_grid = None
        best_score = None
        best_use_weights = None

        for use_weights in [True, False]:
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=-1,
                verbose=2,
                refit=True,
            )

            fit_kwargs = {"model__sample_weight": w_train} if use_weights else {}
            grid.fit(X_train, y_train, **fit_kwargs)

            logger.info(
                f"CV best score ({'with' if use_weights else 'without'} sample_weight): {grid.best_score_}"
            )

            if (best_score is None) or (grid.best_score_ > best_score):
                best_grid = grid
                best_score = grid.best_score_
                best_use_weights = use_weights

        logger.info(f"Best fit used sample_weight: {best_use_weights}")

        # Persist the original feature order for downstream inference consistency.
        features_list = X_train.columns.tolist()

        return best_grid.best_estimator_, best_grid.best_params_, features_list
    
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
