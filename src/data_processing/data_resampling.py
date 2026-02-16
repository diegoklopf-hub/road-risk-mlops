import os
from pathlib import Path
import pandas as pd
from src.custom_logger import logger
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from src.data_processing.schema_manager import SchemaManager
from src.entity import DataResamplingConfig


class DataResampling:
    def __init__(self, config: DataResamplingConfig):
        self.config = config

    def resample_data(self):

        logger.info("Starting resampling...")

        X_train = pd.read_csv(self.config.input_x_path)
        y_train = pd.read_csv(self.config.input_y_path)

        # safety check
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Length mismatch X_train ({len(X_train)}) vs y_train ({len(y_train)})"
            )

        # add score for resampling
        X_train["score_grav"] = y_train["score_grav"]
        y_train["grav_10"] = np.round(y_train["score_grav"], -1)
        y_train.drop(columns=["score_grav"], inplace=True)

        # undersampling
        logger.info("Undersampling with RandomUnderSampler...")
        rus = RandomUnderSampler(sampling_strategy="auto")

        X_res, y_res = rus.fit_resample(X_train, y_train)

        # restore real score
        y_res["score_grav"] = X_res["score_grav"]
        y_res.drop(columns=["grav_10"], inplace=True)
        X_res.drop(columns=["score_grav"], inplace=True)

        logger.info(f"Resampled shape X: {X_res.shape}")
        logger.info(f"Resampled shape y: {y_res.shape}")

        # create output dir
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)

        # schema validation
        try:
            all_cols = set(X_res.columns)

            is_schema_valid = SchemaManager(self.config.schema).check_schema(
                all_cols,
                self.config.status_file,
                "RESAMPLING",
                ignore_calib=True,
                filter_use_for_fit=True,
            )

            if not is_schema_valid:
                raise ValueError(
                    f"Schema validation failed. Check {self.config.status_file}"
                )

            # save files
            X_res.to_csv(os.path.join(self.config.output_path, "X_train.csv"), index=False)
            y_res.to_csv(os.path.join(self.config.output_path, "y_train.csv"), index=False)

            logger.info(f"Resampled data saved to {self.config.output_path}")
            return True

        except Exception as e:
            logger.exception("Resampling export failed")
            raise
