
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

    def resample(self):
        X_train = pd.read_csv(self.config.input_x_path)
        y_train = pd.read_csv(self.config.input_y_path)

        # Check if the lengths of X_train and y_train match
        if (len(X_train) != len(y_train)):
            raise ValueError(f"Length of X_train ({len(X_train)}) and y_train ({len(y_train)}) do not match.")

        # Ajout d'une colonne gravité arrondie à la 10ene pour le resampling dans y et deplacement du score reel dans X pour le resampling
        X_train['score_grav'] = y_train['score_grav']
        y_train['grav_10'] = np.round(y_train['score_grav'], -1)
        y_train.drop(columns=['score_grav'], inplace=True)

        # Undersampling de la classe majoritaire pour équilibrer les classes
        logger.info(f"Undersampling on 'grav_10' column with strategy 'auto' (balance classes)")
        resample = RandomUnderSampler(sampling_strategy='auto')
        X_train_resample, y_train_resample = resample.fit_resample(X_train, y_train)

        y_train_resample["score_grav"] = X_train_resample["score_grav"]
        y_train_resample.drop(columns=["grav_10"], inplace=True)
        X_train_resample.drop(columns=["score_grav"], inplace=True)

        print(X_train_resample.head())
        print(y_train_resample.head())


        
        try:

            all_cols = set(list(X_train_resample.columns))

            is_schema_valid = SchemaManager(self.config.schema).check_schema(
                all_cols,
                self.config.status_file,
                "RESAMPLING",
                ignore_calib=True,
                filter_use_for_fit=True)

            if not is_schema_valid:
                raise ValueError(f"Schema validation failed: See {self.config.status_file} for details.")
            else:
                print("Exporting CSV")
                X_train_resample.to_csv(os.path.join(self.config.output_path, "X_train.csv"), index=False)
                y_train_resample.to_csv(os.path.join(self.config.output_path, "y_train.csv"), index=False)
                print(f"Data exported to {self.config.output_path}")
                logger.info("ENCODAGE export done")
                return is_schema_valid


        except Exception as e:
            logger.exception(e)
            raise

       
