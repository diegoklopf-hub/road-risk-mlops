import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.custom_logger import logger
from src.entity import DataTransformationConfig
from pathlib import Path
import pandas as pd
from src.common_utils import append_status
from src.config import STATUS_FILE
from src.data_processing.schema_manager import SchemaManager



"""Data transformation utilities.

This module handles splitting, normalization and feature selection for the
prepared dataset. All user-visible prints and logger messages are in English.
"""


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        """Split input data into training and test sets.

        Returns X_train, X_test, y_train, y_test.
        """
        print("""------------- 01 Split train/test -------------""")
        data = pd.read_csv(self.config.input_path)

        X = data.drop(columns=["score_grav"])
        y = data["score_grav"]

        # --- FIX CORSE: dep/com peuvent contenir "2A"/"2B" -> rendre numérique

        # dep: "2A"/"2B" -> 20/21
        if "dep" in X.columns:
            X["dep"] = X["dep"].astype("string").replace({"2A": "20", "2B": "21"})
            X["dep"] = pd.to_numeric(X["dep"], errors="coerce")

        # com: peut valoir "2A271"/"2B120" -> "20271"/"21120" puis numérique
        if "com" in X.columns:
            X["com"] = X["com"].astype("string")
            X["com"] = X["com"].str.replace(r"^2A", "20", regex=True)
            X["com"] = X["com"].str.replace(r"^2B", "21", regex=True)
            X["com"] = pd.to_numeric(X["com"], errors="coerce")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # X_train.to_csv(os.path.join(self.config.train_test_path, "X_train.csv"), index = False)
        y_train.to_csv(os.path.join(self.config.train_test_path, "y_train.csv"), index = False)
        # X_test.to_csv(os.path.join(self.config.train_test_path, "X_test.csv"), index = False)
        y_test.to_csv(os.path.join(self.config.train_test_path, "y_test.csv"), index = False)

        logger.info("Split data into training and test sets")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        return X_train, X_test, y_train, y_test
    
    def normalize(self, X_train, X_test):
        """Normalize configured numeric features using MinMaxScaler.

        Only columns marked `normalized` in the `schema` are scaled. The
        function matches expanded column names (OneHotEncoder) back to the
        base column name using regex before applying scaling.
        """
        print("""------------- 02 Normalizing features -------------""")
        # Get cols_to_normalize config  
        cols_used_for_prediction = []
        for col_name, properties in self.config.schema.items():
            if properties.get("use_for_fit") is True:
                cols_used_for_prediction.append(col_name)

        # Get columns from X_test (matching name)
        X_cols_to_normalize = []
        for X_col_name in X_train.columns:
            short_name = re.sub(r'_-?\d+$', '', X_col_name)
            short_name = re.sub(r'_(A|B)$', '', short_name) 
            if short_name in cols_used_for_prediction:
                X_cols_to_normalize.append(X_col_name)


        scaler = MinMaxScaler()
        X_train[X_cols_to_normalize] = scaler.fit_transform(X_train[X_cols_to_normalize])
        X_test[X_cols_to_normalize] = scaler.transform(X_test[X_cols_to_normalize])

        # X_train.to_csv(os.path.join(self.config.train_test_path, "X_train_norm.csv"), index = False)
        # X_test.to_csv(os.path.join(self.config.train_test_path, "X_test_norm.csv"), index = False)

        return X_train, X_test
    
    def features_selection(self,X_train, X_test):
        """Select features configured for model fitting and save results.

        Columns with `use_for_fit=True` in the schema are kept. The function
        handles expanded/dummified column names by mapping them back to base
        names using regex.
        """
        print("""------------- 03 Feature selection -------------""")

        # Build list of base columns configured to be used for prediction
        cols_used_for_prediction = []
        for col_name, properties in self.config.schema.items():
            if properties.get("use_for_fit") is True:
                cols_used_for_prediction.append(col_name)

        # Get columns from X_test (matching name)
        X_cols_used_for_prediction = []
        for X_col_name in X_train.columns:
            short_name = re.sub(r'_-?\d+$', '', X_col_name)
            short_name = re.sub(r'_(A|B)$', '', short_name) 
            if short_name in cols_used_for_prediction:
                X_cols_used_for_prediction.append(X_col_name)

        logger.info(f"Feature selection: {X_train.shape[1]} -> {len(X_cols_used_for_prediction)} columns remaining.")
        X_train = X_train[X_cols_used_for_prediction]
        X_test = X_test[X_cols_used_for_prediction]
        X_train.to_csv(os.path.join(self.config.train_test_path, "X_train.csv"), index = False)
        X_test.to_csv(os.path.join(self.config.train_test_path, "X_test.csv"), index = False)
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")

    def check_transformation_outputs(self):
        output_x_train = Path(self.config.train_test_path) / "X_train.csv"
        output_x_test = Path(self.config.train_test_path) / "X_test.csv"
        if not output_x_train.exists() or not output_x_test.exists():
            details = f"Missing outputs: {[str(p) for p in [output_x_train, output_x_test] if not p.exists()]}"
            append_status(STATUS_FILE, "DATA TRANSFORMATION", False, details)
            raise FileNotFoundError(details)

        all_cols = set(pd.read_csv(output_x_train).columns)
        is_schema_valid = SchemaManager(self.config.schema).check_schema(
            all_cols,
            self.config.status_file,
            "TRANSFORMATION",
            ignore_calib=True,
            filter_use_for_fit=True,
        )
        if not is_schema_valid:
            raise ValueError("Schema validation failed during transformation.")


        
