import numpy as np
import pandas as pd
from src.data_processing.check_structure import drop_columns
from src.data_processing.schema_manager import SchemaManager
from src.entity import DataEncodeConfig
from src.custom_logger import logger
from sklearn.preprocessing import OneHotEncoder
import joblib

def calculate_99th_percentile_grav(df):
    """
    Calculate the 99th percentile of weighted injury severity scores.
    This function computes three weighted risk scores based on injury counts,
    using predefined weights for different severity levels, and returns the
    99th percentile value for each category.
    Args:
        df (pandas.DataFrame): A DataFrame containing injury count columns:
            - count_blesse_leger: Number of lightly injured persons
            - count_blesse_hosp: Number of hospitalized injured persons
            - count_tue: Number of fatalities
    Returns:
        tuple: A tuple of three float values representing the 99th percentile:
            - R_max_BL (float): 99th percentile of lightly injured weighted score
            - R_max_BH (float): 99th percentile of hospitalized injured weighted score
            - R_max_D (float): 99th percentile of fatalities weighted score
    Note:
        Weights applied: w_L=2 (light injury), w_H=5 (hospitalized), w_D=10 (fatality)
    """
    
    w_D, w_H, w_L = 10, 5, 2

    R_BL_series = w_L * df.count_blesse_leger
    R_BH_series = w_H * df.count_blesse_hosp + w_L * df.count_blesse_leger
    R_D_series  = w_D * df.count_tue + w_H * df.count_blesse_hosp + w_L * df.count_blesse_leger

    R_max_BL = R_BL_series.quantile(0.99)
    R_max_BH = R_BH_series.quantile(0.99)
    R_max_D  = R_D_series.quantile(0.99)

    return R_max_BL, R_max_BH, R_max_D

def compute_score_grav(D, H, L,R_max_BL, R_max_BH, R_max_D,w_D=10, w_H=5, w_L=2,alpha=1.0):
    """Severity score on a 0-100 scale with three tiers:

    - 0-40  : only light injuries (L > 0, H = 0, D = 0)
    - 40-80 : at least one hospitalized injury, no fatality (H > 0, D = 0)
    - 80-100: at least one fatality (D > 0)
    """

    # == only light injuries (0 to 40) ==
    if D == 0 and H == 0:
        R_BL = w_L * L
        s = min(R_BL / R_max_BL, 1) if R_max_BL > 0 else 0
        return 40 * (s ** alpha)

    # == at least 1 hospitalized, no fatality (40 to 80) ==
    if D == 0 and H > 0:
        R_BH = w_H * H + w_L * L
        s = min(R_BH / R_max_BH, 1) if R_max_BH > 0 else 0
        return 40 + 40 * (s ** alpha)

    # == at least 1 fatality (80 to 100) ==
    if D > 0:
        R_D = w_D * D + w_H * H + w_L * L
        s = min(R_D / R_max_D, 1) if R_max_D > 0 else 0
        return 80 + 20 * (s ** alpha)

class DataEncodage:
    def __init__(self, config: DataEncodeConfig):
        self.config = config
        self.df = None

    def encode_cyclic_values(self):
        print("""------------- 01 Encoding cyclic features -------------""")
        dtype_dict = {'dep': str, 'com': str}
        self.df = pd.read_csv(self.config.merged_data_path, dtype=dtype_dict)

        self.df['day_of_year'] = pd.to_datetime(
            self.df[['an', 'mois', 'jour']].rename(columns={
                "an": "year",
                "mois": "month",
                "jour": "day",
            }),
            errors="raise"
        ).dt.dayofyear

        self.df['days_in_year'] = np.where((self.df["an"] % 4 == 0), 366, 365)
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / self.df['days_in_year'])
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / self.df['days_in_year'])

        self.df["minute"] = pd.to_datetime(self.df["hrmn"]).dt.hour * 60 + pd.to_datetime(self.df["hrmn"]).dt.minute
        self.df["minute_sin"] = np.sin(2 * np.pi * self.df["minute"] / 1440)
        self.df["minute_cos"] = np.cos(2 * np.pi * self.df["minute"] / 1440)

        self.df['week_day'] = pd.to_datetime(self.df[['an', 'mois', 'jour']].rename(columns={
            "an": "year",
            "mois": "month",
            "jour": "day",
        })).dt.weekday
        self.df['week_day_sin'] = np.sin(2 * np.pi * self.df['week_day'] / 7)
        self.df['week_day_cos'] = np.cos(2 * np.pi * self.df['week_day'] / 7)


        cols_to_drop = ['day_of_year', 'days_in_year', 'minute', 'hrmn', 'Hours', 'jour', 'mois','an','week_day']
        self.df = drop_columns(self.df, cols_to_drop, logger, "merged_data.csv")

        logger.info('Cyclic encoding completed successfully')
        return self.df

    def encode_categorical_values(self):
        print("""------------- 02 Encode categorical features  -------------""")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(self.df[self.config.encode_columns])

        # Build DataFrame with generated column names
        column_names = encoder.get_feature_names_out(self.config.encode_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=column_names, index=self.df.index)

        self.df = self.df.drop(columns=self.config.encode_columns)
        self.df = pd.concat([self.df, encoded_df], axis=1)

        print(f"Encoded joblib saved to: {self.config.model_one_hot_encoder_path}")
        joblib.dump(encoder, self.config.model_one_hot_encoder_path)

    def encode_continue_score_grav(self):
        print("""------------- 03 Encode continuous severity score -------------""")
        R_max_BL, R_max_BH, R_max_D = calculate_99th_percentile_grav(self.df)

        self.df["score_grav"] = self.df.apply(
            lambda row: compute_score_grav(
                D=row["count_tue"],
                H=row["count_blesse_hosp"],
                L=row["count_blesse_leger"],
                R_max_BL=R_max_BL,
                R_max_BH=R_max_BH,
                R_max_D=R_max_D,
            ),
            axis=1
        )

    def validate_data_and_export(self):
        print("------------- 04 Validating data structure and export -------------")

        try:

            all_cols = set(list(self.df.columns))
            is_schema_valid = SchemaManager(self.config.schema).check_schema(all_cols,self.config.status_file,"ENCODAGE",ignore_calib=True)

            if not is_schema_valid:
                raise ValueError(f"Schema validation failed: See {self.config.status_file} for details.")
            else:
                print("Exporting CSV")
                self.df.to_csv(self.config.merged_data_encoded_path, index=False)
                print(f"Data exported to {self.config.merged_data_encoded_path}")
                logger.info("ENCODAGE export done")
                return is_schema_valid


        except Exception as e:
            logger.exception(e)
            raise
