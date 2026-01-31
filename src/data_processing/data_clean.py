import os
import pandas as pd
from src.custom_logger import logger
from src.data_processing.clean.clean_characteristics import clean_characteristics
from src.data_processing.clean.clean_locations import clean_lieux
from src.data_processing.clean.clean_users import clean_usagers
from src.data_processing.clean.clean_vehicles import clean_vehicles
from src.entity import DataCleanConfig


"""Utilities for importing and running cleaning steps on raw CSV accident data.

This module provides helpers to read yearly CSVs, verify consistent columns
across years, report missing values, and run dataset-specific cleaning
functions.
"""


def check_columns(df, expected_columns, year):
    """Ensure dataframe columns match expected set for consistency across years.

    If `expected_columns` is None (first year read), returns the current
    columns set so it can be used as the reference for following years.
    If there is a mismatch, a warning is printed showing missing/extra cols.
    """
    cols = set(df.columns)

    if expected_columns is None:
        return cols
    else:
        # Check by set difference between expected and actual columns
        missing = expected_columns - cols
        extra = cols - expected_columns
        if missing or extra:
            print(f"Warning: Column mismatch in {year}. Missing: {missing}, Extra: {extra}")
    return expected_columns


def read_csv(raw_data_path, base_name, from_year=2019, to_year=2024):
    """Read yearly CSVs for a given base filename and concatenate them.

    Args:
        raw_data_path: directory containing raw CSVs named like "base-YYYY.csv".
        base_name: prefix of CSV files (e.g. 'caracteristiques', 'lieux').
        from_year: first year to include (inclusive).
        to_year: last year to include (inclusive).

    Returns:
        A single pandas DataFrame containing rows from all successfully read years.
    """
    df = pd.DataFrame()
    expected_col = None
    for year in range(from_year, to_year + 1):
        file_path = os.path.join(raw_data_path, f"{base_name}-{year}.csv")
        print(f"Reading file: {file_path}")
        try:
            # Files use semicolon separators and latin1 encoding in this dataset
            new_df = pd.read_csv(file_path, sep=";", encoding="latin1", low_memory=False)
            expected_col = check_columns(new_df, expected_col, year)
            df = pd.concat([df, new_df], ignore_index=True)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            continue
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing {file_path}: {e}")
            continue
    return df


def check_df(df):
    """Log columns with missing values and their percentage.

    Useful as a quick diagnostic after each dataset-specific cleaning step.
    """
    print("Checking dataframe for missing values...")
    percent = (df.isna().sum() / len(df)) * 100
    na_columns = percent[percent > 0].sort_values(ascending=False)
    for column, value in na_columns.items():
        logger.warning(f"Column '{column}' has {value:.2f}% missing values.")

    # If more detailed exploration is needed, uncomment the lines below.
    # print("Describe dataframe:")
    # pd.set_option('display.max_columns', None)
    # print(df.describe())



class DataClean:
    def __init__(self, config: DataCleanConfig):
        self.config = config


    def clean_data(self):
        """Orchestrate the full cleaning pipeline for all datasets.

        This method reads raw yearly CSVs for each table, calls the
        corresponding cleaning function, and writes/checks outputs.
        """
        print("""------------- 01 Starting data cleaning -------------""")

        # Ensure output directory exists
        if os.path.exists(self.config.out_data_relative_path) == False:
            os.makedirs(self.config.out_data_relative_path)

        # `caracteristiques` dataset (characteristics)
        print("Import 'caracteristiques' dataset...")
        df_carac = read_csv(self.config.raw_data_relative_path, "caracteristiques", self.config.from_year, self.config.to_year)
        logger.info("Cleaning 'caracteristiques' data...")
        df_carac = clean_characteristics(df_carac, self.config.out_data_relative_path)
        check_df(df_carac)

        # `lieux` dataset (locations)
        print("Import 'lieux' dataset...")
        df_lieux = read_csv(self.config.raw_data_relative_path, "lieux", self.config.from_year, self.config.to_year)
        logger.info("Cleaning 'lieux' data...")
        df_lieux = clean_lieux(df_lieux, self.config.out_data_relative_path)
        check_df(df_lieux)

        # `usagers` dataset (users)
        print("Import 'usagers' dataset...")
        df_usagers = read_csv(self.config.raw_data_relative_path, "usagers", self.config.from_year, self.config.to_year)
        logger.info("Cleaning 'usagers' data...")
        df_usagers = clean_usagers(df_usagers, self.config.out_data_relative_path)

        # `vehicules` dataset (vehicles)
        print("Import 'vehicules' dataset...")
        df_vehicules = read_csv(self.config.raw_data_relative_path, "vehicules", self.config.from_year, self.config.to_year)
        logger.info("Cleaning 'vehicules' data...")
        df_vehicules = clean_vehicles(df_vehicules, self.config.out_data_relative_path, self.config.cluster_cat_vehicule)
        check_df(df_vehicules)