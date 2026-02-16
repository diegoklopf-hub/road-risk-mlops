import os
import re
from src.custom_logger import logger

def check_existing_file(file_path):
    if os.path.isfile(file_path):
        logger.info(f"{file_path} exists → overwrite allowed")
        return True
    return True

def check_existing_folder(folder_path):
    """
    Docker/ML pipeline version: auto-create folder if missing.
    No interactive prompt.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created missing folder: {folder_path}")
        return True
    return True

# ==============================
# SCHEMA VALIDATION
# ==============================

def check_schema(columns_list, schema, status_file, phase, ignore_calib=False):
    return check_columns(columns_list, schema.keys(), status_file, phase, ignore_calib)


def check_columns(columns_list, expected_cols_list, status_file, phase, ignore_calib=False):
    """
    Validate a DataFrame's columns against a predefined schema.
    """
    validation_status = True

    if ignore_calib:
        columns_list = {re.sub(r'_-?\d+$', '', col) for col in columns_list}
        columns_list = {re.sub(r'_(A|B)$', '', col) for col in columns_list}

    schema_columns = set(expected_cols_list)
    df_columns = set(columns_list)

    extra_cols = df_columns - schema_columns
    missing_cols = schema_columns - df_columns

    with open(status_file, 'w') as f:
        f.write(f"{phase}:")
        if missing_cols:
            f.write(f"Missing columns in DataFrame: {list(missing_cols)}\n")
            validation_status = False
        if extra_cols:
            f.write(f"Extra columns in DataFrame: {list(extra_cols)}\n")
            validation_status = False
        f.write(f"Validation status: {validation_status}")

    logger.info(f"{phase} schema validation completed. Status: {validation_status}. See {status_file} for details.")
    return validation_status


def drop_columns(df, cols_to_drop, logger, filename):
    """
    Remove specified columns from a DataFrame and log the operation.
    """
    msg = f"{filename}: Drop the columns  {list(cols_to_drop)}"
    df = df.drop(columns=cols_to_drop)
    print(msg)
    logger.info(msg)
    return df
