import os
import re

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True
    
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if os.path.exists(folder_path) == False :
        while True:
            response = input(f"{os.path.basename(folder_path)} doesn't exist. Do you want to create it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return False
    
def check_schema(columns_list, schema, status_file, phase,ignore_calib=False):
    return check_columns(columns_list, schema.keys(), status_file, phase,ignore_calib)
    
def check_columns(columns_list, expected_cols_list, status_file, phase,ignore_calib=False):
    """
    Validate a DataFrame's columns against a predefined schema.
    Compares the columns present in a DataFrame with the expected columns defined
    in a schema dictionary. Identifies missing and extra columns, then writes the
    validation results to a status file.
    Parameters
    ----------
    columns_list : list
        List of column names present in the DataFrame to validate.
    schema : dict
        Dictionary where keys represent the expected column names.
    status_file : str
        Path to the file where validation results will be written.
    phase : str
        Name or identifier of the data processing phase being validated.
        Used as a prefix in the status file output.
    Returns
    -------
    bool
        True if validation passes (no missing or extra columns), False otherwise.
    Notes
    -----
    - Results are written to the specified status_file in write mode.
    - The function compares column sets to detect discrepancies.
    - Missing columns indicate schema violations that prevent processing.
    - Extra columns may indicate unexpected data or preprocessing issues.
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
    
    return validation_status

def drop_columns(df,cols_to_drop, logger, filename):
    """
    Remove specified columns from a DataFrame and log the operation.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame from which columns will be dropped.
    cols_to_drop : list or Index
        Column names to be removed from the DataFrame.
    logger : logging.Logger
        Logger instance for recording the operation.
    filename : str
        Name of the file being processed (used for logging context).
    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns removed.
    """
       
    msg = f"{filename}: Drop the columns  {list(cols_to_drop)}"
    df = df.drop(columns=cols_to_drop)
    print(msg)
    logger.info(msg)
    return df
