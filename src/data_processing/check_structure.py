import os
import re
from src.custom_logger import logger

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
