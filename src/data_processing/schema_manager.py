import re
from pathlib import Path
from src.common_utils import append_status

class SchemaManager:
    def __init__(self, schema):
        self.schema = schema


    def get_keys(self):
        """
        Returns a list of all column names defined in the schema.
        Example: get_keys() -> ['mois', 'jour', 'hrmn', ...]
        """
        return list(self.schema.get('COLUMNS', {}).keys())


    def get_description(self, column_name):
        """
        Returns the description for a specific column name.
        Example: get_description('mois') -> "Mois de l'accident"
        """
        # First access the column dictionary,
        # then fetch the value of the 'description' key
        column_info = self.schema.get('COLUMNS', {}).get(column_name)
        
        if column_info and 'description' in column_info:
            return column_info['description']
        
        return f"Description not found for column: {column_name}"

    def check_schema(self, columns_list, status_file, phase,ignore_calib=False, filter_use_for_fit=False):
        """
        Validate that the provided columns match the schema keys.

        Parameters
        ----------
        columns_list : list
            List of column names to validate against the schema.
        schema : dict
            Dictionary containing the expected schema with keys representing column names.
        status_file : str or Path
            File path where validation status or results will be recorded.
        phase : str
            Processing phase identifier (e.g., 'train', 'test', 'validation').
        ignore_calib : bool, optional
            Flag to ignore calibration-related columns during validation. Default is False.

        Returns
        -------
        bool or dict
            Result of the schema validation check performed by check_columns function.

        See Also
        --------
        check_columns : Underlying function that performs the actual column validation.
        """
    
        if filter_use_for_fit:
            expected_columns = [
                col for col, info in self.schema.items() 
                if isinstance(info, dict) and info.get('use_for_fit') is True
            ]
        else:
            expected_columns = self.schema.keys()

        return check_columns(columns_list, expected_columns, status_file, phase,ignore_calib)
        
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

    details_lines = []
    if missing_cols:
        details_lines.append(f"Missing columns in DataFrame: {list(missing_cols)}")
        validation_status = False
    if extra_cols:
        details_lines.append(f"Extra columns in DataFrame: {list(extra_cols)}")
        validation_status = False

    details = "\n".join(details_lines) if details_lines else None
    append_status(Path(status_file), phase, validation_status, details)
    
    return validation_status