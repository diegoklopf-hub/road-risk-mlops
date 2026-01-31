import os
import numpy as np
from src.custom_logger import logger
from src.data_processing.check_structure import drop_columns

def merge_safety_codes(ligne):
    """
    Merge and concatenate non-empty safety codes from a row.
    This function extracts safety codes (secu1, secu2, secu3) from a given row,
    filters out invalid codes (marked as -1), sorts the remaining codes, and
    returns them as a concatenated string.
    Args:
        ligne (dict): A dictionary containing at least the keys 'secu1', 'secu2', 
                      and 'secu3' with integer values representing safety codes.
    Returns:
        str: A string of sorted safety codes concatenated together. Returns "-1" 
             if no valid safety codes are found (all codes are -1 or missing).
    Example:
        >>> merge_safety_codes({'secu1': 1, 'secu2': -1, 'secu3': 3})
        '13'
        >>> merge_safety_codes({'secu1': -1, 'secu2': -1, 'secu3': -1})
        '-1'
    """

    secus = [ligne['secu1'], ligne['secu2'], ligne['secu3']]
    secus = [v for v in secus if v != -1]
    if len(secus) == 0:
        return "-1"
    secus = sorted(secus)
    return "".join(str(v) for v in secus)

def clean_usagers(df,out_path):
    """Clean the `usagers` (users) DataFrame and save to `out_path`.

    Adds computed columns (age, secu_merged), fills defaults and drops low-quality
    columns. All messages and logs are in English.
    """
    print("    -> Initial shape of users dataframe:", df.shape)
    print("    -> Columns in the dataframe:", df.columns.tolist())


    # Clean target variable 'grav'
    df['grav'] = df['grav'].replace(-1, np.nan)
    # Map source grav categories to target coding
    df['grav'] = df['grav'].replace([1, 2, 3, 4], [0, 2, 3, 1])

    # Handle sentinel -1 values for categorical columns
    df['place'] = df['place'].replace(-1, 10)  # not applicable
    df['sexe'] = df['sexe'].replace(-1, 0)    # Unknown sex category
    
    # Compute age and handle outliers
    df["year_acc"] = df["Num_Acc"].astype(str).apply(lambda x: x[:4]).astype(int)
    df['age'] = df["year_acc"] - df["an_nais"]
    df.loc[(df["age"] > 120) | (df["age"] < 0), "age"] = np.nan
    df["age"] = df["age"].fillna(df.groupby('grav')['age'].transform('median'))
    df["age"] = df["age"].fillna(df["age"].median())

    # Clean categorical columns with default values
    df['trajet'] = df['trajet'].fillna(9)
    df["secu1"] = df["secu1"].fillna(-1).astype("int64")
    df["secu2"] = df["secu2"].fillna(-1).astype("int64")
    df["secu3"] = df["secu3"].fillna(-1).astype("int64")
    median_by_catu = df.groupby('catu')['locp'].median()
    df['locp'] = df['locp'].fillna(df['catu'].map(median_by_catu))
    median_by_catu = df.groupby('catu')['etatp'].median()
    df['etatp'] = df['etatp'].fillna(df['catu'].map(median_by_catu))

    # Merge safety equipment modalities
    df["secu_merged"] = df.apply(merge_safety_codes, axis=1)

    # Drop columns with more than 10% missing values
    percent = (df.isna().sum() / len(df)) * 100
    cols_to_drop = percent[percent > 10].index
    cols_to_drop = cols_to_drop.tolist()
    cols_to_drop.extend(["secu1", "secu2", "secu3", "an_nais", "year_acc"])
    df = drop_columns(df, cols_to_drop, logger, "usagers.csv")

    # Save cleaned file
    df.to_csv(os.path.join(out_path, "usagers.csv"), index=False)
    print("    -> Cleaned 'usagers' data saved to:", os.path.join(out_path, "usagers.csv"))

    return df
