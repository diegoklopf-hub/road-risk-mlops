import os
import numpy as np
import pandas as pd
from src.custom_logger import logger
from src.data_processing.check_structure import drop_columns

def clean_categorical_columns(df,colonnes, default_value):
    """Clean a list of categorical columns replacing common bad tokens.

    Args:
        df: DataFrame to operate on
        colonnes: iterable of column names to clean
        default_value: value to use for missing/invalid entries
    """
    for col in colonnes:
        print(f"    -> Cleaning column: {col} -> default: {default_value}")
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["\xa0", "nan", "", "-", "--", "-1", "-1.", " -1", " -1."], default_value)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-1, default_value)
    return df

def clean_lieux(df,out_path):
    """Clean and standardize `lieux` (location) DataFrame and save to `out_path`.

    Returns the cleaned DataFrame. All user-visible messages are English.
    """
    print("    -> Initial shape of lieux dataframe:", df.shape)
    print("    -> Columns in the dataframe:", df.columns.tolist())

    # Clean "nbv" column
    df["nbv"] = df["nbv"].astype(str).str.strip()
    df["nbv"] = df["nbv"].replace(["#ERREUR", "#VALEURMULTI", "","-1.0"], np.nan)
    df["nbv"] = pd.to_numeric(df["nbv"], errors="coerce")
    df["nbv"] = df["nbv"].fillna(df["nbv"].mode()[0])

    # Clean "pr" and "pr1" columns
    df = clean_categorical_columns(df,["pr", "pr1"], np.nan)

    # Clean "larrout" column
    df["larrout"] = df["larrout"].replace(",", ".")
    df = clean_categorical_columns(df,["larrout"], np.nan)
    mode_value = df["larrout"].mode()[0]
    df.loc[df['larrout'] > 150, 'larrout'] = (df.loc[df['larrout'] > 150, 'larrout'] // 10)
    df["larrout"] = df["larrout"].fillna(mode_value)

    # Clean "v1" column
    df = clean_categorical_columns(df,["v1"], 0)

    # Clean "vma" column
    df = clean_categorical_columns(df,["vma"], np.nan)
    df.loc[df['vma'] > 130, 'vma'] = (df.loc[df['vma'] > 130, 'vma'] // 10)
    df['vma'] = df['vma'].where((df['vma'] >= 20) & (df['vma'] % 5 == 0), np.nan)
    df['vma'] = df['vma'].fillna(round(df['vma'].mean())).astype(int)

    # Clean "circ", "prof", "plan" columns
    df = clean_categorical_columns(df,["circ", "prof", "plan"], 5)

    # Clean "surf", "infra" columns
    df = clean_categorical_columns(df,["surf", "infra"], 9)

    # Clean "vosp" column
    df = clean_categorical_columns(df,["vosp"], 4)

    # Clean "situ" column
    df = clean_categorical_columns(df,["situ"], 8)

    # Drop columns with more than 10% missing values
    percent = (df.isna().sum() / len(df)) * 100
    cols_to_drop = percent[percent > 10].index
    cols_to_drop = cols_to_drop.tolist()
    df = drop_columns(df, cols_to_drop, logger, "lieux.csv")

    # Save cleaned file
    df.to_csv(os.path.join(out_path, "lieux.csv"), index=False)
    print("    -> Cleaned 'lieux' data saved to:", os.path.join(out_path, "lieux.csv"))

    return df
