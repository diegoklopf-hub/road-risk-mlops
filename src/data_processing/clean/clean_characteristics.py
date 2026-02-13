import os
import numpy as np
import pandas as pd
import unidecode
import re


from src.data_processing.check_structure import drop_columns
from src.data_processing.clean.INSEE_config import MAPPING_CORSE
from src.custom_logger import logger

def clean_hours(hrmn):
    """Return a pd.Timestamp (time of day) or pd.NaT on failure.

    Args:
        hrmn: original time input (str, int, float, NaN...)
    """
    # Handle NaN/NaT inputs
    if pd.isna(hrmn):
        return pd.NaT

    # Normalize to string and strip whitespace
    heure_str = str(hrmn).strip()

    # Case 1: already in expected "HH:MM" format
    if ':' in heure_str:
        # Expect exactly 5 characters "HH:MM"
        if len(heure_str) == 5:
            # Parse to Timestamp; return pd.NaT if invalid
            return pd.to_datetime(heure_str, format='%H:%M', errors='coerce')
        else:
            # Contains colon but wrong length
            logger.warning(f"Invalid time format (colon present but wrong length): {heure_str}")
            return pd.NaT

    # Case 2: compact numeric formats (hhmm, hmm, mm, m)
    elif heure_str.isdigit():
        # Zero-pad to 4 digits (e.g. "9" -> "0009", "930" -> "0930")
        heure_str = heure_str.zfill(4)
        # Build "HH:MM" from the 4 digits
        heure_str = heure_str[:2] + ":" + heure_str[-2:]
        # Parse; return pd.NaT if invalid (e.g. "24:00")
        return pd.to_datetime(heure_str, format='%H:%M', errors='coerce')

    # Case 3: invalid format (contains letters/symbols, etc.)
    else:
        logger.warning(f"Invalid time format: {heure_str}")
        return pd.NaT

def clean_year(annee):
    """Normalize year to a 4-digit integer.

    Args:
        annee: original year value (int/float)
    Returns:
        int year in yyyy format or pd.NA if input is missing.
    """
    # Return missing indicator if input is NaN/NaT
    if pd.isna(annee):
       return pd.NA

    # If year is two digits (e.g. 21) assume 2000+year (dataset spans 2000s)
    if annee < 100:
        return annee + 2000
    else:
        # Otherwise assume already a 4-digit year
        return annee


# Create a set of valid department codes (INSEE)
VALID_DEPARTMENTS = set(list(range(1, 96)) + ["2A", "2B"] + list(range(971, 979)) + [984, 986, 987, 988, 989])

def clean_department(dep):
    """Standardize INSEE department code.

    Returns the standardized code (int or str) or np.nan on failure.
    """

    # 1. Explicitly handle missing inputs
    if pd.isna(dep):
        return np.nan

    # If input is a string, strip and try to convert to int when possible
    if isinstance(dep, str):
        dep = dep.strip()
        try:
            dep = int(dep)
        except ValueError:
            dep = str(dep)

    # If the code is already a valid department identifier, return it
    if dep in VALID_DEPARTMENTS:
        return dep
    # Special cases for Corsica codes encoded as 201/202
    elif dep == 201:
        return "2A"
    elif dep == 202:
        return "2B"
    # If code is numeric (int/float)
    elif isinstance(dep, (int, float)):
        # Handle values that appear multiplied by 10 (e.g. 1300 -> 130)
        if (dep >= 100) and (dep % 10 == 0):
            new_dep = int(dep / 10)
            return clean_department(new_dep)  # recursive simplification
        else:
            logger.warning(f"Error: [{dep}] is not a valid department code")
            return np.nan
    else:
        logger.warning(f"Error: [{dep}] is not a valid department code")
        return np.nan
    
def clean_commune_code(com,dep):
    """Standardize commune INSEE code: dep (2 chars) + commune (3 chars).
    
    Args:
        com: commune code from source. Can be a 3-digit code, 5-digit code (already prefixed with department),
            or numeric value. Missing values (NaN) are handled and returned as NaN.
        dep: department code (2 characters). Used as prefix for the standardized INSEE code.
            Only the first 2 characters are used if a longer value is provided.
    
    Returns:
        str: Standardized 5-digit commune code or np.nan on failure.
    """

    # Return NaN if commune code is missing
    if pd.isna(com):
        return np.nan

    # Convert to string and remove any trailing ".0"
    com_str = str(com).replace(".0", "")
    # Department prefix: first 2 characters, zero-padded
    dep_str = str(dep)[:2].zfill(2)

    # If commune code length is 5 (already dep+com), validate prefix
    if len(com_str) == 5:
        if com_str[:2] == dep_str:
            return com_str
        else:
            logger.warning(f"Mismatch between commune and department codes: [{com_str},{dep}]")
            return np.nan

    # If commune code is shorter than 4 chars, zero-pad to 3 and prepend dep
    if len(com_str) < 4:
        com_str = com_str.zfill(3)
        return dep_str + com_str

    # Unhandled cases
    logger.warning(f"Unhandled commune/department case: [{com_str},{dep_str}]")
    return np.nan


def handle_corse_codes(df):
    """Replace Corsica department code "20" with "2A" or "2B" where needed.

    Uses `MAPPING_CORSE` to map commune codes back to the correct department.
    """

    # Create inverse mapping from commune code to department
    dict_inverse = {
        code: departement
        for departement, codes in MAPPING_CORSE.items()
        for code in codes
        }
    
    # Normalize 'dep' and 'com' columns by removing trailing ".0"
    df['dep'] = df['dep'].astype(str).str.replace(".0", "", regex=False)
    df['com'] = df['com'].astype(str).str.replace(".0", "", regex=False)

    # Extraction des 3 derniers chiffres du code de la commune
    code3 = df['com'].str[-3:]

    # Recherche du nouveau code de département potentiel à partir du code de la commune
    new_dep = code3.map(dict_inverse)

    # Remplacement du code département "20" par le nouveau code si la commune existe dans le dictionnaire
    mask = (df['dep'] == "20") & new_dep.notna()
    df.loc[mask, 'dep'] = new_dep[mask]
    
    return df

def clean_road_data(clean_adr):
    # Initial cleaning
    clean_adr = clean_adr.astype(str).str.lower().str.strip()
    clean_adr = clean_adr.str.encode('latin1').str.decode('utf-8', errors='ignore').fillna("")

    # Remove street numbers
    clean_adr = clean_adr.str.replace(r'^\d+\s*(bis|ter|quater)?\s*', '', regex=True)

    # Remove narrative noise and technical details
    clean_adr = clean_adr.apply(lambda x: re.split(r' - |\(| sur | venant | vta | vt ', x)[0].strip())

    # Normalize characters
    clean_adr = clean_adr.apply(lambda x: unidecode.unidecode(x))

    # Replace common abbreviations and errors
    replacements = {
        r'libration': 'libération',
        r'rpublique': 'république',
        r'prsident': 'président',
        r'marchal': 'maréchal',
        r'flix': 'félix',
        r'franois': 'françois',
        r'clmenceau': 'clémenceau',
        r'ambars|d’ambares': "d'ambarès",
        r'gravires': 'gravières',
        r'\bave\b': 'avenue',
        r'\brte\b': 'route',
        r'\brn\b': 'route nationale',
        r'\bav\b': 'avenue',
        r'\bst\b': 'saint',
        r'\b(rn|n|route nationale)\s*230\b': 'route nationale 230',
        r'\b(a|autoroute)\s*630\b': 'autoroute a630',
        r'\b(autoroute\s+a\s*10|a\s*10)\b': 'autoroute a10',
        "autoroute autoroute": "autoroute",
        r"d'd'": "d'",
        r"accs": "acces",
        r"cte de": "cote de",
        r"lon blum": "leon blum",
        r"andr ricard": "andre ricard",
        r" -rd \d+": "",
        r"place de leglise": "place de l'eglise",
        "ctre cial": "centre commercial",
        "ccial": "centre commercial",
    }

    # Simplify major axes
    clean_adr = clean_adr.str.replace(r'.*(route nationale 230|autoroute a630|autoroute a10).*', r'\1', regex=True)

    for pattern, replacement in replacements.items():
        clean_adr = clean_adr.str.replace(pattern, replacement, regex=True)

    # Remove empty lines and process intersections
    clean_adr = clean_adr[(clean_adr != "") & (clean_adr != "sur parking")]
    clean_adr = clean_adr.apply(lambda x: x.split(' / ')[0].strip())

    return clean_adr

def clean_characteristics(df,out_path):
    print("    -> Initial shape of caracteristiques dataframe:", df.shape)
    print("    -> Columns in the dataframe:", df.columns.tolist())

    # Harmonize Num_Acc (2022) and Accident_Id (other years)
    mask = df["Accident_Id"].notna()
    df.loc[mask, 'Num_Acc'] = df.loc[mask, 'Accident_Id']

    # Normalize time variable (hrmn)
    df['hrmn'] = df['hrmn'].apply(clean_hours)

    # Harmonize years
    df['an'] = df['an'].apply(clean_year)

    # Harmonize address variable (adr)
    #df['adr'] = clean_road_data(df['adr'])
    
    # Clean and standardize department codes
    df['dep'] =  df['dep'].apply(clean_department)

    # Fix error in com INSEE Code
    df['com'] = df['com'].astype(str)
    df['dep'] = df['dep'].astype(str)
    df.loc[(df['com'].astype(str) == "38411") & (df['dep'].astype(str) == "69"), 'com'] = "69287"
    df.loc[(df['com'].astype(str) == "78469") & (df['dep'].astype(str) == "91"), 'com'] = "91390"
    df.loc[(df['com'].astype(str) == "75075") & (df['dep'].astype(str) == "92"), 'com'] = "92075"
    df['com'] = df.apply(lambda row: clean_commune_code(row['com'], row['dep']), axis=1)
    df = handle_corse_codes(df)

    # Replace NaN and sentinel values with designated 'other' codes
    df['int'] = df['int'].fillna(9)
    df['int'] = df['int'].replace(-1, 9)
    df['int'].value_counts(dropna=False)

    df['atm'] = df['atm'].fillna(9)
    df['atm'] = df['atm'].replace(-1, 9)
    df['atm'].value_counts(dropna=False)

    df['col'] = df['col'].fillna(6)
    df['col'] = df['col'].replace(-1, 6)

    # Add weekday
    df['week_day'] = pd.to_datetime(df[['an', 'mois', 'jour']].rename(columns={'an': 'year', 'mois': 'month', 'jour': 'day'})).dt.dayofweek

    # Extract hour of day from 'hrmn'
    df['Hours'] = df['hrmn'].dt.hour

    # Fix latitude/longitude format (comma -> dot)
    df['lat'] = df['lat'].str.replace(",", ".", regex=True).astype(float)
    df['long'] = df['long'].str.replace(",", ".", regex=True).astype(float)

    # Handle light level missing values by hour (impute with hourly median)
    median_by_hour = df.groupby('Hours')['lum'].median()
    df['lum'] = df['lum'].fillna(df['Hours'].map(median_by_hour))

    # Drop columns with more than 10% missing values (also drop 'adr')
    percent = (df.isna().sum() / len(df)) * 100
    cols_to_drop = list(percent[percent > 10].index)
    cols_to_drop.append('adr')  # also remove 'adr'
    df = drop_columns(df, cols_to_drop, logger, "caracteristiques.csv")

    # Réorganisation des colonnes et sauvegarde du fichier nettoyé
    df.to_csv(os.path.join(out_path, "caracteristiques.csv"), index=False)
    print("    -> Cleaned 'caracteristiques' data saved to:", os.path.join(out_path, "caracteristiques.csv"))

    return df