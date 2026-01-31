from collections import Counter
import re
import pandas as pd
from src.custom_logger import logger

def agg_cat_unique_with_count(cat_series, col_name=None):
    vals = cat_series.dropna().astype(str)
    if len(vals) == 0:
        return "NONE"
    counter = Counter(vals)
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    joined_vals = "_".join(f"{modalite}({nb})" for modalite, nb in sorted_items)
    return f"{col_name}_{joined_vals}" if col_name else joined_vals

def categorize_accident(senc_series):
    """
    Categorize collision type based on 'senc' distribution:
    0 - Sens opposé
    1 - Même sens
    2 - Un seul véhicule
    3 - Absence de repère
    4 - Inconnu
    """
    # Count occurrences of each 'senc' value
    counts = senc_series.value_counts()
    count_n1 = counts.get(-1.0, 0) #-1 – Non renseigné
    count_0 = counts.get(0.0, 0)   # 0 – Inconnu
    count_1 = counts.get(1.0, 0)   # 1 – PK ou PR ou numéro d’adresse postale croissant
    count_2 = counts.get(2.0, 0)   # 2 – PK ou PR ou numéro d’adresse postale décroissant
    count_3 = counts.get(3.0, 0)   # 3 – Absence de repère
    # Sens opposé : au moins un véhicule en 1.0 et un en 2.0
    if (count_1 > 0) and (count_2 > 0):
        return 0
    # Même sens : plusieurs véhicules dans le même sens (1.0 ou 2.0)
    elif (count_1 > 1) or (count_2 > 1):
        return 1
    # Un seul véhicule impliqué
    elif senc_series.shape[0] == 1:
        return 2
    # Plusieurs véhicules avec senc=0.0 ou 3.0 ou -1
    elif (count_0 > 0) or (count_3 > 0) or (count_n1 > 0):
        return 3
    # Autres cas
    else:
        return 4

def categorize_gender(sexe_series):
    """
    Compute a weighted mean summarizing the distribution of the 'sexe' column.
    Weights:
    - 0 for male (1.0)
    - 0.5 for unknown (0.0)
    - 1 for female (2.0)

    Args:
        sexe_series (pd.Series): Series with values (0.0: Unknown, 1.0: Male, 2.0: Female).

    Returns:
        float: Weighted score between 0 and 1; higher means more female representation.
    """
    counts = sexe_series.value_counts()
    count_0 = counts.get(0.0, 0)   # 0 – Inconnu
    count_1 = counts.get(1.0, 0)   # 1 – Homme
    count_2 = counts.get(2.0, 0)   # 2 – Femme

    return (count_1 * 0 + count_0 * 0.5 + count_2 * 1) / len(sexe_series)


def extract_normalize(s=None):
    """
    Parse and normalize a string containing one or more tokens of the form
    NAME(NUMBER) or NAME(NUMBER)_NAME(NUMBER)_... and return a pandas.Series
    with normalized numeric proportions per NAME.

    Examples:
        'catv_17(1)_7(2)' -> {'catv17': 0.333, 'catv7': 0.667}

    Parameters
    ----------
    s : str or None
        Input string to parse.

    Returns
    -------
    pandas.Series
        Series of normalized proportions, empty Series on parse errors.
    """
    # Regular expression to capture name(number) pairs
    try:
        s = s.replace(" ", "")
        matches = re.findall(r'([A-Za-z0-9-]+)\((\d+)\)', s)
    except re.error as e:
        logger.warning(f"Regex validation error: {e}")
        return pd.Series()

    # No matches -> return empty Series
    if not matches:
        print(f"No 'name(number)' pattern detected in: '{s}'")
        return pd.Series()

    # Reconstruct string from matches to ensure full coverage
    reconstructed = "_".join([f"{name}({num})" for name, num in matches])
    # Compare with the original substring (after first underscore) used in this dataset
    if f"{reconstructed}" != s.split('_', 1)[1]:
        print(f"The string '{s}' contains uncaptured characters.")
        print(f"=>  '{reconstructed}'")
        print(f"=> '{s.split('_', 1)[1]}'")
        return pd.Series()

    # Single match -> return series with value 1
    if len(matches) == 1:
        name, _ = matches[0]
        return pd.Series({name: 1})

    # Multiple matches -> normalize numeric parts
    if len(matches) > 1:
        try:
            numbers = [int(num) for _, num in matches]
            total = sum(numbers)
            result = {}
            for name, num in matches:
                result[name] = int(num) / total if total != 0 else 0
            return pd.Series(result)
        except ValueError:
            logger.warning(f"Unable to convert '{num}' to int in '{s}'")
            return pd.Series()