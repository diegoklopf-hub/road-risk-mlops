from collections import Counter
import re
import numpy as np
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


# ---------------------------------------------------------------------------
# Compiled regex (compiled once, not per call)
# ---------------------------------------------------------------------------
_PATTERN = re.compile(r'([A-Za-z0-9-]+)\((\d+)\)')


def _parse_single(s: str) -> dict | None:
    """
    Parse one aggregated string like 'catv_17(1)_7(2)' into a dict of
    normalized proportions: {'17': 0.333, '7': 0.667}.

    Returns None on parse errors (same semantics as the original empty Series).
    """
    if not isinstance(s, str) or s == "NONE":
        return None

    s_clean = s.replace(" ", "")
    matches = _PATTERN.findall(s_clean)

    if not matches:
        return None

    # Integrity check: reconstructed vs original (after first '_')
    reconstructed = "_".join(f"{name}({num})" for name, num in matches)
    parts = s_clean.split('_', 1)
    if len(parts) < 2 or reconstructed != parts[1]:
        return None

    if len(matches) == 1:
        return {matches[0][0]: 1.0}

    numbers = [int(num) for _, num in matches]
    total = sum(numbers)
    if total == 0:
        return {name: 0.0 for name, _ in matches}
    return {name: int(num) / total for name, num in matches}


def expand_column_vectorized(series: pd.Series, col_prefix: str) -> pd.DataFrame:
    """
    Vectorized replacement for:
        series.apply(extract_normalize).add_prefix(col_prefix + "_")

    Instead of creating N pd.Series objects, we:
    1. Parse all strings into lightweight dicts (list comprehension)
    2. Build one DataFrame from the list of dicts (single allocation)
    3. Fill NaN with 0 and add prefix

    Parameters
    ----------
    series : pd.Series
        Column of aggregated strings (e.g. 'catv_17(1)_7(2)').
    col_prefix : str
        Prefix for the resulting column names.

    Returns
    -------
    pd.DataFrame
        One column per unique category, values are normalized proportions.
    """
    # Step 1: Parse all rows into dicts — no pd.Series created per row
    parsed = [_parse_single(s) for s in series.values]

    # Step 2: Build DataFrame in one shot from list of dicts
    # pd.DataFrame(list_of_dicts) is highly optimized internally
    expanded = pd.DataFrame(parsed, index=series.index)

    # Step 3: Fill NaN (categories absent for a given row) with 0
    expanded = expanded.fillna(0.0)

    # Step 4: Prefix columns
    expanded.columns = [f"{col_prefix}_{c}" for c in expanded.columns]

    return expanded


# ---------------------------------------------------------------------------
# Keep original extract_normalize for backward compatibility if needed
# ---------------------------------------------------------------------------
def extract_normalize(s=None):
    """
    Original function kept for backward compatibility.
    Prefer expand_column_vectorized() for batch processing.
    """
    result = _parse_single(s)
    if result is None:
        return pd.Series()
    return pd.Series(result)