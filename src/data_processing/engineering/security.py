
"""User safety utilities.

This module maps vehicle clusters to expected safety equipment and
provides a function to compute a simple compliance score per user.
"""

import numpy as np
import pandas as pd


cluster_names = {
    0: "VU 1.5T-3.5T",
    1: "VL",
    2: "Autre",
    3: "Moto >125cm3",
    4: "Bicyclette",
    5: "Scooter <50cm3",
    6: "Scooter 50-125cm3",
    7: "DP moteur",
    8: "Piéton",
}

equipement_dict = {
    -1: "Non renseigné",
    0: "Aucun équipement",
    1: "Ceinture",
    2: "Casque",
    3: "Dispositif enfants",
    4: "Gilet réfléchissant",
    5: "Airbag (2RM/3RM)",
    6: "Gants (2RM/3RM)",
    7: "Gants + Airbag (2RM/3RM)",
    8: "Non déterminable",
    9: "Autre"
}

# Recommended/required equipment per vehicle cluster.
# Keys use the original French names to avoid changing other code.
equipements_par_catv = {
    0: {  # VU 1.5T-3.5T
        "obligatoires": [1],    # Ceinture
        "conseillés": [4, 3],   # Gilet réfléchissant, Dispositif enfants
    },
    1: {  # VL
        "obligatoires": [1],    # Ceinture
        "conseillés": [4, 3],   # Gilet réfléchissant, Dispositif enfants
    },
    2: {  # autre
        "obligatoires": [],    # Aucun
        "conseillés": [],      # Aucun 
    },
    3: {  # Moto >125cm3
        "obligatoires": [2, 6], # Casque, Gants certifiés CE
        "conseillés": [5, 9],   # Airbag (2RM/3RM), "Autre"
    },
    4: {  # Bicyclette
        "obligatoires": [],    # Aucun équipement obligatoire
        "conseillés": [2, 4],   # Casque, Gilet réfléchissant
    },
    5: {  # Scooter <50cm3
        "obligatoires": [2],    # Casque
        "conseillés": [6, 4, 9],# Gants certifiés CE, Gilet réfléchissant, "Autre"
    },
    6: {  # Scooter 50-125cm3
        "obligatoires": [2, 6], # Casque, Gants certifiés CE
        "conseillés": [5, 9],   # Airbag (2RM/3RM), "Autre"
    },
    7: {  # EDP moteur
        "obligatoires": [],    
        "conseillés": [2, 6, 4, 9],# Casque, Gants certifiés CE, Gilet réfléchissant, "Autre"
    },
    8: {  # Piéton
        "obligatoires": [],    # Aucun
        "conseillés": [],      # Aucun 
    },
}

def format_log_message(cluster_id, code_set,eqt_obligatoires):
    """
    Format a log message for vehicle equipment compliance information.
    Args:
        cluster_id (int): The identifier for the vehicle cluster.
        code_set (set or iterable): A set of equipment codes (as strings) to be formatted.
        eqt_obligatoires (str or int): The mandatory equipment requirement identifier or description.
    Returns:
        str: A formatted log message containing the cluster name, sorted equipment codes
                with their descriptions, and mandatory equipment requirements.
                Format: "Cluster Name => [code (description), ...] Equipements obligatoires: value]"
    Raises:
        None explicitly, but handles ValueError for invalid code conversion gracefully.
    Note:
        - Equipment codes are looked up in the global 'equipement_dict' dictionary.
        - Cluster names are looked up in the global 'cluster_names' dictionary.
        - Invalid (non-numeric) codes are flagged with "? Val invalide" description.
        - Unknown codes default to "Inconnu" (Unknown) description.
    """
    # Vehicle cluster name
    c_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    # Sort codes and lookup descriptions
    descriptions = []
    for code_str in sorted(code_set):
        try:
            code_int = int(code_str)
            desc = equipement_dict.get(code_int, "Inconnu")
            descriptions.append(f"{code_str} ({desc})")
        except ValueError:
            # Cas où le code n'est pas un chiffre
            descriptions.append(f"{code_str} (? Val invalide)")
            
    return f"{c_name} => [{', '.join(descriptions)}] Equipements obligatoires: {eqt_obligatoires}]"

def user_safety_score(row):
    """
    Compute a user safety score between 0.0 and 1.0.

    - 0.0 = not all required equipment present
    - 0.8 = all required equipment present, not all recommended
    - 1.0 = all required and recommended equipment present

    Variables used:
    - secu_merged : concatenation of equipment codes (e.g. '18', '256', ...)
    - catv_cluster : vehicle cluster id (0..8)
    """
    cluster = row["catv_cluster"]
    secu_val = row["secu_merged"]

    # --- Normalize security value ---
    if pd.isna(secu_val):
        return np.nan

    if isinstance(secu_val, (int, float)) and not isinstance(secu_val, bool):
        secu_str = str(int(secu_val))   # ex : 18.0 -> '18'
    else:
        secu_str = str(secu_val)

    # Remove duplicate characters while preserving order and build a set
    secu_str = ''.join(sorted(set(secu_str), key=secu_str.index))
    secu_set = set(secu_str) 
    # --- Replace code 7 with {5, 6} for verification ---
    if '7' in secu_set:
        secu_set.remove('7')
        secu_set.update({'5', '6'})

    codes_presents = {int(c) for c in secu_set if c.isdigit()}

    if ('8' == secu_set) | ('2' == secu_set) :
        return 0.8 # Default: assume required equipment present
 
    # --- Get required and recommended equipment lists ---
    obligatoires = set(equipements_par_catv[cluster]["obligatoires"])
    conseilles = set(equipements_par_catv[cluster]["conseillés"])

    # --- Score computation logic ---

    # 1. Check if ALL required equipment is present
    # If any required item is missing, return 0.0 immediately
    if not obligatoires.issubset(codes_presents):
        return 0.0

    # 2. All required equipment present -> check recommended items
    if conseilles.issubset(codes_presents):
        return 1.0
    
    # 3. Required present but not all recommended -> partial score
    return 0.8


