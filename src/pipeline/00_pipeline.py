import os
from pathlib import Path
import sys

# Add parent directory to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# Liste de tes fichiers dans l'ordre d'exécution
scripts = [
    r".\src\pipeline\01_data_import.py",
    r".\src\pipeline\02_data_clean.py",
    r".\src\pipeline\03_merge.py",
    r".\src\pipeline\04_encodage.py",
    r".\src\pipeline\05_data_transformation.py"
]

for script in scripts:
    print(f"--- Exécution de {script} ---")
    exit_code = os.system(f"python {script}")
    
    # Si un script échoue (code différent de 0), on arrête tout
    if exit_code != 0:
        print(f"Erreur lors de l'exécution de {script}. Arrêt du processus.")
        break