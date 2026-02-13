import sys
import subprocess
from pathlib import Path

# Trouve la racine du projet (le dossier qui contient "src/")
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common_utils import reset_status_file
from src.config import STATUS_FILE

modules = [
    "src.pipeline.01_data_import",
    "src.pipeline.02_data_clean",
    "src.pipeline.03_merge",
    "src.pipeline.04_encodage",
    "src.pipeline.05_data_transformation",
    "src.pipeline.06_resampling",
    "src.pipeline.07_model_trainer",
    "src.pipeline.08_model_evaluation",
]

print("RUNNING:", __file__)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("Python used  =", sys.executable)

reset_status_file(STATUS_FILE)

for mod in modules:
    print(f"\n--- Exécution du module {mod} ---")
    subprocess.run(
        [sys.executable, "-m", mod],
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    print(f"✅ {mod} exécuté avec succès.")
