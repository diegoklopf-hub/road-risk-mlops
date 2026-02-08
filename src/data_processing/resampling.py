import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Project root (pour que "import src...." marche même en python fichier.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.custom_logger import logger

DATA_DIR = Path("data/train_test")

# Paramètres "resampling" (en vrai: weighting)
THRESHOLD = 40.0          
HEAVY_WEIGHT = 3.0    
NORMAL_WEIGHT = 1.0     


def build_sample_weights(y: pd.Series,
                         threshold: float = THRESHOLD,
                         w_pos: float = HEAVY_WEIGHT,
                         w_neg: float = NORMAL_WEIGHT) -> pd.Series:
    """
    Pour une target continue, on fait du "reweighting" :
    - y >= threshold -> poids w_pos
    - y <  threshold -> poids w_neg
    """
    y = pd.Series(y).astype(float)
    w = np.where(y >= threshold, w_pos, w_neg).astype("float32")
    return pd.Series(w, name="sample_weight")


def main() -> None:
    y_train_path = DATA_DIR / "y_train.csv"
    y_test_path  = DATA_DIR / "y_test.csv"

    if not y_train_path.exists():
        raise FileNotFoundError(f"Missing: {y_train_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing: {y_test_path}")

    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # Sanity checks
    logger.info(f"y_train dtype={y_train.dtype} min={float(np.min(y_train)):.2f} max={float(np.max(y_train)):.2f}")
    logger.info(f"y_test  dtype={y_test.dtype}  min={float(np.min(y_test)):.2f}  max={float(np.max(y_test)):.2f}")

    # % de "graves" selon seuil (utile pour voir le déséquilibre)
    train_pos = float((pd.Series(y_train).astype(float) >= THRESHOLD).mean())
    test_pos  = float((pd.Series(y_test).astype(float) >= THRESHOLD).mean())
    logger.info(f"Threshold={THRESHOLD} => Train grave%={train_pos*100:.2f}% | Test grave%={test_pos*100:.2f}%")

    # Build weights
    w_train = build_sample_weights(y_train)
    w_test = build_sample_weights(y_test)

    # Save
    out_train = DATA_DIR / "sample_weight_train.csv"
    out_test = DATA_DIR / "sample_weight_test.csv"
    w_train.to_csv(out_train, index=False)
    w_test.to_csv(out_test, index=False)

    logger.info(f"Saved: {out_train.resolve()}")
    logger.info(f"Saved: {out_test.resolve()}")

    # Petit check distribution poids
    logger.info(f"Weight distribution train:\n{w_train.value_counts().to_string()}")


if __name__ == "__main__":
    main()
