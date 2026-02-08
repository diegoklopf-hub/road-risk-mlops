from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# =========================
# DATA IMPORT
# =========================

@dataclass(frozen=True)
class DataImportConfig:
    raw_data_relative_path: Path
    source_url: str
    from_year: int
    to_year: int
    resources: List[Dict[str, Any]]

# =========================
# DATA CLEAN
# =========================

@dataclass(frozen=True)
class DataCleanConfig:
    raw_data_relative_path: Path
    out_data_relative_path: Path
    cluster_cat_vehicule: Dict[int, List[int]]

# =========================
# DATA MERGE
# =========================

@dataclass(frozen=True)
class DataMergeConfig:
    input_data_relative_path: Path
    out_merged_data_relative_path: Path
    status_file: Path
    all_schema: Dict[str, Any]

# =========================
# DATA ENCODAGE
# =========================

@dataclass(frozen=True)
class DataEncodeConfig:
    merged_data_path: Path
    merged_data_encoded_path: Path
    encode_columns: List[str]
    status_file: Path

# =========================
# DATA TRANSFORMATION
# =========================

@dataclass(frozen=True)
class DataTransformationConfig:
    input_path: Path
    train_test_path: Path
    status_file: Path

# =========================
# MODEL TRAINER
# =========================

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path
    y_test_path: Path
    sample_weight_train_path: Path | None
    model_name: str

# =========================
# MODEL EVALUATION
# =========================

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    X_test_path: Path
    y_test_path: Path
    model_path: Path
    metric_file_name: Path
    mlflow_uri: Optional[str] = None
