from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass(frozen=True)
class DataCleanConfig:
    """Configuration pour le nettoyage des données brutes BAAC."""
    raw_data_relative_path: Path
    out_data_relative_path: Path
    from_year: int
    to_year: int
    cluster_cat_vehicule: bool

@dataclass(frozen=True)
class DataImportConfig:
    """Configuration pour le téléchargement et l'import initial."""
    raw_data_relative_path: Path
    from_year: int
    to_year: int
    csv_files: List[str]
    source_url: str

@dataclass(frozen=True)
class DataMergeConfig:
    """Configuration pour la fusion des fichiers (Carac, Lieux, Veh, Usagers)."""
    input_data_relative_path: Path
    out_merged_data_relative_path: Path
    all_schema: Dict[str, Any]
    STATUS_FILE: Path

@dataclass(frozen=True)
class DataEncodeConfig:
    """Configuration pour l'encodage des variables """
    merged_data_path:  Path
    merged_data_encoded_path: Path
    encode_columns: List[str]
    schema: Dict[str, Any]
    STATUS_FILE: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    input_path: Path
    train_test_path: Path
    schema: Dict[str, Any]
    STATUS_FILE: Path

# @dataclass(frozen=True)
# class ModelTrainerConfig:
#     root_dir: Path
#     X_train_path: Path
#     y_train_path: Path
#     X_test_path: Path
#     y_test_path: Path
#     model_name: str
#     alpha: float
#     l1_ratio: float

# @dataclass(frozen=True)
# class ModelEvaluationConfig:
#     root_dir: Path
#     X_test_path: Path
#     y_test_path: Path
#     model_path: Path
#     metric_file_name: Path
#     all_params: dict
#     metric_file_name: Path
#     mlflow_uri: str