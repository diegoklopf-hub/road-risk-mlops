from pathlib import Path
from typing import Any, Dict, List, Union

from src.config import CONFIG_FILE_PATH, SCHEMA_FILE_PATH, PARAMS_FILE_PATH, STATUS_FILE
from src.common_utils import create_directories, read_yaml
from src.entity import (
    DataImportConfig,
    DataCleanConfig,
    DataMergeConfig,
    DataEncodeConfig,
    DataTransformationConfig,
    DataResamplingConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ShapExplicabilityConfig,
)


def _get(cfg: Any, key: str):
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def _to_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _normalize_cluster_map(cluster_map: Dict[Any, List[int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for k, v in cluster_map.items():
        out[int(k)] = list(v)
    return out


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

    def get_final_schema(self):
        schema_origin = self.schema.COLUMNS
        additional_schema = self.schema.ADDITIONAL_ENCODED_COLUMNS
        remove_col = self.schema.REMOVE_ENCODED_COLUMNS
        schema = {**schema_origin, **additional_schema}
        for col in remove_col:
            schema.pop(col, None)
        return schema

    # ---------------- DATA ---------------- #

    def get_data_import_config(self) -> DataImportConfig:
        cfg = _get(self.config, "data_import")

        return DataImportConfig(
            raw_data_relative_path=_to_path(_get(cfg, "raw_data_relative_path")),
            source_url=_get(cfg, "source_url"),
            from_year=int(_get(cfg, "from_year")),
            to_year=int(_get(cfg, "to_year")),
            resources=list(_get(cfg, "resources")),
        )

    def get_data_clean_config(self) -> DataCleanConfig:
        cfg = _get(self.config, "data_clean")

        cluster = _get(cfg, "cluster_cat_vehicule")
        cluster = _normalize_cluster_map(cluster)

        return DataCleanConfig(
            raw_data_relative_path=_to_path(_get(cfg, "raw_data_relative_path")),
            out_data_relative_path=_to_path(_get(cfg, "out_data_relative_path")),
            cluster_cat_vehicule=cluster,
        )

    def get_data_merge_config(self) -> DataMergeConfig:
        cfg = _get(self.config, "data_merge")
        schema_columns = _get(self.schema, "COLUMNS")

        return DataMergeConfig(
            input_data_relative_path=_to_path(_get(cfg, "input_data_relative_path")),
            out_merged_data_relative_path=_to_path(_get(cfg, "out_merged_data_relative_path")),
            status_file=_to_path(STATUS_FILE),
            all_schema=schema_columns,
        )

    def get_data_encodage_config(self) -> DataEncodeConfig:
        cfg = _get(self.config, "data_encodage")

        schema = self.get_final_schema()

        return DataEncodeConfig(
            merged_data_path=_to_path(_get(cfg, "merged_data_path")),
            merged_data_encoded_path=_to_path(_get(cfg, "merged_data_encoded_path")),
            encode_columns=list(_get(cfg, "encode_columns")),
            model_one_hot_encoder_path=_to_path(_get(cfg, "model_one_hot_encoder_path")),
            status_file=_to_path(STATUS_FILE),
            schema=schema,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = _get(self.config, "data_transformation")

        train_test_path = _to_path(_get(cfg, "train_test_path"))
        create_directories([train_test_path])

        schema = self.get_final_schema()

        return DataTransformationConfig(
            input_path=_to_path(_get(cfg, "input_path")),
            train_test_path=train_test_path,
            status_file=_to_path(STATUS_FILE),
            schema=schema,
        )

    def get_data_resampling_config(self) -> DataResamplingConfig:
        cfg = _get(self.config, "data_resampling")

        output_path = _to_path(_get(cfg, "output_path"))
        create_directories([output_path])

        schema = self.get_final_schema()

        return DataResamplingConfig(
            input_x_path=_to_path(_get(cfg, "input_x_path")),
            input_y_path=_to_path(_get(cfg, "input_y_path")),
            output_path=output_path,
            status_file=_to_path(STATUS_FILE),
            schema=schema,
        )

    # ---------------- MODEL ---------------- #

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = _get(self.config, "model_trainer")

        param_grid: Dict[str, Any] = {}
        if (isinstance(self.params, dict) and "XGBoost" in self.params) or hasattr(self.params, "XGBoost"):
            xgb_params = _get(self.params, "XGBoost")
            if (isinstance(xgb_params, dict) and "param_grid" in xgb_params) or hasattr(xgb_params, "param_grid"):
                param_grid = _get(xgb_params, "param_grid")

        return ModelTrainerConfig(
            X_train_path=_to_path(_get(cfg, "X_train_path")),
            y_train_path=_to_path(_get(cfg, "y_train_path")),
            model_path=_to_path(_get(cfg, "model_path")),
            features_path=_to_path(_get(cfg, "features_path")),
            param_grid=param_grid,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = _get(self.config, "model_evaluation")

        root_dir = _to_path(_get(cfg, "root_dir"))
        create_directories([root_dir])

        return ModelEvaluationConfig(
            root_dir=root_dir,
            X_test_path=_to_path(_get(cfg, "X_test_path")),
            y_test_path=_to_path(_get(cfg, "y_test_path")),
            model_path=_to_path(_get(cfg, "model_path")),
            metric_file_name=_to_path(_get(cfg, "metric_file_name")),
            mlflow_uri=_get(cfg, "mlflow_uri"),
            pushgateway_url=_get(cfg, "pushgateway_url"),
        )
    
    def get_shap_explicability_config(self) -> ShapExplicabilityConfig:
        cfg = _get(self.config, "shap_explicability")

        root_dir = _to_path(_get(cfg, "root_dir"))
        create_directories([root_dir])

        return ShapExplicabilityConfig(
            root_dir=root_dir,
            model_path=_to_path(_get(cfg, "model_path")),
            X_train_path=_to_path(_get(cfg, "X_train_path")),
            shap_explainer_path=_to_path(_get(cfg, "shap_explainer_path")),
            sample_size=int(_get(cfg, "sample_size")),
        )

