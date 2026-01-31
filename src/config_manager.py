from src.config import CONFIG_FILE_PATH, SCHEMA_FILE_PATH, PARAMS_FILE_PATH
from src.common_utils import create_directories, read_yaml
from src.entity import DataImportConfig, DataCleanConfig, DataMergeConfig, DataEncodeConfig, DataTransformationConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            self.schema = read_yaml(schema_filepath)

            
    def get_data_import_config(self) -> DataImportConfig:
          config = self.config.data_import

          data_import_config = DataImportConfig(
            raw_data_relative_path=config.raw_data_relative_path,
            from_year=config.from_year,
            to_year=config.to_year,
            csv_files=config.resources,
            source_url=config.source_url
          )

          return data_import_config

    def get_data_clean_config(self) -> DataCleanConfig:
          configImport = self.config.data_import
          config = self.config.data_clean

          data_clean_config = DataCleanConfig(
            raw_data_relative_path=config.raw_data_relative_path,
            out_data_relative_path=config.out_data_relative_path,
            from_year=configImport.from_year,
            to_year=configImport.to_year,
            cluster_cat_vehicule=config.cluster_cat_vehicule
          )

          return data_clean_config
    
    def get_data_merge_config(self) -> DataMergeConfig:
          config = self.config.data_merge
          schema = self.schema.COLUMNS

          data_merge_config = DataMergeConfig(
            input_data_relative_path=config.input_data_relative_path,
            out_merged_data_relative_path=config.out_merged_data_relative_path,
            all_schema = schema,
            STATUS_FILE=config.STATUS_FILE
          )

          return data_merge_config
    
    def get_data_encodage_config(self):
          config = self.config.data_encodage
          schema_origin = self.schema.COLUMNS
          additional_schema = self.schema.ADDITIONAL_ENCODED_COLUMNS
          remove_col = self.schema.REMOVE_ENCODED_COLUMNS
          schema = {**schema_origin, **additional_schema}
          for col in remove_col:
                schema.pop(col, None)

          data_encodage_config = DataEncodeConfig(
              merged_data_path=config.merged_data_path,
              merged_data_encoded_path=config.merged_data_encoded_path,
              encode_columns=config.encode_columns,
              schema = schema,
              STATUS_FILE=config.STATUS_FILE
          )

          return data_encodage_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
          config = self.config.data_transformation
          schema_origin = self.schema.COLUMNS
          additional_schema = self.schema.ADDITIONAL_ENCODED_COLUMNS
          remove_col = self.schema.REMOVE_ENCODED_COLUMNS
          schema = {**schema_origin, **additional_schema}
          for col in remove_col:
                schema.pop(col, None)

          create_directories([config.train_test_path])

          data_transformation_config = DataTransformationConfig(
                input_path = config.input_path,
                train_test_path = config.train_test_path,
                schema = schema,
                STATUS_FILE = config.STATUS_FILE
          )

          return data_transformation_config
    
#     def get_model_trainer_config(self) -> ModelTrainerConfig:
#           config = self.config.model_trainer
#           params = self.params.ElasticNet
          
#           create_directories([config.root_dir])

#           model_trainer_config = ModelTrainerConfig(
#                 root_dir = config.root_dir,
#                 X_train_path = config.X_train_path,
#                 y_train_path = config.y_train_path,
#                 X_test_path = config.X_test_path,
#                 y_test_path = config.y_test_path,
#                 model_name = config.model_name,
#                 alpha = params.alpha,
#                 l1_ratio = params.l1_ratio
#           )

#           return model_trainer_config
    
#     def get_model_evaluation_config(self) -> ModelEvaluationConfig:
#           config = self.config.model_evaluation
#           params = self.params.ElasticNet

#           create_directories([config.root_dir])
          
#           model_evaluation_config = ModelEvaluationConfig(
#                 root_dir=config.root_dir,
#                 X_test_path = config.X_test_path,
#                 y_test_path = config.y_test_path,
#                 model_path=config.model_path,
#                 metric_file_name=config.metric_file_name,
#                 all_params=params,
#                 mlflow_uri="https://dagshub.com/licence.pedago/overview_mlops_wine_quality.mlflow",
#           )

#           return model_evaluation_config