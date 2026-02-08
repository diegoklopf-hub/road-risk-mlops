import os
import logging
import pandas as pd
from src.data_processing.check_structure import check_columns, check_schema
from src.data_processing.engineering.aggregation_functions import agg_cat_unique_with_count, categorize_accident, categorize_gender, extract_normalize
from src.data_processing.engineering.security import user_safety_score
from src.entity import DataMergeConfig
from src.custom_logger import logger


def aggregate_accident_features(df,aggregate_features_ref,status_file):
    # --- Groupby principal ---
        aggregate_columns = {
            "age": ["mean", "min", "max"],
            "securite_usager": ["mean", "min"],
            "senc" : categorize_accident, 
            "place": lambda x: agg_cat_unique_with_count(x, col_name="place"),
            "sexe" : categorize_gender,
            "trajet": lambda x: agg_cat_unique_with_count(x, col_name="trajet"),
            "locp": lambda x: agg_cat_unique_with_count(x, col_name="locp"),
            "actp": lambda x: agg_cat_unique_with_count(x, col_name="actp"),
            "etatp": lambda x: agg_cat_unique_with_count(x, col_name="etatp"),
            "catv_cluster": lambda x: agg_cat_unique_with_count(x, col_name="catv"),
            "obs": lambda x: agg_cat_unique_with_count(x, col_name="obs"),
            "obsm": lambda x: agg_cat_unique_with_count(x, col_name="obsm"),
            "choc": lambda x: agg_cat_unique_with_count(x, col_name="choc"),
            "manv": lambda x: agg_cat_unique_with_count(x, col_name="manv"),
            "motor": lambda x: agg_cat_unique_with_count(x, col_name="motor"),
        }
        
        # --- Verify columns ---
        is_shema_valid = check_columns(aggregate_columns.keys(),aggregate_features_ref,status_file,"AGGREGATE")

        if (is_shema_valid)== False:
            error_msg = f"Export failed: Verification of columns to aggregate FAILED"
            logger.error(error_msg)
            raise ValueError(error_msg)

        print("Grouping dataframe by Num_Acc...")
        acc_agg = df.groupby("Num_Acc").agg(aggregate_columns)

        # --- Flatten multi-index columns ---
        acc_agg.columns = [
            "_".join(col) if isinstance(col, tuple) else col
            for col in acc_agg.columns
        ]

        # Adjust representation to remove trailing .0 in aggregated strings (e.g. 1.0_3.0 -> 1_3)
        cols_to_expand = ['place_<lambda>',
                'trajet_<lambda>', 
                'locp_<lambda>', 
                'actp_<lambda>', 
                'etatp_<lambda>', 
                'catv_cluster_<lambda>', 
                'obs_<lambda>', 
                'obsm_<lambda>', 
                'choc_<lambda>', 
                'manv_<lambda>', 
                'motor_<lambda>'] 

        # Remove '.0' from strings
        for col in cols_to_expand: 
            acc_agg[col] = acc_agg[col].str.replace(r'\.0', '', regex=True)

        # --- Expand aggregated columns into proportion columns ---
        expanded_frames = []  # will store new DataFrames for each expanded column

        for col in cols_to_expand:

            print(f"    .... Processing column: {col}")

            # apply extract_normalize() row-wise -> returns a Series per row
            expanded = acc_agg[col].apply(extract_normalize)

            # prefix the new columns with the original column name
            expanded = expanded.add_prefix(f"{col}_")

            expanded_frames.append(expanded)

        # --- Concatenate all expanded frames ---
        expanded_df = pd.concat(expanded_frames, axis=1)

        # --- Add expanded proportions back to the main DataFrame ---
        acc_agg = pd.concat([acc_agg, expanded_df], axis=1)

        # --- Drop original aggregated columns ---
        acc_agg = acc_agg.drop(columns=cols_to_expand)

        # Many NaNs were created where proportion is zero; replace them with zeros
        cols_to_exclude = ['age_mean', 'age_min', 'age_max', 'senc_categoriser_accident']
        acc_agg[acc_agg.columns.difference(cols_to_exclude)] = acc_agg[acc_agg.columns.difference(cols_to_exclude)].fillna(0)

        # Shorten column names by removing suffixes introduced by aggregation functions
        acc_agg.columns = (
            acc_agg.columns.str.replace(r'_\<lambda\>', '', regex=True).str.replace(r'_categorize_accident', '', regex=True).str.replace(r'_categorize_gender', '', regex=True)
        )

        # nb_usagers based on the number of rows in the users table
        acc_agg["nb_usagers"] = df.groupby("Num_Acc").size()

        # --- Add number of unique vehicles per accident ---
        acc_agg["nb_vehicules"] = df.groupby("Num_Acc")["id_vehicule"].nunique()

        return acc_agg



class DataMerge:
    def __init__(self, config: DataMergeConfig):
        self.config = config
        self.df = None
        self.columns_not_aggregated = []
        self.columns_aggregated = []

    def merge_by_usager(self):
        """
        Merge multiple accident-related datasets by user (usager) and create an aggregated dataframe.
        This method combines data from four CSV files (caracteristiques, vehicules, usagers, lieux)
        through a series of left joins on the accident identifier (Num_Acc). It handles the hierarchical
        nature of the data where accidents can have multiple vehicles and users per vehicle.
        Process:
        1. Creates output directory if it doesn't exist
        2. Loads four cleaned CSV files: caracteristiques, vehicules, usagers, and lieux
        3. Creates severity count columns (count_indemne, count_blesse_leger, count_blesse_hosp, count_tue)
        4. Removes rows with missing values
        Returns:
            pd.DataFrame: Merged dataframe with all accident, vehicle, and user information denormalized
                         at the user level, including aggregated severity counts per accident.
        """

        print("""------------- 01 Starting data merging -------------""")
        logger.info("Merge multiple accident-related datasets by user (usager)")
        if os.path.exists(self.config.out_merged_data_relative_path) == False :
            os.makedirs(self.config.out_merged_data_relative_path)
        
        #Read cleaned data
        caracteristiques = pd.read_csv(self.config.input_data_relative_path / "caracteristiques.csv")
        vehicules = pd.read_csv(self.config.input_data_relative_path / "vehicules.csv")
        usagers = pd.read_csv(self.config.input_data_relative_path / "usagers.csv")
        lieux = pd.read_csv(self.config.input_data_relative_path / "lieux.csv")

        
        # Initialize list of non-aggregated columns
        self.columns_not_aggregated = []
        self.columns_not_aggregated.extend(caracteristiques.columns.tolist())
        self.columns_not_aggregated.extend(lieux.columns.tolist())

        # Initialize list of columns that will be aggregated
        self.columns_aggregated.extend(usagers.columns.tolist())
        self.columns_aggregated.extend(vehicules.columns.tolist())
        self.columns_aggregated = [col for col in self.columns_aggregated if col != "Num_Acc"]

        # First merge between 'lieux' and 'caracteristiques' (one row per accident)
        self.df = pd.merge(left = lieux, right = caracteristiques, left_on = 'Num_Acc', right_on = 'Num_Acc')

        # Then merge vehicles table; this duplicates rows because vehicles are one per vehicle
        self.df = pd.merge(left = self.df, right = vehicules, left_on = 'Num_Acc', right_on = 'Num_Acc')
        
        # Finally merge users table; this further expands rows for multiple users per vehicle
        self.df = pd.merge(left = self.df, right = usagers, left_on = ['Num_Acc', 'id_vehicule', 'num_veh'], right_on = ['Num_Acc', 'id_vehicule', 'num_veh'])

        # Drop 'catu' column because its information is redundant with 'place'
        self.df = self.df.drop('catu', axis = 1)
        self.columns_aggregated.remove('catu')
        msg = f"Drop the column {['catu']}"
        print(msg)
        logger.info(msg)    

        # Add 4 columns counting each severity category per accident
        counts = self.df.groupby("Num_Acc")["grav"].value_counts().unstack().fillna(0)
        grav_mapping = {
            0: "count_indemne",
            1: "count_blesse_leger",
            2: "count_blesse_hosp",
            3: "count_tue"
        }
        counts = counts.rename(columns=grav_mapping, errors="raise")
        counts = counts.reset_index()
        self.df = self.df.merge(counts, on="Num_Acc", how="left")
        print("Removing rows with NA: total NA count=", self.df.isna().sum().sum())
        print("NA per column:", self.df.isna().sum()[self.df.isna().sum()>0])
        self.df = self.df.dropna()
        self.columns_not_aggregated.extend(grav_mapping.values())
        self.columns_aggregated.remove('grav')

        # Create an extra cluster for pedestrians where place == 10
        self.df.loc[self.df.place == 10, 'catv_cluster'] = 8

        logger.info('csv merged by usager successfully')        

        return self.df 

    def feature_engineering(self):
        """
        Perform feature engineering on the dataframe.
        This method creates a new 'securite_usager' feature by applying the securite_usager
        function to each row of the dataframe. It then removes the 'secu_merged' column that
        was used in the feature creation and updates the internal column tracking lists.
        Returns
        -------
        pd.DataFrame
            The dataframe with the new 'securite_usager' feature and 'secu_merged' column removed.
        """

        print("""------------- 02 Starting feature engineering -------------""")
        logger.info("Perform feature engineering on the dataframe")
        self.df["securite_usager"] = self.df.apply(user_safety_score, axis=1)
        self.df = self.df.drop(columns=["secu_merged"]) 
        self.columns_aggregated.remove('secu_merged')
        self.columns_aggregated.extend(['securite_usager'])
        msg = f"Drop the column {list(["secu_merged"])}"
        print(msg)
        logger.info(msg)    
        logger.info('feature engineering completed successfully')   
        return self.df
    
    def merge_by_accident(self):
        """
        Merge and aggregate data by accident identifier.
        This method performs the following operations:
        1. Removes 'id_vehicule' and 'num_veh' from the columns to be aggregated
        2. Aggregates accident features using the configured aggregation method
        3. Merges aggregated data with non-aggregated columns on 'Num_Acc'
        """
        print("""------------- 03 Starting data merge by accident -------------""")
        logger.info("Merge and aggregate data by accident identifier")

        # Remove unnecessary columns from the list of columns to aggregate
        self.columns_aggregated = [col for col in self.columns_aggregated if col != 'id_vehicule' and col != 'num_veh']

        # Aggregating data by accident
        acc_agg = aggregate_accident_features(self.df,self.columns_aggregated,self.config.status_file).reset_index()
        self.df = self.df.reset_index(drop=True)  

        # Final merge with not aggregated data
        unique_not_aggregated_cols = list(set(self.columns_not_aggregated))
        self.df = acc_agg.merge(self.df[unique_not_aggregated_cols], on='Num_Acc', how='left')
        self.df = self.df.drop_duplicates()
        logger.info('data aggregated by accident successfully')


    def validate_data_and_export(self):

        logger.info("04 - Validate & export")

        schema = self.config.all_schema

        if "COLUMNS" not in schema:
            raise ValueError("Schema must contain a COLUMNS section")

        columns_schema = schema["COLUMNS"]
        expected_cols = list(columns_schema.keys())

        self.df = self.df.reindex(columns=expected_cols)

        is_schema_valid = check_schema(
            set(self.df.columns),
            columns_schema,
            self.config.status_file,
            "DATA MERGE",
            ignore_calib=True,
        )


        if not is_schema_valid:
            raise ValueError("Schema validation failed")

        output_file = os.path.join(
            self.config.out_merged_data_relative_path,
            "merged_data.csv",
        )
        self.df.to_csv(output_file, index=False)
        logger.info(f"Exported to {output_file}")

        return True
