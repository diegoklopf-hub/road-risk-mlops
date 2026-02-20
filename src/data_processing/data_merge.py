import os
import pandas as pd
from src.data_processing.schema_manager import check_columns, SchemaManager
from src.data_processing.engineering.aggregation_functions import (
    agg_cat_unique_with_count,
    categorize_accident,
    categorize_gender,
    expand_column_vectorized,
)
from src.data_processing.engineering.security import user_safety_score
from src.entity import DataMergeConfig
from src.custom_logger import logger


def _compute_all_custom(group):
    """
    Compute all custom aggregations for a single accident group in one pass.
    Returns a pd.Series with all custom features.
    Called once per group instead of 13 times.
    """
    return pd.Series({
        "senc": categorize_accident(group["senc"]),
        "sexe": categorize_gender(group["sexe"]),
        "place": agg_cat_unique_with_count(group["place"], col_name="place"),
        "trajet": agg_cat_unique_with_count(group["trajet"], col_name="trajet"),
        "locp": agg_cat_unique_with_count(group["locp"], col_name="locp"),
        "actp": agg_cat_unique_with_count(group["actp"], col_name="actp"),
        "etatp": agg_cat_unique_with_count(group["etatp"], col_name="etatp"),
        "catv_cluster": agg_cat_unique_with_count(group["catv_cluster"], col_name="catv"),
        "obs": agg_cat_unique_with_count(group["obs"], col_name="obs"),
        "obsm": agg_cat_unique_with_count(group["obsm"], col_name="obsm"),
        "choc": agg_cat_unique_with_count(group["choc"], col_name="choc"),
        "manv": agg_cat_unique_with_count(group["manv"], col_name="manv"),
        "motor": agg_cat_unique_with_count(group["motor"], col_name="motor"),
    })


def aggregate_accident_features(df, aggregate_features_ref, status_file):

        # --- All columns that will be aggregated ---
        all_agg_keys = [
            "age", "securite_usager", "senc", "place", "sexe", "trajet",
            "locp", "actp", "etatp", "catv_cluster", "obs", "obsm",
            "choc", "manv", "motor",
        ]

        # --- Verify columns ---
        is_shema_valid = check_columns(all_agg_keys, aggregate_features_ref, status_file, "AGGREGATE")

        if not is_shema_valid:
                raise ValueError(f"Schema validation failed: See {status_file} for details.")

        logger.info("Aggregating accident features by Num_Acc")
        grouped = df.groupby("Num_Acc")

        # ===================================================================
        # 1. Native aggregations (vectorized, executed in C by pandas)
        # ===================================================================
        logger.info("Grouping dataframe by Num_Acc (native aggregations)...")
        native_agg = {
            "age": ["mean", "min", "max"],
            "securite_usager": ["mean", "min"],
        }
        acc_native = grouped.agg(native_agg)

        # Flatten multi-index columns for native part
        acc_native.columns = ["_".join(col) for col in acc_native.columns]

        # ===================================================================
        # 2. Custom aggregations — single pass over all groups
        # ===================================================================
        logger.info("Grouping dataframe by Num_Acc (custom aggregations — single pass)...")
        acc_custom = grouped.apply(_compute_all_custom, include_groups=False)

        # ===================================================================
        # 3. Merge native + custom
        # ===================================================================
        logger.info("Merging native and custom aggregations...")
        acc_agg = pd.concat([acc_native, acc_custom], axis=1)
        del acc_native, acc_custom

        # Columns to expand (names now come directly from _compute_all_custom keys)
        cols_to_expand = [
            'place', 'trajet', 'locp', 'actp', 'etatp',
            'catv_cluster', 'obs', 'obsm', 'choc', 'manv', 'motor',
        ]

        # Remove '.0' from aggregated strings (e.g. 1.0_3.0 -> 1_3)
        for col in cols_to_expand:
            acc_agg[col] = acc_agg[col].str.replace(r'\.0', '', regex=True)

        # --- Expand aggregated columns into proportion columns (VECTORIZED) ---
        expanded_frames = []

        for col in cols_to_expand:
            logger.info(f"    .... Processing column: {col}")
            expanded = expand_column_vectorized(acc_agg[col], col_prefix=col)
            expanded_frames.append(expanded)

        # --- Concatenate all expanded frames ---
        expanded_df = pd.concat(expanded_frames, axis=1)

        # --- Add expanded proportions back to the main DataFrame ---
        acc_agg = pd.concat([acc_agg, expanded_df], axis=1)
        del expanded_frames, expanded_df

        # --- Drop original aggregated columns ---
        acc_agg = acc_agg.drop(columns=cols_to_expand)

        # Many NaNs were created where proportion is zero; replace them with zeros
        cols_to_exclude = ['age_mean', 'age_min', 'age_max', 'senc']
        acc_agg[acc_agg.columns.difference(cols_to_exclude)] = acc_agg[acc_agg.columns.difference(cols_to_exclude)].fillna(0)

        # nb_usagers based on the number of rows in the users table
        acc_agg["nb_usagers"] = grouped.size()

        # --- Add number of unique vehicles per accident ---
        acc_agg["nb_vehicules"] = grouped["id_vehicule"].nunique()

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

        logger.info("-------------  Starting data merge by user-level records -------------")
        logger.info("Merging cleaned datasets: caracteristiques, lieux, vehicules, usagers")
        if os.path.exists(self.config.out_merged_data_relative_path) == False:
            os.makedirs(self.config.out_merged_data_relative_path)

        # Read cleaned data
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
        self.df = pd.merge(left=lieux, right=caracteristiques, left_on='Num_Acc', right_on='Num_Acc')
        del lieux, caracteristiques

        # Then merge vehicles table; this duplicates rows because vehicles are one per vehicle
        self.df = pd.merge(left=self.df, right=vehicules, left_on='Num_Acc', right_on='Num_Acc')
        del vehicules

        # Finally merge users table; this further expands rows for multiple users per vehicle
        self.df = pd.merge(left=self.df, right=usagers, left_on=['Num_Acc', 'id_vehicule', 'num_veh'], right_on=['Num_Acc', 'id_vehicule', 'num_veh'])
        del usagers

        # Drop 'catu' column because its information is redundant with 'place'
        self.df = self.df.drop('catu', axis=1)
        self.columns_aggregated.remove('catu')
        msg = f"Drop the column {['catu']}"
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
        del counts

        logger.info("Removing rows with NA: total NA count=%s", self.df.isna().sum().sum())
        logger.info("NA per column: %s", self.df.isna().sum()[self.df.isna().sum() > 0])
        self.df = self.df.dropna()
        self.columns_not_aggregated.extend(grav_mapping.values())
        self.columns_aggregated.remove('grav')

        # Create an extra cluster for pedestrians where place == 10
        self.df.loc[self.df.place == 10, 'catv_cluster'] = 8

        logger.info("User-level merged dataset built successfully")

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

        logger.info("------------- Starting feature engineering -------------")
        logger.info("Computing engineered features on merged dataframe")
        self.df["securite_usager"] = self.df.apply(user_safety_score, axis=1)
        self.df = self.df.drop(columns=["secu_merged"])
        self.columns_aggregated.remove('secu_merged')
        self.columns_aggregated.extend(['securite_usager'])
        msg = "Drop the column ['secu_merged']"
        logger.info(msg)
        logger.info("Feature engineering completed successfully")
        return self.df

    def merge_by_accident(self):
        """
        Merge and aggregate data by accident identifier.
        This method performs the following operations:
        1. Removes 'id_vehicule' and 'num_veh' from the columns to be aggregated
        2. Aggregates accident features using the configured aggregation method
        3. Merges aggregated data with non-aggregated columns on 'Num_Acc'
        """
        logger.info("------------- Starting aggregation by accident -------------")
        logger.info("Merge and aggregate data by accident identifier")

        # Remove unnecessary columns from the list of columns to aggregate
        self.columns_aggregated = [col for col in self.columns_aggregated if col != 'id_vehicule' and col != 'num_veh']

        # Aggregating data by accident
        acc_agg = aggregate_accident_features(self.df, self.columns_aggregated, self.config.status_file).reset_index()
        self.df = self.df.reset_index(drop=True)

        # Final merge with not aggregated data
        unique_not_aggregated_cols = list(set(self.columns_not_aggregated))
        self.df = acc_agg.merge(self.df[unique_not_aggregated_cols], on='Num_Acc', how='left')
        self.df = self.df.drop_duplicates(subset=['Num_Acc'])
        logger.info("Data aggregated by accident successfully")


    def validate_data_and_export(self):

        logger.info("Validate & export")
        schema = self.config.all_schema

        is_schema_valid = SchemaManager(schema).check_schema(
            set(self.df.columns),
            self.config.status_file,
            "DATA MERGE",
            ignore_calib=True,
        )

        if not is_schema_valid:
                raise ValueError(f"Schema validation failed: See {self.config.status_file} for details.")
        else:
            output_file = os.path.join(
                self.config.out_merged_data_relative_path,
                "merged_data.csv",
            )
            self.df.to_csv(output_file, index=False)
            logger.info(f"Exported to {output_file}")

            return True