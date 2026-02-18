import os
from src.data_processing.check_structure import drop_columns
from src.custom_logger import logger

def clean_vehicles(df,out_path,cluster_cat_vehicule):
    """Clean the `vehicules` (vehicles) DataFrame and save to `out_path`.

    Replaces sentinel values, maps vehicle categories to clusters and drops
    columns with excessive missing values. Messages are in English.
    """
    print("    -> Initial shape of vehicles dataframe:", df.shape)
    print("    -> Columns in the dataframe:", df.columns.tolist())

    # Replace -1 with 0 (unknown category) for specified columns
    cols_to_fix = ['senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 'occutc']
    df[cols_to_fix] = df[cols_to_fix].replace(-1, 0)

    # Fill missing 'motor' values using the mode per vehicle category
    df['motor'] = (df['motor'].fillna(df.groupby('catv')['motor'].transform(lambda x: x.mode()[0])))

    inverse_dict = {}
    for cluster, values in cluster_cat_vehicule.items():
        for value in values:
            inverse_dict[value] = cluster
    df['catv_cluster'] = df['catv'].map(inverse_dict)

    # Drop columns with more than 10% missing values
    percent = (df.isna().sum() / len(df)) * 100
    cols_to_drop = percent[percent > 10].index
    cols_to_drop = cols_to_drop.tolist()
    cols_to_drop.extend(['catv'])
    df = drop_columns(df, cols_to_drop, logger, "vehicules.csv")   

    # Save cleaned file
    df.to_csv(os.path.join(out_path, "vehicules.csv"), index=False)
    logger.info("Cleaned 'vehicules' data saved to: %s", os.path.join(out_path, "vehicules.csv"))

    return df
