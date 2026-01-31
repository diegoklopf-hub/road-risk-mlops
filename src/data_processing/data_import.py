import requests
import os
from src.data_processing.check_structure import check_existing_folder
from src.entity import DataImportConfig
from src.custom_logger import logger

"""
Module for importing raw data files from a remote source and saving them locally.
"""

def check_existing_data(file_path):
    """Return True if it's OK to write `file_path`.

    If the file already exists this function prompts the user whether to
    overwrite. Returns True if the file does not exist or the user agrees
    to overwrite; otherwise False.
    """
    if os.path.isfile(file_path):
        while True:
            response = input(f"Files already exist. Do you want to overwrite all existing files? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True

def import_raw_data(raw_data_relative_path, from_year, to_year, csv_files, base_url):
    """Download raw CSV files for the given year range to `raw_data_relative_path`.

    Parameters
    - `raw_data_relative_path`: destination folder for downloaded CSVs.
    - `from_year`, `to_year`: inclusive year range to download.
    - `csv_files`: iterable of metadata dicts describing available files.
    - `base_url`: base URL to build download links from `resource_id`.

    Note: This function logs download status and prompts before overwriting
    existing files.
    """

    # Ensure destination folder exists. (The `check_existing_folder` helper
    # indicates whether the folder already exists; the current call here keeps
    # the existing behavior of creating the folder when needed.)
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)

    # download all the files
    overwrite = False
    for file in csv_files:
        # Only consider files within the requested year range
        if (file['year'] >= from_year) and (file['year'] <= to_year):
            output_file = os.path.join(raw_data_relative_path, file['subgroup'] + '-' + str(file['year']) + '.csv')
            # If user chose to overwrite earlier files, `overwrite` remains True
            if overwrite or check_existing_data(output_file):
                overwrite = True
                download_url = f"{base_url}{file['resource_id']}"
                response = requests.get(download_url, allow_redirects=True)
                logger.info(f'downloading {download_url} as {os.path.basename(output_file)}')
                if response.status_code == 200:
                    # Response content is text; write as UTF-8 encoded bytes
                    content = response.text
                    with open(output_file, "wb") as text_file:
                        text_file.write(content.encode('utf-8'))
                        # Print/log download success in English
                        print(f"Downloaded {file['subgroup']} {file['year']} to {os.path.basename(output_file)} OK:", response.status_code)
                        logger.info(f"Downloaded {file['subgroup']} {file['year']} to {os.path.basename(output_file)}")
                else:
                    # Log failed access. `file['url']` may not always be present
                    # in the metadata; this line keeps the existing behavior.
                    logger.error(f"Error accessing the object {file.get('url', 'unknown')}: {response.status_code}")


class DataImport:
    def __init__(self, config: DataImportConfig):
        self.config = config

    def import_csv(self):
        import_raw_data(self.config.raw_data_relative_path, self.config.from_year, self.config.to_year, self.config.csv_files, self.config.source_url)
        logger.info(f'raw data set imported into {self.config.raw_data_relative_path}')
                
