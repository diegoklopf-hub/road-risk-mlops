import requests
import os
from src.data_processing.check_structure import check_existing_folder
from src.entity import DataImportConfig
from src.custom_logger import logger
from pathlib import Path
from src.common_utils import append_status
from src.config import STATUS_FILE

"""
Module for importing raw data files from a remote source and saving them locally.
"""

def check_existing_data(output_file):
    if os.path.exists(output_file):
        print("Files exist → auto overwrite enabled")
        return True
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
        os.makedirs(raw_data_relative_path, exist_ok=True)

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
        csv_files = getattr(self.config, "csv_files", None) or self.config.resources
        import_raw_data(
            self.config.raw_data_relative_path,
            self.config.from_year,
            self.config.to_year,
            csv_files,
            self.config.source_url,
    )

    def check_imported_files(self):
        expected_files = []
        for item in self.config.resources:
            if self.config.from_year <= item["year"] <= self.config.to_year:
                expected_files.append(
                    Path(self.config.raw_data_relative_path)
                    / f"{item['subgroup']}-{item['year']}.csv"
                )

        missing = [str(p) for p in expected_files if not p.exists()]
        ok = len(missing) == 0
        details = f"Missing files: {missing}" if missing else None
        append_status(STATUS_FILE, "DATA IMPORT", ok, details)
        if not ok:
            raise FileNotFoundError(details)



                
