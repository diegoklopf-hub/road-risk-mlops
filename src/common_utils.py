import os
from box.exceptions import BoxValueError
import yaml
from src.custom_logger import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def reset_status_file(path: Path):
    """Clear the status file at pipeline start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    logger.info(f"status file reset: {path}")


@ensure_annotations
def append_status(path: Path, phase: str, ok: bool, details: str | None = None):
    """Append a schema/structure check result to the status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{phase}: Validation status: {ok}\n")
        if details:
            f.write(f"{details}\n")
    logger.info(f"status updated: {phase} -> {ok}")


@ensure_annotations
def is_last_status_ok(path: Path) -> bool:
    """Return True when the last recorded status is OK or file is empty/missing."""
    if not path.exists():
        return True

    content = path.read_text().strip().splitlines()
    if not content:
        return True

    for line in reversed(content):
        if "Validation status:" in line:
            return line.split("Validation status:", 1)[1].strip().lower() == "true"

    return True



