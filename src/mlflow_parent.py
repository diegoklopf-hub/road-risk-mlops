"""
Helper to manage the MLflow parent run.
Two modes:
    - Via MLFLOW_PARENT_RUN_ID environment variable (local call with run_pipeline_mlflow.py)
    - Via shared file /app/logs/mlflow_parent_run_id.txt (Airflow call)
"""
import os
import mlflow
from pathlib import Path

PARENT_RUN_FILE = "/app/logs/mlflow_parent_run_id.txt"


def get_or_create_parent_run() -> str:
    """
    1. If MLFLOW_PARENT_RUN_ID is defined (local call) -> use it
    2. Otherwise, if the shared file exists (steps 02-08 via Airflow) -> read it
    3. Otherwise, create a new parent run and write the ID to the file (step 01 via Airflow)
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))
    mlflow.set_experiment("GLOBAL_PIPELINE")

    # Mode 1: environment variable (run_pipeline_mlflow.py)
    env_parent_id = os.getenv("MLFLOW_PARENT_RUN_ID", "").strip()
    if env_parent_id:
        return env_parent_id

    # Mode 2: shared file (Airflow) - read
    if os.path.exists(PARENT_RUN_FILE):
        parent_run_id = Path(PARENT_RUN_FILE).read_text().strip()
        if parent_run_id:
            return parent_run_id

    # Mode 3: first Airflow step - create
    with mlflow.start_run(run_name="GLOBAL_PIPELINE_RUN") as parent_run:
        parent_run_id = parent_run.info.run_id

    Path(PARENT_RUN_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(PARENT_RUN_FILE).write_text(parent_run_id)

    return parent_run_id
