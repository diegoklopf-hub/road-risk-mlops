import mlflow
import subprocess
import os

MLFLOW_TRACKING = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING)
mlflow.set_experiment("GLOBAL_PIPELINE")

STEPS = [
    "01_data_import",
    "02_data_clean",
    "03_merge",
    "04_encodage",
    "05_data_transformation",
    "06_resampling",
    "07_model_trainer",
    "08_model_evaluation"
]

with mlflow.start_run(run_name="GLOBAL_PIPELINE_RUN") as parent:

    parent_id = parent.info.run_id
    print("\n🧠 PARENT RUN:", parent_id)

    for step in STEPS:
        print(f"\n🚀 Running {step}")

        subprocess.run([
            "docker","run","--rm",
            "--network","mlflow_default",

            "-v", f"{os.getcwd()}/data:/app/data",
            "-v", f"{os.getcwd()}/models:/app/models",
            "-v", f"{os.getcwd()}/metrics:/app/metrics",

            # MLflow tracking
            "-e","MLFLOW_TRACKING_URI=http://mlflow_server:5000",
            "-e",f"MLFLOW_PARENT_RUN_ID={parent_id}",

            # MINIO CREDS (OBLIGATOIRE)
            "-e","AWS_ACCESS_KEY_ID=minio",
            "-e","AWS_SECRET_ACCESS_KEY=minio123",
            "-e","MLFLOW_S3_ENDPOINT_URL=http://minio:9000",

            f"pipeline-{step}:latest"
        ], check=True)

print("\n🔥 PIPELINE TERMINÉE")
