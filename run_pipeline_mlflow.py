import argparse
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
    "08_model_evaluation",
    "09_shap_explicability"
]

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Step number to start from [1, 2, 3, ..]")
args = parser.parse_args()

if args.start == 0:
    args.start = 1
    print("🚀 Starting pipeline from the first step")
elif args.start < 1 or args.start > len(STEPS):
    raise ValueError(f"--start must be between 1 and {len(STEPS)}: current value is {args.start}")

with mlflow.start_run(run_name="GLOBAL_PIPELINE_RUN") as parent:

    parent_id = parent.info.run_id
    print("\n🧠 PARENT RUN:", parent_id)
    print(f"▶️  Starting from step {args.start}: {STEPS[args.start - 1]}\n")

    for step in STEPS[args.start-1:]:
        print(f"\n🚀 Running {step}")

        subprocess.run([
            "docker","run","--rm",
            "--network","saver_network",

            "-v", f"{os.getcwd()}/data:/app/data",
            "-v", f"{os.getcwd()}/logs/run_logs:/app/logs/run_logs",
            "-v", f"{os.getcwd()}/models:/app/models",
            "-v", f"{os.getcwd()}/metrics:/app/metrics",

            # MLflow tracking
            "-e","MLFLOW_TRACKING_URI=http://mlflow_server:5000",
            "-e",f"MLFLOW_PARENT_RUN_ID={parent_id}",

            # MINIO credentials (required)
            "-e","AWS_ACCESS_KEY_ID=minio",
            "-e","AWS_SECRET_ACCESS_KEY=minio123",
            "-e","MLFLOW_S3_ENDPOINT_URL=http://minio:9000",

            f"pipeline-{step}:latest"
        ], check=True)

print("\n🔥 PIPELINE FINISHED")
