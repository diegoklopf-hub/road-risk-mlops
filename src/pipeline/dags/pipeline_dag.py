from pathlib import Path
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
import os

PROJECT_ROOT_ON_HOST = os.environ['HOST_PROJECT_ROOT']

images_list = [
    'pipeline-01_data_import:latest',
    'pipeline-02_data_clean:latest',
    'pipeline-03_merge:latest',
    'pipeline-04_encodage:latest',
    'pipeline-05_data_transformation:latest',
    'pipeline-06_resampling:latest',
    'pipeline-07_model_trainer:latest',
    'pipeline-08_model_evaluation:latest',
    'pipeline-09_shap_explicability:latest'
]


def get_task_simple_name(image_name):
    return image_name.replace('pipeline-', '').split(':')[0]


SHARED_MOUNTS = [
    {'Source': f"{PROJECT_ROOT_ON_HOST}/data", 'Target': '/app/data', 'Type': 'bind'},
    {'Source': f"{PROJECT_ROOT_ON_HOST}/models", 'Target': '/app/models', 'Type': 'bind'},
    {'Source': f"{PROJECT_ROOT_ON_HOST}/metrics", 'Target': '/app/metrics', 'Type': 'bind'},
    {'Source': f"{PROJECT_ROOT_ON_HOST}/logs", 'Target': '/app/logs', 'Type': 'bind'},
]

SHARED_ENV = {
    'MLFLOW_TRACKING_URI': 'http://mlflow_server:5000',
    'AWS_ACCESS_KEY_ID': 'minio',
    'AWS_SECRET_ACCESS_KEY': 'minio123',
    'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
    'AWS_REGION': 'us-east-1',
    'PYTHONUNBUFFERED': '1', 
}

with DAG(
    dag_id='pipeline_dag',
    tags=['S.A.V.E.R', 'Liora Projet MLOps'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1)
    },
    catchup=False
) as my_dag:

    # Clean parent-run file before each execution
    clean_parent_run = DockerOperator(
        task_id='clean_parent_run',
        image='alpine:latest',
        api_version='auto',
        network_mode="saver_network",
        auto_remove=True,
        command="sh -c 'rm -f /app/logs/mlflow_parent_run_id.txt && echo CLEANED'",
        mounts=SHARED_MOUNTS,
    )

    task_list = [DockerOperator(
        task_id=get_task_simple_name(img),
        image=img,
        api_version='auto',
        network_mode="saver_network",
        auto_remove=True,
        mounts=SHARED_MOUNTS,
        environment=SHARED_ENV,
    ) for img in images_list]

    clean_parent_run >> task_list[0]
    for i in range(len(task_list) - 1):
        task_list[i] >> task_list[i + 1]