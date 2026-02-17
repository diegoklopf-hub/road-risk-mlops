from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
import os


DAG_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ON_HOST = os.path.abspath(os.path.join(DAG_FOLDER, "../../../"))

images_list = [
    'pipeline-01_data_import:latest',
    'pipeline-02_data_clean:latest',
    'pipeline-03_merge:latest',
    'pipeline-04_encodage:latest',
    'pipeline-05_data_transformation:latest',
    'pipeline-06_resampling:latest',
    'pipeline-07_model_trainer:latest',
    'pipeline-08_model_evaluation:latest'
]


def get_task_simple_name(image_name):
    return image_name.replace('pipeline-', '').split(':')[0]


MLFLOW_PARENT_RUN_ID_TEMPLATE = "{{ dag_run.conf.get('mlflow_parent_run_id', '') if dag_run and dag_run.conf else '' }}"

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
    
    task_list = [ DockerOperator(
        task_id=get_task_simple_name(img),
        image=img,
        api_version='auto',
        network_mode="saver_network",
        auto_remove=True,
        mounts=[{'Source': f"{PROJECT_ROOT_ON_HOST}/data",'Target': '/app/data','Type': 'bind'},
                {'Source': f"{PROJECT_ROOT_ON_HOST}/models",'Target': '/app/models','Type': 'bind'},
                {'Source': f"{PROJECT_ROOT_ON_HOST}/metrics",'Target': '/app/metrics','Type': 'bind'},
                {'Source': f"{PROJECT_ROOT_ON_HOST}/logs/run_logs", 'Target': '/app/logs/run_logs', 'Type': 'bind'}
            ],
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow_server:5000',
            'MLFLOW_PARENT_RUN_ID': MLFLOW_PARENT_RUN_ID_TEMPLATE,
            'AWS_ACCESS_KEY_ID': 'minio',
            'AWS_SECRET_ACCESS_KEY': 'minio123',
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_REGION': 'us-east-1'
        }
    ) for i, img in enumerate(images_list) ]

    for i in range(len(task_list) - 1):
        task_list[i] >> task_list[i+1]

