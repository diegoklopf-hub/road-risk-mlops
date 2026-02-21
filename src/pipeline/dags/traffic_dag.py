from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from traffic_generator import generate_traffic

# ── Config ─────────────────────────────────────────────────────
BASE_URL       = "https://nginx"
AUTH           = ("admin", "password")
TOTAL_REQUESTS = 20_000
CONCURRENCY    = 40

# ── DAG ─────────────────────────────────────────────────────────
with DAG(
    dag_id="traffic_dag",
    tags=["S.A.V.E.R", "Liora Projet MLOps"],
    schedule_interval=None,
    default_args={
        "owner": "airflow",
        "start_date": days_ago(0, minute=1),
    },
    catchup=False,
) as dag:

    PythonOperator(
        task_id="generate_traffic",
        python_callable=generate_traffic,
        op_kwargs={
            "total_requests": TOTAL_REQUESTS,
            "concurrency": CONCURRENCY,
            "base_url": BASE_URL,
            "auth": AUTH,
        },
    )