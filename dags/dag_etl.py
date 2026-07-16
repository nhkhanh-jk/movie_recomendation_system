from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="etl_pipeline",
    description="Đọc từ MySQL → clean → ghi vào PostgreSQL",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",         # chạy mỗi ngày 1 lần tự động
    catchup=False,             # không chạy bù các ngày đã qua
    tags=["etl", "movielens"],
) as dag:

    run_etl = BashOperator(
        task_id="run_etl_job",
        bash_command="python /opt/airflow/scripts/etl.py",
    )
