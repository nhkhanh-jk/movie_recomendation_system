from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_training_pipeline",
    description="Train CB + SVD + Hybrid models từ PostgreSQL",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",        # train lại mỗi tuần
    catchup=False,
    tags=["ml", "training"],
) as dag:

    run_training = BashOperator(
        task_id="run_training_job",
        bash_command="python /opt/airflow/main.py --mode train",
    )
