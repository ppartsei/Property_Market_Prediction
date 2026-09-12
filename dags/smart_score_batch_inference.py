from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable


from scripts.db_utils import (
    create_target_table,
    validate_source_data,
    extract_snapshot_to_s3,
    save_predictions_to_db,
)

from scripts.s3_utils import check_model_exists

from scripts.inference import run_inference

# ─────────────────────────────────────────────────────────────
# Настройки из Airflow Variables
# ─────────────────────────────────────────────────────────────

SOURCE_TABLE = Variable.get(
    "smartscore_source_table",
    default_var="final_project.smartscore_test_snapshot",
)

TARGET_TABLE = Variable.get(
    "smartscore_target_table",
    default_var="final_project.SmartScore_predict",
)

S3_BUCKET = Variable.get(
    "smartscore_s3_bucket",
)

S3_MODEL_KEY = Variable.get(
    "smartscore_model_key",
    default_var="catboost_model.pkl",
)

S3_INPUT_PREFIX = Variable.get(
    "smartscore_input_prefix",
    default_var="smartscore/input",
)

S3_OUTPUT_PREFIX = Variable.get(
    "smartscore_output_prefix",
    default_var="smartscore/output",
)

THRESHOLD = float(
    Variable.get(
        "smartscore_threshold",
        default_var="0.56",
    )
)

PG_CONN_ID = Variable.get(
    "smartscore_pg_conn_id",
    default_var="postgres_smartscore",
)

S3_CONN_ID = Variable.get(
    "smartscore_s3_conn_id",
    default_var="yandex_s3",
)


# ─────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "data_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="smartscore_batch_inference",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["smartscore", "inference"],
    description="Батч-инференс SmartScore для фиксированного snapshot",
) as dag:

    # ─────────────────────────────────────────────────────────
    # Проверка модели
    # ─────────────────────────────────────────────────────────

    check_model = PythonOperator(
        task_id="check_model",
        python_callable=check_model_exists,
        op_kwargs={
            "bucket": S3_BUCKET,
            "model_key": S3_MODEL_KEY,
            "s3_conn_id": S3_CONN_ID,
        },
    )

    # ─────────────────────────────────────────────────────────
    # Подготовка БД
    # ─────────────────────────────────────────────────────────

    with TaskGroup(
        group_id="load_data",
        tooltip="Проверка и загрузка snapshot",
    ) as load_data:

        create_table = PythonOperator(
            task_id="create_target_table",
            python_callable=create_target_table,
            op_kwargs={
                "pg_conn_id": PG_CONN_ID,
                "target_table": TARGET_TABLE,
            },
        )

        validate_data = PythonOperator(
            task_id="validate_source_data",
            python_callable=validate_source_data,
            op_kwargs={
                "pg_conn_id": PG_CONN_ID,
                "source_table": SOURCE_TABLE,
            },
        )

        extract_data = PythonOperator(
            task_id="extract_snapshot_to_s3",
            python_callable=extract_snapshot_to_s3,
            op_kwargs={
                "pg_conn_id": PG_CONN_ID,
                "source_table": SOURCE_TABLE,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_INPUT_PREFIX,
                "s3_conn_id": S3_CONN_ID,
            },
        )

        create_table >> validate_data >> extract_data

    # ─────────────────────────────────────────────────────────
    # Инференс
    # ─────────────────────────────────────────────────────────

    with TaskGroup(
        group_id="inference",
        tooltip="Предобработка и батч-инференс",
    ) as inference:

        predict = PythonOperator(
            task_id="run_inference",
            python_callable=run_inference,
            op_kwargs={
                "s3_bucket": S3_BUCKET,
                "model_key": S3_MODEL_KEY,
                "output_prefix": S3_OUTPUT_PREFIX,
                "threshold": THRESHOLD,
                "s3_conn_id": S3_CONN_ID,
            },
        )

    # ─────────────────────────────────────────────────────────
    # Сохранение результата
    # ─────────────────────────────────────────────────────────

    with TaskGroup(
        group_id="save_results",
        tooltip="Сохранение предсказаний в PostgreSQL",
    ) as save_results:

        save_predictions = PythonOperator(
            task_id="save_predictions",
            python_callable=save_predictions_to_db,
            op_kwargs={
                "pg_conn_id": PG_CONN_ID,
                "target_table": TARGET_TABLE,
            },
        )

    # ─────────────────────────────────────────────────────────
    # Порядок выполнения
    # ─────────────────────────────────────────────────────────

    check_model >> load_data >> inference >> save_results
