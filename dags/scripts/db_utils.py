# scripts/db_utils.py

import logging
import re

from airflow.providers.postgres.hooks.postgres import PostgresHook

from scripts.s3_utils import download_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Обязательные колонки snapshot
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "id",
    "name",
    "host_listings_count",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_nights_avg_ntm",
    "instant_bookable",
    "neighbourhood_cleansed",
    "snapshot_date",
]


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные проверки
# ─────────────────────────────────────────────────────────────────────────────


def _validate_table_name(table_name: str) -> None:
    """Проверяет, что имя таблицы имеет вид schema.table."""
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*", table_name):
        raise ValueError(
            f"Невалидное имя таблицы: {table_name!r}. "
            "Ожидается формат 'schema.table'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Проверка snapshot в БД
# ─────────────────────────────────────────────────────────────────────────────


def validate_source_data(
    pg_conn_id: str,
    source_table: str,
    **context,
):
    """
    Проверяет:
      - что snapshot содержит все обязательные колонки;
      - что snapshot не пуст;
      - что MAX(snapshot_date) определён.

    Через XCom передаёт inference_date (строкой).
    """
    _validate_table_name(source_table)

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    # 1. Проверка колонок
    columns_query = f"SELECT * FROM {source_table} LIMIT 0"
    df_columns = hook.get_pandas_df(columns_query)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_columns.columns]

    if missing_columns:
        raise ValueError(
            f"В {source_table} отсутствуют обязательные колонки: {missing_columns}"
        )

    # 2. Проверка на пустоту + получение max(snapshot_date) одним запросом
    stats_query = f"""
        SELECT
            COUNT(*)             AS row_count,
            MAX(snapshot_date)   AS max_date
        FROM {source_table}
    """

    row_count, inference_date = hook.get_first(stats_query)

    if row_count == 0:
        raise ValueError(f"Таблица {source_table} пуста.")

    if inference_date is None:
        raise ValueError(
            "Не удалось определить дату инференса: "
            "snapshot_date содержит только NULL."
        )

    context["ti"].xcom_push(
        key="inference_date",
        value=str(inference_date),
    )

    logger.info(
        "Источник проверен. Строк: %d. Дата инференса: %s",
        row_count,
        inference_date,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Выгрузка snapshot в S3
# ─────────────────────────────────────────────────────────────────────────────


def extract_snapshot_to_s3(
    pg_conn_id: str,
    source_table: str,
    s3_bucket: str,
    s3_prefix: str,
    s3_conn_id: str,
    **context,
):
    """
    Загружает в S3 только последний snapshot (за inference_date).
    Через XCom передаётся только ключ S3.
    """
    _validate_table_name(source_table)

    ti = context["ti"]

    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )

    if not inference_date:
        raise ValueError("Не найдена дата инференса (inference_date).")

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    query = f"""
        SELECT *
        FROM {source_table}
        WHERE snapshot_date = %s
    """

    df = hook.get_pandas_df(query, parameters=(inference_date,))

    if df.empty:
        raise ValueError(f"Snapshot за дату {inference_date} пуст.")

    logger.info("Загружено %d строк за дату %s.", len(df), inference_date)

    key = f"{s3_prefix}/snapshot_{inference_date}.json"

    from scripts.s3_utils import upload_json

    upload_json(
        df.to_dict(orient="records"),
        bucket=s3_bucket,
        key=key,
        s3_conn_id=s3_conn_id,
    )

    ti.xcom_push(key="input_s3_key", value=key)

    logger.info("Snapshot сохранён в s3://%s/%s", s3_bucket, key)


# ─────────────────────────────────────────────────────────────────────────────
# Запись предсказаний в существующую витрину
# ─────────────────────────────────────────────────────────────────────────────


def save_predictions_to_db(
    pg_conn_id: str,
    target_table: str,
    **context,
):
    """
    Загружает результаты в СУЩЕСТВУЮЩУЮ витрину.

    Идемпотентность: если в target_table уже есть строки
    за inference_date — загрузка пропускается.
    """
    _validate_table_name(target_table)

    ti = context["ti"]

    output_s3_key = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="output_s3_key",
    )
    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )
    s3_bucket = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="s3_bucket",
    )
    s3_conn_id = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="s3_conn_id",
    )

    if not output_s3_key:
        raise ValueError("Не найден S3-файл с результатами (output_s3_key).")

    if not inference_date:
        raise ValueError("Не найдена дата инференса (inference_date).")

    if not s3_bucket or not s3_conn_id:
        raise ValueError("Не найдены параметры S3 (s3_bucket / s3_conn_id).")

    # 1. Скачиваем предсказания
    predictions = download_json(
        bucket=s3_bucket,
        key=output_s3_key,
        s3_conn_id=s3_conn_id,
    )

    if not predictions:
        raise ValueError(
            f"Файл с предсказаниями пуст: s3://{s3_bucket}/{output_s3_key}"
        )

    logger.info(
        "Скачано %d предсказаний из s3://%s/%s",
        len(predictions),
        s3_bucket,
        output_s3_key,
    )

    # 2. Подключение к БД
    hook = PostgresHook(postgres_conn_id=pg_conn_id)
    conn = hook.get_conn()
    cursor = None

    try:
        cursor = conn.cursor()

        # 3. Проверка идемпотентности
        cursor.execute(
            f"SELECT EXISTS ("
            f"  SELECT 1 FROM {target_table} WHERE inference_date = %s"
            f")",
            (inference_date,),
        )
        already_exists = cursor.fetchone()[0]

        if already_exists:
            logger.warning(
                "Результаты за дату %s уже есть в %s. Загрузка пропущена.",
                inference_date,
                target_table,
            )
            return

        # 4. Подготовка батча
        rows = [
            (int(row["id"]), int(row["score"]), inference_date) for row in predictions
        ]

        # 5. Батчевая вставка
        from psycopg2.extras import execute_values

        execute_values(
            cursor,
            f"INSERT INTO {target_table} (id, score, inference_date) VALUES %s",
            rows,
            page_size=1000,
        )

        conn.commit()

        logger.info(
            "В %s добавлено %d строк за дату %s.",
            target_table,
            len(rows),
            inference_date,
        )

    except Exception:
        conn.rollback()
        logger.exception("Ошибка при записи предсказаний в %s", target_table)
        raise

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()
