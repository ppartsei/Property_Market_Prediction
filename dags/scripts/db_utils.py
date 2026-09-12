import logging

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

from scripts.s3_utils import download_json

from scripts.s3_utils import upload_json

logger = logging.getLogger(__name__)


# Обязательные колонки snapshot,
# необходимые для preprocessing и inference.
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


def create_target_table(
    pg_conn_id: str,
    target_table: str,
    **context,
):
    """
    Создаёт витрину, если она ещё не существует.
    """

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    schema, table = target_table.split(".", 1)

    query = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            id BIGINT NOT NULL,
            score INTEGER NOT NULL,
            inference_date DATE NOT NULL
        )
    """

    logger.info(f"Проверяем наличие таблицы {target_table}")

    hook.run(query)

    logger.info(f"Таблица {target_table} готова к работе.")


def validate_source_data(
    pg_conn_id: str,
    source_table: str,
    **context,
):
    """
    Проверяет:
    - что snapshot не пустой;
    - что snapshot_date существует;
    - что обязательные колонки присутствуют;
    - что максимальная дата определена.
    """

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    columns_query = f"""
        SELECT *
        FROM {source_table}
        LIMIT 0
    """

    df_columns = hook.get_pandas_df(columns_query)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df_columns.columns
    ]

    if missing_columns:
        raise ValueError(
            f"В {source_table} отсутствуют " f"обязательные колонки: {missing_columns}"
        )

    count_query = f"""
        SELECT COUNT(*)
        FROM {source_table}
    """

    count = hook.get_first(count_query)[0]

    if count == 0:
        raise ValueError(f"Таблица {source_table} пуста.")

    max_date_query = f"""
        SELECT MAX(snapshot_date)
        FROM {source_table}
    """

    inference_date = hook.get_first(max_date_query)[0]

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
        f"Источник проверен. " f"Строк: {count}. " f"Дата инференса: {inference_date}"
    )


def extract_snapshot_to_s3(
    pg_conn_id: str,
    source_table: str,
    s3_bucket: str,
    s3_prefix: str,
    s3_conn_id: str,
    **context,
):
    """
    Загружает только последний snapshot в S3.

    Через XCom передаётся только ключ S3,
    сами данные через XCom не передаются.
    """

    ti = context["ti"]

    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )

    if not inference_date:
        raise ValueError("Не найдена дата инференса.")

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    query = f"""
        SELECT *
        FROM {source_table}
        WHERE snapshot_date = %s
    """

    df = hook.get_pandas_df(
        query,
        parameters=(inference_date,),
    )

    if df.empty:
        raise ValueError(f"Snapshot за дату {inference_date} пуст.")

    logger.info(f"Загружено {len(df)} строк " f"за дату {inference_date}.")

    key = f"{s3_prefix}/" f"snapshot_{inference_date}.json"

    upload_json(
        df.to_dict(orient="records"),
        bucket=s3_bucket,
        key=key,
        s3_conn_id=s3_conn_id,
    )

    ti.xcom_push(
        key="input_s3_key",
        value=key,
    )

    logger.info(f"Snapshot сохранён в " f"s3://{s3_bucket}/{key}")


def save_predictions_to_db(
    pg_conn_id: str,
    target_table: str,
    **context,
):
    """
    Загружает результаты в витрину.

    Если inference_date уже существует,
    загрузка полностью пропускается.
    """

    ti = context["ti"]

    output_s3_key = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="output_s3_key",
    )

    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )

    if not output_s3_key:
        raise ValueError("Не найден S3-файл с результатами.")

    if not inference_date:
        raise ValueError("Не найдена дата инференса.")

    from scripts.s3_utils import download_json

    s3_bucket = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="s3_bucket",
    )

    s3_conn_id = ti.xcom_pull(
        task_ids="inference.run_inference",
        key="s3_conn_id",
    )

    if not s3_bucket or not s3_conn_id:
        raise ValueError("Не найдены параметры S3.")

    predictions = download_json(
        bucket=s3_bucket,
        key=output_s3_key,
        s3_conn_id=s3_conn_id,
    )

    if not predictions:
        raise ValueError("Файл с предсказаниями пуст.")

    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    conn = hook.get_conn()

    try:
        cursor = conn.cursor()

        # Проверяем наличие результатов за дату.
        check_query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM {target_table}
                WHERE inference_date = %s
            )
        """

        cursor.execute(
            check_query,
            (inference_date,),
        )

        already_exists = cursor.fetchone()[0]

        if already_exists:
            logger.warning(
                f"Результаты за дату "
                f"{inference_date} уже существуют. "
                "Загрузка пропущена."
            )
            return

        insert_query = f"""
            INSERT INTO {target_table}
                (id, score, inference_date)
            VALUES (%s, %s, %s)
        """

        rows = [
            (
                int(row["id"]),
                int(row["score"]),
                inference_date,
            )
            for row in predictions
        ]

        cursor.executemany(
            insert_query,
            rows,
        )

        conn.commit()

        logger.info(
            f"В {target_table} добавлено "
            f"{len(rows)} строк. "
            f"Дата инференса: {inference_date}"
        )

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


logger = logging.getLogger(__name__)


def save_predictions_to_db(pg_conn_id, target_table, **context):
    ti = context["ti"]

    output_s3_key = ti.xcom_pull(
        task_ids="inference.run_inference", key="output_s3_key"
    )
    s3_bucket = ti.xcom_pull(task_ids="inference.run_inference", key="s3_bucket")
    s3_conn_id = ti.xcom_pull(task_ids="inference.run_inference", key="s3_conn_id")

    if not output_s3_key:
        raise ValueError("Не получен путь к файлу предсказаний из XCom")

    predictions = download_json(s3_bucket, output_s3_key, s3_conn_id)
    logger.info("Скачано %d предсказаний из S3: %s", len(predictions), output_s3_key)

    inference_date = None
    for row in predictions:
        if row.get("inference_date"):
            inference_date = row["inference_date"]
            break

    if not inference_date:
        raise ValueError("Не удалось определить inference_date из предсказаний")

    logger.info("Дата инференса: %s", inference_date)

    hook = PostgresHook(postgres_conn_id=pg_conn_id)
    conn = hook.get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT COUNT(*) FROM {table} WHERE inference_date = %s".format(
                table=target_table
            ),
            (inference_date,),
        )
        count = cursor.fetchone()[0]
        logger.info(
            "Найдено %d записей за дату %s в таблице %s",
            count,
            inference_date,
            target_table,
        )

        if count > 0:
            logger.info(
                "Данные за %s уже существуют — пропускаем загрузку", inference_date
            )
            return

        values = [(row["id"], row["score"], inference_date) for row in predictions]

        batch_size = 1000
        inserted = 0
        for i in range(0, len(values), batch_size):
            batch = values[i : i + batch_size]
            cursor.executemany(
                "INSERT INTO {table} (id, score, inference_date) VALUES (%s, %s, %s)".format(
                    table=target_table
                ),
                batch,
            )
            inserted += len(batch)
            logger.info("Вставлено %d / %d строк", inserted, len(values))

        conn.commit()
        logger.info(
            "Готово. Всего вставлено %d строк за дату %s", inserted, inference_date
        )

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
