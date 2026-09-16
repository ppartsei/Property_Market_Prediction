# scripts/s3_utils.py

import io
import json
import logging
import pickle

import boto3
from airflow.hooks.base import BaseHook
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Константы Yandex Object Storage
# ─────────────────────────────────────────────────────────────────────────────

S3_ENDPOINT = "https://storage.yandexcloud.net"
S3_REGION = "ru-central1"


# ─────────────────────────────────────────────────────────────────────────────
# Создание S3-клиента
# ─────────────────────────────────────────────────────────────────────────────


def get_s3_client(s3_conn_id: str):
    """
    Создаёт boto3 S3-клиент, используя credentials из Airflow Connection.
    """
    connection = BaseHook.get_connection(s3_conn_id)

    if not connection.login or not connection.password:
        raise ValueError(
            f"В Airflow Connection '{s3_conn_id}' "
            "не указаны access key (login) и secret key (password)."
        )

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=connection.login,
        aws_secret_access_key=connection.password,
        region_name=S3_REGION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Проверка модели
# ─────────────────────────────────────────────────────────────────────────────


def check_model_exists(
    bucket: str,
    model_key: str,
    s3_conn_id: str,
    **context,
):
    """
    Проверяет наличие модели в S3 через head_object.
    Не тянет содержимое — только метаданные.

    При отсутствии модели задача завершается с ошибкой.
    """
    logger.info("Проверяем модель s3://%s/%s", bucket, model_key)

    s3_client = get_s3_client(s3_conn_id)

    try:
        response = s3_client.head_object(
            Bucket=bucket,
            Key=model_key,
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        logger.error(
            "Модель недоступна: s3://%s/%s (код: %s)",
            bucket,
            model_key,
            error_code,
        )
        raise RuntimeError(
            f"Актуальная модель отсутствует или недоступна в S3: "
            f"s3://{bucket}/{model_key}"
        ) from exc

    size = response.get("ContentLength", 0)
    modified = response.get("LastModified")

    logger.info(
        "Модель найдена. Размер: %d байт. Последнее изменение: %s",
        size,
        modified,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pickle — загрузка модели и статистик
# ─────────────────────────────────────────────────────────────────────────────


def download_pickle(
    bucket: str,
    key: str,
    s3_conn_id: str,
):
    """
    Загружает pickle-объект из S3.
    Используется для модели и для stats (preprocess_stats.pkl).
    """
    logger.info("Загрузка pickle из s3://%s/%s", bucket, key)

    s3_client = get_s3_client(s3_conn_id)

    buffer = io.BytesIO()

    try:
        s3_client.download_fileobj(
            Bucket=bucket,
            Key=key,
            Fileobj=buffer,
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise FileNotFoundError(
            f"Не удалось скачать s3://{bucket}/{key} (код: {error_code})"
        ) from exc

    buffer.seek(0)
    obj = pickle.load(buffer)

    logger.info("Pickle успешно загружен из S3: %s", type(obj).__name__)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# JSON — промежуточные артефакты (snapshot, predictions)
# ─────────────────────────────────────────────────────────────────────────────


def upload_json(
    data,
    bucket: str,
    key: str,
    s3_conn_id: str,
):
    """
    Сохраняет JSON в S3.
    Кириллица сохраняется как есть (ensure_ascii=False).
    """
    s3_client = get_s3_client(s3_conn_id)

    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ContentEncoding="utf-8",
    )

    logger.info("JSON сохранён в s3://%s/%s (%d байт)", bucket, key, len(body))


def download_json(
    bucket: str,
    key: str,
    s3_conn_id: str,
):
    """
    Загружает JSON из S3.
    """
    s3_client = get_s3_client(s3_conn_id)

    try:
        response = s3_client.get_object(
            Bucket=bucket,
            Key=key,
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise FileNotFoundError(
            f"Не удалось скачать s3://{bucket}/{key} (код: {error_code})"
        ) from exc

    body = response["Body"].read()
    data = json.loads(body.decode("utf-8"))

    logger.info(
        "JSON загружен из s3://%s/%s (%d записей)",
        bucket,
        key,
        len(data) if hasattr(data, "__len__") else 0,
    )
    return data
