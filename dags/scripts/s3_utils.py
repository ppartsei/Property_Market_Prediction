import io
import json
import logging
import pickle

import os
import tempfile
import csv
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

import boto3

from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)

S3_ENDPOINT = "https://storage.yandexcloud.net"
S3_REGION = "ru-central1"


def get_s3_client(s3_conn_id: str):
    """
    Создаёт S3-клиент, используя credentials из Airflow Connection.
    """

    connection = BaseHook.get_connection(s3_conn_id)

    if not connection.login or not connection.password:
        raise ValueError(
            f"В Airflow Connection '{s3_conn_id}' "
            "не указаны access key и secret key."
        )

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=connection.login,
        aws_secret_access_key=connection.password,
        region_name=S3_REGION,
    )


def check_model_exists(
    bucket: str,
    model_key: str,
    s3_conn_id: str,
    **context,
):
    """
    Проверяет наличие модели в S3.

    При отсутствии модели задача завершается с ошибкой.
    """

    logger.info(f"Проверяем модель s3://{bucket}/{model_key}")

    s3_client = get_s3_client(s3_conn_id)

    try:
        response = s3_client.head_object(
            Bucket=bucket,
            Key=model_key,
        )

    except Exception as exc:
        logger.error(f"Модель недоступна: s3://{bucket}/{model_key}")
        raise RuntimeError(
            "Актуальная модель отсутствует или недоступна в S3."
        ) from exc

    size = response.get("ContentLength", 0)
    modified = response.get("LastModified")

    logger.info(
        f"Модель найдена. Размер: {size} байт. " f"Последнее изменение: {modified}"
    )


def download_model(
    bucket: str,
    model_key: str,
    s3_conn_id: str,
):
    """
    Загружает pickle-модель из S3.
    """

    logger.info(f"Загрузка модели s3://{bucket}/{model_key}")

    s3_client = get_s3_client(s3_conn_id)

    buffer = io.BytesIO()

    s3_client.download_fileobj(
        Bucket=bucket,
        Key=model_key,
        Fileobj=buffer,
    )

    buffer.seek(0)

    model = pickle.load(buffer)

    logger.info("Модель успешно загружена из S3.")

    return model


def upload_json(
    data,
    bucket: str,
    key: str,
    s3_conn_id: str,
):
    """
    Сохраняет JSON в S3.
    """

    s3_client = get_s3_client(s3_conn_id)

    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )

    logger.info(f"JSON сохранён в s3://{bucket}/{key}")


def download_json(
    bucket: str,
    key: str,
    s3_conn_id: str,
):
    """
    Загружает JSON из S3.
    """

    s3_client = get_s3_client(s3_conn_id)

    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    body = response["Body"].read()

    return json.loads(body.decode("utf-8"))
