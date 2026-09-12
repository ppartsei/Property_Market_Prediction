import logging

import pandas as pd

from catboost import Pool
from scripts.preprocess import cat_features, text_features


from scripts.preprocess import preprocess_data
from scripts.s3_utils import (
    download_json,
    download_model,
    upload_json,
)

logger = logging.getLogger(__name__)


def run_inference(
    s3_bucket: str,
    model_key: str,
    output_prefix: str,
    threshold: float,
    s3_conn_id: str,
    **context,
):
    """
    Загружает snapshot из S3,
    выполняет preprocessing,
    загружает модель,
    делает бинарное предсказание
    и сохраняет результат обратно в S3.

    Через XCom передаётся только путь к результату.
    """

    ti = context["ti"]

    input_s3_key = ti.xcom_pull(
        task_ids="load_data.extract_snapshot_to_s3",
        key="input_s3_key",
    )

    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )

    if not input_s3_key:
        raise ValueError("Не найден входной snapshot в S3.")

    if not inference_date:
        raise ValueError("Не найдена дата инференса.")

    logger.info(f"Начинаем inference за дату " f"{inference_date}.")

    # ─────────────────────────────────────────────
    # 1. Загружаем snapshot из S3
    # ─────────────────────────────────────────────

    raw_data = download_json(
        bucket=s3_bucket,
        key=input_s3_key,
        s3_conn_id=s3_conn_id,
    )

    df_raw = pd.DataFrame(raw_data)

    if df_raw.empty:
        raise ValueError("Входной snapshot пуст.")

    logger.info(f"Получено {len(df_raw)} строк.")

    # ─────────────────────────────────────────────
    # 2. Предобработка
    # ─────────────────────────────────────────────

    df_processed = preprocess_data(df_raw)

    logger.info(f"Предобработка завершена. " f"Признаков: {df_processed.shape[1]}")

    # ─────────────────────────────────────────────
    # 3. Загружаем модель
    # ─────────────────────────────────────────────

    model = download_model(
        bucket=s3_bucket,
        model_key=model_key,
        s3_conn_id=s3_conn_id,
    )

    logger.info(f"Используется модель: " f"s3://{s3_bucket}/{model_key}")

    # ─────────────────────────────────────────────
    # 4. Получаем вероятность класса 1
    # ─────────────────────────────────────────────

    test_pool = Pool(
        data=df_processed,
        cat_features=cat_features,
        text_features=text_features,
    )

    y_proba = model.predict_proba(test_pool)[:, 1]

    # ─────────────────────────────────────────────
    # 5. Бинаризация по threshold = 0.56
    # ─────────────────────────────────────────────

    y_pred = (y_proba >= threshold).astype(int)

    # ─────────────────────────────────────────────
    # 6. Формируем результат
    # ─────────────────────────────────────────────

    result_df = pd.DataFrame(
        {
            "id": df_raw["id"].values,
            "score": y_pred,
        }
    )

    logger.info(
        f"Инференс завершён. "
        f"Всего объявлений: {len(result_df)}. "
        f"Положительных предсказаний: "
        f"{int(result_df['score'].sum())}. "
        f"Отрицательных: "
        f"{int((result_df['score'] == 0).sum())}."
    )

    # ─────────────────────────────────────────────
    # 7. Сохраняем результат в S3
    # ─────────────────────────────────────────────

    output_s3_key = f"{output_prefix}/" f"predictions_{inference_date}.json"

    upload_json(
        result_df.to_dict(orient="records"),
        bucket=s3_bucket,
        key=output_s3_key,
        s3_conn_id=s3_conn_id,
    )

    # Передаём через XCom только небольшие значения.
    ti.xcom_push(
        key="output_s3_key",
        value=output_s3_key,
    )

    ti.xcom_push(
        key="s3_bucket",
        value=s3_bucket,
    )

    ti.xcom_push(
        key="s3_conn_id",
        value=s3_conn_id,
    )

    logger.info(f"Результаты сохранены в " f"s3://{s3_bucket}/{output_s3_key}")
