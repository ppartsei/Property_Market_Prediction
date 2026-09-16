# scripts/inference.py

import logging

import pandas as pd
from catboost import Pool

from scripts.preprocess import (
    all_features,
    cat_features,
    preprocess_data,
    text_features,
)
from scripts.s3_utils import (
    download_json,
    download_pickle,
    upload_json,
)

logger = logging.getLogger(__name__)


def run_inference(
    s3_bucket: str,
    model_key: str,
    stats_key: str,
    output_prefix: str,
    threshold: float,
    s3_conn_id: str,
    **context,
):
    """
    Полный цикл инференса:
      1. Скачивает snapshot из S3.
      2. Скачивает stats (preprocess_stats.pkl) из S3.
      3. Скачивает модель (.pkl) из S3.
      4. Выполняет предобработку.
      5. Считает вероятности и бинаризует по threshold.
      6. Сохраняет {id, score} в S3.

    Через XCom передаётся только путь к результату в S3.
    """
    ti = context["ti"]

    # ─────────────────────────────────────────────────────────
    # 0. Читаем ключи из XCom
    # ─────────────────────────────────────────────────────────

    input_s3_key = ti.xcom_pull(
        task_ids="load_data.extract_snapshot_to_s3",
        key="input_s3_key",
    )

    inference_date = ti.xcom_pull(
        task_ids="load_data.validate_source_data",
        key="inference_date",
    )

    if not input_s3_key:
        raise ValueError("Не найден входной snapshot в S3 (input_s3_key).")

    if not inference_date:
        raise ValueError("Не найдена дата инференса (inference_date).")

    logger.info("Запуск inference за дату %s", inference_date)
    logger.info("Модель: s3://%s/%s", s3_bucket, model_key)
    logger.info("Stats:  s3://%s/%s", s3_bucket, stats_key)
    logger.info("Threshold: %s", threshold)

    # ─────────────────────────────────────────────────────────
    # 1. Snapshot из S3
    # ─────────────────────────────────────────────────────────

    raw_records = download_json(
        bucket=s3_bucket,
        key=input_s3_key,
        s3_conn_id=s3_conn_id,
    )
    df_raw = pd.DataFrame(raw_records)

    if df_raw.empty:
        raise ValueError(f"Snapshot пуст: s3://{s3_bucket}/{input_s3_key}")

    logger.info("Получено строк из snapshot: %d", len(df_raw))

    # ─────────────────────────────────────────────────────────
    # 2. Stats из S3
    # ─────────────────────────────────────────────────────────

    stats = download_pickle(
        bucket=s3_bucket,
        key=stats_key,
        s3_conn_id=s3_conn_id,
    )

    if not isinstance(stats, dict):
        raise ValueError(f"Stats должен быть dict, получен {type(stats).__name__}")

    required_stats_keys = {
        "bedrooms_by_acc",
        "beds_by_acc",
        "bedrooms_global",
        "beds_global",
        "clip_upper",
    }
    missing_stats = required_stats_keys - set(stats.keys())

    if missing_stats:
        raise ValueError(f"В preprocess_stats.pkl отсутствуют ключи: {missing_stats}")

    logger.info("Stats загружены. Ключи: %s", list(stats.keys()))

    # ─────────────────────────────────────────────────────────
    # 3. Модель из S3
    # ─────────────────────────────────────────────────────────

    model = download_pickle(
        bucket=s3_bucket,
        key=model_key,
        s3_conn_id=s3_conn_id,
    )

    logger.info(
        "Модель загружена. Тип: %s, ожидает признаков: %d",
        type(model).__name__,
        len(model.feature_names_),
    )

    # ─────────────────────────────────────────────────────────
    # 4. Предобработка
    # ─────────────────────────────────────────────────────────

    df_processed = preprocess_data(df_raw, stats)

    logger.info(
        "Предобработка завершена. Строк: %d, признаков: %d",
        df_processed.shape[0],
        df_processed.shape[1],
    )

    # ─────────────────────────────────────────────────────────
    # 5. Проверка совпадения признаков модели и данных
    # ─────────────────────────────────────────────────────────

    expected = list(model.feature_names_)
    actual = list(df_processed.columns)

    if expected != actual:
        # покажем, что именно не совпадает
        only_model = set(expected) - set(actual)
        only_data = set(actual) - set(expected)
        raise ValueError(
            "Порядок/набор признаков модели и данных не совпадает.\n"
            f"  Только в модели: {sorted(only_model)}\n"
            f"  Только в данных: {sorted(only_data)}\n"
            f"  Модель: {expected}\n"
            f"  Данные: {actual}"
        )

    logger.info("Признаки совпадают с моделью ✅")

    # ─────────────────────────────────────────────────────────
    # 6. Pool + предсказание
    # ─────────────────────────────────────────────────────────

    test_pool = Pool(
        data=df_processed,
        cat_features=cat_features,
        text_features=text_features,
    )

    y_proba = model.predict_proba(test_pool)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    n_pos = int(y_pred.sum())
    n_neg = int((y_pred == 0).sum())

    logger.info(
        "Инференс завершён. Всего: %d. "
        "Положительных: %d (%.2f%%). Отрицательных: %d.",
        len(y_pred),
        n_pos,
        100.0 * n_pos / len(y_pred) if len(y_pred) else 0.0,
        n_neg,
    )

    # ─────────────────────────────────────────────────────────
    # 7. Результат и S3
    # ─────────────────────────────────────────────────────────

    if "id" not in df_raw.columns:
        raise ValueError("В snapshot отсутствует колонка 'id'.")

    result_df = pd.DataFrame(
        {
            "id": df_raw["id"].values,
            "score": y_pred,
        }
    )

    output_s3_key = f"{output_prefix}/predictions_{inference_date}.json"

    upload_json(
        result_df.to_dict(orient="records"),
        bucket=s3_bucket,
        key=output_s3_key,
        s3_conn_id=s3_conn_id,
    )

    # ─────────────────────────────────────────────────────────
    # 8. XCom для следующего таска
    # ─────────────────────────────────────────────────────────

    ti.xcom_push(key="output_s3_key", value=output_s3_key)
    ti.xcom_push(key="s3_bucket", value=s3_bucket)
    ti.xcom_push(key="s3_conn_id", value=s3_conn_id)

    logger.info(
        "Результаты сохранены в s3://%s/%s",
        s3_bucket,
        output_s3_key,
    )
