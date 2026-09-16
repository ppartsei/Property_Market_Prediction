# scripts/preprocess.py

import ast
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Списки признаков
# Должны ТОЧНО совпадать с тем, что использовалось при обучении модели.
# ─────────────────────────────────────────────────────────────────────────────

text_features = ["name", "amenities"]

cat_features = [
    "property_type",
    "neighbourhood_cleansed",
    "room_type",
    "has_phone_verification",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "has_email_verification",
    "has_work_email_verification",
    "is_shared_bathroom",
]

num_features = [
    "host_listings_count",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_nights_avg_ntm",
    "host_trust_score",
    "guests_per_bedroom",
    "amenities_count",
]

all_features = num_features + cat_features + text_features


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────


def _extract_bathrooms(text):
    """Извлекает количество санузлов из bathrooms_text."""
    if pd.isna(text):
        return None

    text = str(text).lower()

    if "half-bath" in text:
        return 0.5

    match = re.search(r"([\d.]+)", text)
    return float(match.group(1)) if match else None


def _to_binary(value):
    """Приводит boolean-признак к 0/1. Поддерживает True/False, строки, 1/0, NaN."""
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "t", "1", "yes"):
            return 1
        if v in ("false", "f", "0", "no"):
            return 0

    if value in (1, 1.0):
        return 1
    if value in (0, 0.0):
        return 0

    return 0


def _parse_verifications(x):
    """Преобразует host_verifications в Python list."""
    try:
        if isinstance(x, str):
            result = ast.literal_eval(x)
            return result if isinstance(result, list) else []
        if isinstance(x, list):
            return x
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def _count_amenities(x):
    """Считает количество удобств в сырой JSON-строке amenities."""
    if pd.isna(x) or x == "[]":
        return 0
    if isinstance(x, list):
        return len(x)
    return str(x).count(",") + 1


def amenities_preprocessor(text):
    """
    '["Wifi", "Lock on bedroom door"]' -> 'Wifi Lock_on_bedroom_door'

    ВАЖНО: вызывать ПОСЛЕ создания amenities_count — тот считает
    по сырой JSON-строке с запятыми.
    """
    if pd.isna(text):
        return ""

    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ""

    if not isinstance(items, list):
        return ""

    return " ".join(str(item).replace(" ", "_") for item in items)


def _fix_surrogates(s):
    """
    Склеивает суррогатные пары (эмодзи, сохранённые как два UTF-16 code unit)
    обратно в настоящие Unicode-символы. Одиночные суррогаты заменяет на '?'.

    Нужно, потому что CatBoost строго проверяет UTF-8 при создании text-Pool
    и падает на строках вида '\\ud83d\\udd4a'.
    """
    if not isinstance(s, str):
        s = str(s)

    try:
        return s.encode("utf-16", "surrogatepass").decode("utf-16")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s.encode("utf-8", "replace").decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Stateless feature engineering (построчные операции, без статистик)
# ─────────────────────────────────────────────────────────────────────────────


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Построчные преобразования, повторяющие train-пайплайн.
    Не использует статистик по датасету — безопасно применять к любому срезу.
    """
    df = df.copy()

    # 1. bathrooms из bathrooms_text
    df["bathrooms_from_text"] = df["bathrooms_text"].apply(_extract_bathrooms)

    mask = df["bathrooms"].isnull() & df["bathrooms_from_text"].notna()
    df.loc[mask, "bathrooms"] = df.loc[mask, "bathrooms_from_text"]

    df["bathrooms"] = df["bathrooms"].fillna(0)
    df = df.drop(columns=["bathrooms_from_text"])

    # 2. host_verifications -> 3 бинарных
    hv = df["host_verifications"].apply(_parse_verifications)

    df["has_email_verification"] = hv.apply(lambda x: int("email" in x))
    df["has_phone_verification"] = hv.apply(lambda x: int("phone" in x))
    df["has_work_email_verification"] = hv.apply(lambda x: int("work_email" in x))

    # 3. is_shared_bathroom
    df["is_shared_bathroom"] = (
        df["bathrooms_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("shared")
        .astype(int)
    )

    # 4. бинарные -> int
    for col in ["host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        df[col] = df[col].apply(_to_binary).astype(int)

    # 5. производные
    df["host_trust_score"] = (
        df["has_email_verification"]
        + df["has_phone_verification"]
        + df["has_work_email_verification"]
        + df["host_identity_verified"]
        + df["host_has_profile_pic"]
    )

    # 6. guests_per_bedroom — считаем до заполнения bedrooms медианами,
    #    чтобы не занести в производную сигнал из train.
    df["guests_per_bedroom"] = np.where(
        df["bedrooms"] > 0,
        df["accommodates"] / df["bedrooms"],
        0,
    )

    # 7. amenities_count — по СЫРОЙ строке, до amenities_preprocessor
    df["amenities_count"] = df["amenities"].apply(_count_amenities)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stateful-статистики (значения приходят из train, не считаем на проде!)
# ─────────────────────────────────────────────────────────────────────────────


def apply_stateful_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()

    for col in ["bedrooms", "beds"]:
        medians = df["accommodates"].map(stats[f"{col}_by_acc"])
        df[col] = df[col].fillna(medians)
        df[col] = df[col].fillna(stats[f"{col}_global"])
        if pd.isna(stats[f"{col}_global"]):
            df[col] = df[col].fillna(0)

    for col, upper in stats["clip_upper"].items():
        if col in df.columns:
            df[col] = df[col].clip(upper=upper)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Публичная функция: полный пайплайн предобработки
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_data(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Полная предобработка данных для инференса.

    Параметры
    ---------
    df : pd.DataFrame
        Сырой snapshot из S3.
    stats : dict
        Артефакт, посчитанный на train:
        bedrooms_by_acc, beds_by_acc, bedrooms_global, beds_global, clip_upper.

    Возвращает
    ----------
    pd.DataFrame с колонками из all_features в правильном порядке.
    """
    logger.info("Начало предобработки. Вход: %s", df.shape)

    df = df.copy()

    # 1. Stateless FE
    df = create_features(df)
    logger.info("Stateless FE завершён.")

    # 2. Stateful статистики из train
    df = apply_stateful_stats(df, stats)
    logger.info("Stateful статистики применены.")

    # 3. amenities -> текстовая строка (ПОСЛЕ amenities_count!)
    df["amenities"] = df["amenities"].apply(amenities_preprocessor)
    logger.info("amenities нормализован в текст.")

    # 4. Чистим суррогаты и приводим текстовые/категориальные к object
    #    (np.fromiter + isetitem — в обход Arrow-конвертации pandas)
    for col in cat_features + text_features:
        arr = np.fromiter(
            (_fix_surrogates(s) for s in df[col].tolist()),
            dtype=object,
            count=len(df),
        )
        df.isetitem(df.columns.get_loc(col), arr)

    # 5. Проверка наличия всех признаков
    missing = [col for col in all_features if col not in df.columns]
    if missing:
        logger.error("Отсутствуют колонки: %s", missing)
        raise ValueError(f"В данных нет необходимых признаков: {missing}")

    # 6. Финальный DataFrame в порядке all_features
    result = df[all_features].copy()

    logger.info("Предобработка завершена. Выход: %s", result.shape)
    return result
