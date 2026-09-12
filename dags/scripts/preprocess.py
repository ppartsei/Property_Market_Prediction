# scripts/preprocess.py

import re
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


# ─── Фиксированный список топ-50 удобств ──────────────────────────────────────
# Порядок важен: он должен совпадать с порядком при обучении модели.

TOP_50_AMENITIES = [
    "Wifi",
    "Kitchen",
    "Smoke alarm",
    "Hot water",
    "Hair dryer",
    "Dishes and silverware",
    "Bed linens",
    "Hangers",
    "Iron",
    "Essentials",
    "Cooking basics",
    "Refrigerator",
    "Microwave",
    "Carbon monoxide alarm",
    "Hot water kettle",
    "Shampoo",
    "Dedicated workspace",
    "Dining table",
    "TV",
    "Heating",
    "Freezer",
    "Self check-in",
    "Washer",
    "Cleaning products",
    "Toaster",
    "Wine glasses",
    "Shower gel",
    "Oven",
    "Long term stays allowed",
    "Dishwasher",
    "Body soap",
    "Extra pillows and blankets",
    "Fire extinguisher",
    "Coffee",
    "First aid kit",
    "Room-darkening shades",
    "Bathtub",
    "Stove",
    "Coffee maker",
    "Baking sheet",
    "Drying rack for clothing",
    "Private entrance",
    "Lockbox",
    "Conditioner",
    "Central heating",
    "Clothing storage",
    "Luggage dropoff allowed",
    "Exterior security cameras on property",
    "Air conditioning",
    "Laundromat nearby",
]

TOP_50_SET = set(TOP_50_AMENITIES)


# ─── Списки признаков ─────────────────────────────────────────────────────────

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
    "amenities_count",
    "host_trust_score",
    "guests_per_bedroom",
]


binary_features = [
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "has_email_verification",
    "has_work_email_verification",
    "is_shared_bathroom",
]


# Генерируем имена amenity_* по тому же правилу,
# которое использовалось при обучении.
amenity_features = [
    f'amenity_{amenity.lower().replace(" ", "_").replace("-", "_").replace("/", "_")}'
    for amenity in TOP_50_AMENITIES
]


text_features = ["name"]


# has_phone_verification является категориальным признаком
# и поэтому находится здесь отдельно от binary_features.
cat_features = (
    [
        "property_type",
        "neighbourhood_cleansed",
        "room_type",
        "has_phone_verification",
    ]
    + binary_features
    + amenity_features
)


# Полный набор признаков модели.
all_features = num_features + cat_features + text_features


# ─── Вспомогательные функции ──────────────────────────────────────────────────


def _parse_amenities(x):
    """
    Преобразует строковое представление списка amenities
    в Python list.
    """
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)

        if isinstance(x, list):
            return x

        return []

    except (ValueError, SyntaxError, TypeError):
        return []


def _extract_bathrooms(text):
    """
    Извлекает количество санузлов из bathrooms_text.

    half-bath -> 0.5
    иначе используется первое найденное число.
    """
    if pd.isna(text):
        return None

    text = str(text).lower()

    if "half-bath" in text:
        return 0.5

    match = re.search(r"([\d.]+)", text)

    if match:
        return float(match.group(1))

    return None


def _to_binary(value):
    """
    Приводит boolean-признак к 0/1.

    Поддерживает:
    - True / False
    - 'True' / 'False'
    - 1 / 0
    - NaN

    Это необходимо, поскольку при обучении два признака
    были object, а в SmartScore_test_snapshot они имеют
    PostgreSQL boolean.
    """
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        value = value.strip().lower()

        if value == "true":
            return 1

        if value == "false":
            return 0

    if value in (1, 1.0):
        return 1

    if value in (0, 0.0):
        return 0

    return 0


def _parse_verifications(x):
    """
    Преобразует host_verifications в список.
    """
    try:
        if isinstance(x, str):
            result = ast.literal_eval(x)
            return result if isinstance(result, list) else []

        if isinstance(x, list):
            return x

        return []

    except (ValueError, SyntaxError, TypeError):
        return []


# ─── Основная предобработка ───────────────────────────────────────────────────


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полная предобработка данных для инференса.

    Повторяет feature engineering, использованный при обучении
    модели, и возвращает только признаки из all_features.
    """

    logger.info(f"Начало предобработки. Вход: {df.shape}")

    df = df.copy()

    # ─────────────────────────────────────────────────────────
    # 1. Генерация amenity_*
    # ─────────────────────────────────────────────────────────

    df["amenities_list"] = df["amenities"].apply(_parse_amenities)

    df["amenities_top50"] = df["amenities_list"].apply(
        lambda x: [amenity for amenity in x if amenity in TOP_50_SET]
    )

    mlb = MultiLabelBinarizer(classes=TOP_50_AMENITIES)

    amenities_encoded = mlb.fit_transform(df["amenities_top50"])

    amenities_df = pd.DataFrame(
        amenities_encoded,
        columns=amenity_features,
        index=df.index,
    )

    df = pd.concat(
        [df, amenities_df],
        axis=1,
    )

    df.drop(
        ["amenities_list", "amenities_top50"],
        axis=1,
        errors="ignore",
        inplace=True,
    )

    # ─────────────────────────────────────────────────────────
    # 2. bathrooms
    # ─────────────────────────────────────────────────────────

    df["bathrooms_from_text"] = df["bathrooms_text"].apply(_extract_bathrooms)

    mask = df["bathrooms"].isnull() & df["bathrooms_from_text"].notna()

    df.loc[mask, "bathrooms"] = df.loc[mask, "bathrooms_from_text"]

    # ─────────────────────────────────────────────────────────
    # 3. Признаки верификации
    # ─────────────────────────────────────────────────────────

    verification_cols = [
        "email",
        "phone",
        "work_email",
    ]

    for verification in verification_cols:

        df[f"has_{verification}_verification"] = df["host_verifications"].apply(
            lambda x: int(verification in _parse_verifications(x))
        )

    # ─────────────────────────────────────────────────────────
    # 4. Общий санузел
    # ─────────────────────────────────────────────────────────

    df["is_shared_bathroom"] = (
        df["bathrooms_text"]
        .str.contains(
            "shared",
            case=False,
            na=False,
        )
        .astype(int)
    )

    # ─────────────────────────────────────────────────────────
    # 5. Количество удобств
    # ─────────────────────────────────────────────────────────

    df["amenities_count"] = df["amenities"].apply(
        lambda x: (x.count(",") + 1 if x != "[]" else 0)
    )

    # ─────────────────────────────────────────────────────────
    # 6. Индекс доверия хоста
    # ─────────────────────────────────────────────────────────

    df["host_trust_score"] = (
        df["has_email_verification"]
        + df["has_phone_verification"]
        + df["has_work_email_verification"]
        + df["host_identity_verified"].apply(_to_binary)
        + df["host_has_profile_pic"].apply(_to_binary)
    )

    # ─────────────────────────────────────────────────────────
    # 7. Плотность заселения
    # ─────────────────────────────────────────────────────────

    df["guests_per_bedroom"] = np.where(
        df["bedrooms"] > 0,
        df["accommodates"] / df["bedrooms"],
        0,
    )

    # ─────────────────────────────────────────────────────────
    # 8. Конвертация boolean-признаков в int
    # ─────────────────────────────────────────────────────────

    bool_cols = [
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]

    for column in bool_cols:
        df[column] = df[column].apply(_to_binary)

    # ─────────────────────────────────────────────────────────
    # 9. Проверка наличия всех признаков модели
    # ─────────────────────────────────────────────────────────

    missing = [col for col in all_features if col not in df.columns]

    if missing:
        logger.error(f"Отсутствуют колонки: {missing}")
        raise ValueError(f"В данных нет необходимых признаков: {missing}")

    # ─────────────────────────────────────────────────────────
    # 10. Финальный DataFrame
    # ─────────────────────────────────────────────────────────

    result = df[all_features].copy()

    # Приводим категориальные и текстовые признаки к строкам
    for col in text_features + cat_features:
        result[col] = result[col].astype(str)

    logger.info(f"Предобработка завершена. " f"Выход: {result.shape}")

    return result
