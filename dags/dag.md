DAG: smartscore_batch_inference
Назначение:
Батч-инференс модели CatBoost для предсказания высокого рейтинга объявлений. DAG читает snapshot данных из PostgreSQL, выгружает его в S3, выполняет предсказание и сохраняет результат обратно в БД — в таблицу public.SmartScore_predict.

**Особенности:**
- Модель — CatBoost (сохранена в S3 в формате `.pkl`).
- `amenities` и `name` — текстовые признаки (`text_features`).
- Предобработка использует статистики из train (`preprocess_stats.pkl`):
медианы `bedrooms`/`beds` по `accommodates` и clip 99-го процентиля.
- При повторном запуске за ту же дату данные не дублируются.

Схема работы:

check_model
    │
    ▼
┌─────────── load_data ───────────┐
│  validate_source_data           │
│       │                         │
│       ▼                         │
│  extract_snapshot_to_s3         │
└─────────────────────────────────┘
    │
    ▼
┌─────────── inference ──────────┐
│  run_inference                 │
└────────────────────────────────┘
    │
    ▼
┌─────────── save_results ───────┐
│  save_predictions              │
└────────────────────────────────┘
Описание задач:

1. check_model
Проверяет наличие файла модели в S3.
Функция: check_model_exists (scripts/s3_utils.py)
Параметры: бакет S3, ключ модели (catboost_model.pkl), ID соединения S3.
Результат: если модель не найдена — задача падает, DAG останавливается.

2. load_data.validate_source_data
Проверяет наличие данных в исходной таблице PostgreSQL за дату инференса.
Функция: validate_source_data (scripts/db_utils.py)
Параметры: ID соединения PostgreSQL, имя исходной таблицы.
Результат: через XCom передаёт inference_date — дату, за которую есть данные.

3. load_data.extract_snapshot_to_s3
Выгружает snapshot из PostgreSQL в JSON-файл в S3.
Функция: extract_snapshot_to_s3 (scripts/db_utils.py)
Параметры: ID соединения PostgreSQL, исходная таблица, бакет S3, префикс (smartscore/input), ID соединения S3.
Результат: JSON-файл со snapshot данных в S3. Через XCom передаёт путь к файлу (input_s3_key).

4. inference.run_inference
Скачивает snapshot из S3, выполняет предобработку, загружает модель CatBoost, делает бинарное предсказание (0/1) по порогу threshold, сохраняет результат в JSON в S3.
Функция: run_inference (scripts/inference.py)
Параметры: бакет S3, ключ модели, выходной префикс (smartscore/output), порог (0.55), ID соединения S3.
Результат: JSON-файл с предсказаниями в S3. Через XCom передаёт путь к файлу (output_s3_key), бакет (s3_bucket) и ID соединения (s3_conn_id).

5. save_results.save_predictions
Скачивает JSON с предсказаниями из S3, проверяет, есть ли уже данные за эту дату в таблице public.SmartScore_predict, и если нет — вставляет новые строки.
Функция: save_predictions_to_db (scripts/db_utils.py)
Параметры: ID соединения PostgreSQL, целевая таблица (public.SmartScore_predict).
Результат: данные записаны в PostgreSQL. При повторном запуске за ту же дату — пропускает вставку.

Источники данных:

| Этап                      | Источник                                                     | Формат     |
| :-----------------------  | :----------------------------------------------------------- | :--------- |
|   Входные данные          | PostgreSQL, таблица `final_project.smartscore_test_snapshot` | Таблица БД |
|   Модель                  | S3, ключ `catboost_model.pkl`                                | Pickle     |
|   Промежуточный snapshot  | S3, префикс `smartscore/input`                               | JSON       |
|   Статистики train        | S3, `smartscore/artifacts/preprocess_stats.pkl`              | Pickle     |
|   Результат инференса     | S3, префикс `smartscore/output`                              | JSON       |


Место хранения артефактов:

| Артефакт             | Хранилище  | Путь                                                            |
| :------------------- | :--------- | :-------------------------------------------------------------- |
| Модель CatBoost      | S3         | {S3_BUCKET}/catboost_model.pkl                                  |
| Входной snapshot     | S3         | {S3_BUCKET}/smartscore/input/predictions_{inference_date}.json  |
| Статистики train     | S3         | `{S3_BUCKET}/smartscore/artifacts/preprocess_stats.pkl`         |
| Результат инференса  | S3         | {S3_BUCKET}/smartscore/output/predictions_{inference_date}.json |
| Итоговые предсказания| PostgreSQL | public.SmartScore_predict (колонки: id, score, inference_date)  |



Параметры DAG: 

| Параметр    | Значение                   | Источник     |
| :---------- | :------------------------- | :----------- |
| dag_id      | smartscore_batch_inference | Код DAG      |
| schedule    | None (ручной запуск)       | Код DAG      |
| catchup     | False                      | Код DAG      |
| start_date  | 2026-01-01                 | Код DAG      |
| tags        | smartscore, inference      | Код DAG      |
| retries     | 1                          | default_args |
| retry_delay | 5 минут                    | default_args |


Airflow Variables:


| Переменная              | Назначение                         | Значение по умолчанию                  |
| :---------------------- | :--------------------------------- | :------------------------------------- |
| smartscore_source_table | Исходная таблица в PostgreSQL      | final_project.smartscore_test_snapshot |
| smartscore_target_table | Целевая таблица в PostgreSQL       | public.SmartScore_predict              |
| smartscore_s3_bucket    | Имя бакета S3                      | (обязательно)                          |
| smartscore_model_key    | Ключ модели в S3                   | catboost_model.pkl                     |
| smartscore_input_prefix | Префикс входных данных в S3        | smartscore/input                       |
| smartscore_output_prefix| Префикс результатов в S3           | smartscore/output                      |
| smartscore_threshold    | Порог бинарной классификации       | 0.55                                   |
| smartscore_pg_conn_id   | ID соединения PostgreSQL в Airflow | postgres_smartscore                    |
| smartscore_s3_conn_id   | ID соединения S3 в Airflow         | yandex_s3                              |


Airflow Connections:

| Connection ID       | Тип                 | Назначение                               |
| :------------------ | :------------------ | :--------------------------------------- |
| postgres_smartscore | PostgreSQL          | Чтение исходных данных, запись результатов |
| yandex_s3           | Amazon Web Services | Доступ к S3 (Yandex Object Storage)      |


Все задачи используют logging.info() для отслеживания процесса:

validate_source_data — логирует найденную дату инференса и количество строк.
extract_snapshot_to_s3 — логирует путь к выгруженному файлу и количество записей.
run_inference — логирует загрузку модели, размер snapshot, количество предсказаний и путь к результату.
save_predictions — логирует количество скачанных предсказаний, дату инференса, наличие/отсутствие данных за эту дату в БД, прогресс вставки батчами и итоговое количество строк.
