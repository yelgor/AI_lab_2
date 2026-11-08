# Перевірка коду

19 автоматизованих перевірок пройшли на CPU. Ruff check також пройшов. Усі code cells notebook компілюються; конфігурації проходять CLI validation.

Перевірено:
- competition layout на синтетичних frames, валідні та некоректні labels;
- відсутність перетину clips між folds, повнота та неперетин OOF;
- класи 0–7 → internal 1–8 → submission 0–7;
- letterbox round-trip із непарними розмірами кадру;
- masking ignored regions зі збереженням target boxes;
- COCO crowd semantics, порожні predictions;
- порядок рядків CSV, clipping координат, максимум 500 детекцій;
- реальні forward/backward та відновлення ваг усіх чотирьох сімейств;
- RetinaNet L1/GIoU, FCOS GIoU+L1, RT-DETR зміна ваги L1;
- training → checkpoint → inference → submission;
- відновлення після змодельованого обриву: CPU-ваги ідентичні безперервному запуску;
- відновлення pretrained topology TorchVision без скачування: FrozenBatchNorm, eps і trainable layers. Значення downloaded pretrained weights у цих трьох тестах замінені локальними synthetic weights.

Середовище: Python 3.12.14, torch 2.6.0+cpu, torchvision 0.21.0+cpu, transformers 4.51.3. Є попередження про deprecated API у залежності matplotlib/pyparsing; падінь тестів немає.

Не перевірено: GPU/AMP performance, завантаження реальних pretrained weights, повне навчання на competition dataset, точний leaderboard scorer, leaderboard score. Даних і GPU тут немає. Не трактуйте synthetic tests як підтвердження якості детектора.

## Завантаження Kaggle data в Colab

Додано 8 offline-перевірок downloader та notebook-авторизації: успішне розпакування і кеш, вкладена структура, повторне використання ZIP, відсутність complete marker після збою, перевірка шляхів архіву й вільного місця, відхилення кешу іншого competition, приховане введення token без його появи в command arguments/output, пропуск авторизації для готових даних. Усі 8 пройшли; Ruff також пройшов. CLI syntax та auth звірені з офіційними docs Kaggle CLI 2.2.5.

Ці перевірки використовують локальні тестові ZIP та заміну мережевого виклику. Реальне завантаження 11 GB і перевірка прав приватного competition тут не виконувалися: потрібні credentials учасника.

## Python 3.13 / актуальний Colab

Виправлено обмеження Python: підтримується 3.11–3.13. Для сумісного встановлення та оцінювання pycocotools оновлено до 2.0.11; в optional dependencies Kaggle додано six 1.17.0, потрібний Kaggle CLI 2.2.5. Перша клітинка notebook виводить версію Python та повний stdout/stderr pip/git.

Перевірено в ізольованому Python 3.13.17: реальне `pip install -e '.[dev,kaggle]'` із CPU wheels Torch 2.6.0/TorchVision 0.21.0, `pip check`, `kaggle --version`, синтаксис notebook і передавання успішного та помилкового завершення bootstrap-команди. Усі 27 тестів пройшли на CPU, включно з forward/backward чотирьох сімейств, метриками та відновленням навчання. Додатково два тести метрик/OOF пройшли під Python 3.12.14 з pycocotools 2.0.11.

CUDA у Colab та приватне завантаження змагання перевірятимуться в користувацькому runtime; у середовищі розробки GPU й credentials Kaggle немає.

## Пошук mounted data і Kaggle Secrets

Виправлено пропуск API-завантаження в Kaggle без під'єднаного competition. Notebook спочатку використовує явний CVLAB_DATA_ROOT або змонтовані дані, потім готовий кеш, інакше завантажує через API. Додано читання KAGGLE_API_TOKEN через Kaggle Secrets разом із Colab Secrets і прихованим введенням. Реальні сигнали Colab мають пріоритет над випадково створеною папкою /kaggle. Пошук dataset показує відсутні папки; sample_submission.csv необов'язковий, а за його відсутності порядок submission береться з test_images.csv.

22 цільові перевірки пошуку/завантаження/авторизації/генерації CSV пройшли на Python 3.13.17; Ruff та синтаксис усіх клітинок notebook перевірено. Мережеві завантаження й доступ до service Secrets замінені тестовими відповідями; реальний competition dataset у цьому середовищі не доступний.

## Фактична помилка Kaggle CLI

Прибрано узагальнений wrapper про 401/403: downloader передає stdout/stderr CLI в notebook і зберігає код завершення та останні рядки справжньої помилки. CLI запускається через поточний Python. Додано check_access() для завантаження тільки малого test_images.csv у тимчасову папку.

25 перевірок downloader пройшли, зокрема нові перевірки streaming output, збереження конкретних 401/403/ModuleNotFoundError без заміни причини, приховування token/Authorization та запиту тільки CSV з очищенням тимчасових файлів. Реальний виклик `python -m kaggle.cli --version` перевірено під Python 3.13.17; SDK 2.2.5. Доступ до приватного competition без користувацьких credentials у цьому середовищі не перевірявся.
