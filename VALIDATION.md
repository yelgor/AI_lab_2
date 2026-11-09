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

## Дані, скачані через браузер

Додано import_local_data() і optional-клітинку LOCAL_DATA_SOURCE перед API-завантаженням. Підтримуються готові ZIP і розпаковані dataset. Для ZIP із Google Drive створюється локальна копія, а вихідний архів зберігається. Після розпакування notebook використовує CVLAB_DATA_ROOT та не звертається до Kaggle API.

Усі 32 перевірки downloader/importer пройшли під Python 3.13.17. Нові перевірки охоплюють збереження оригіналу ZIP і пропуск повторного копіювання готового dataset, папку без ZIP, архів уже в output, відхилення одиничного CSV, місце перед копіюванням, шляхи архіву та повний перехід від ручної клітинки до автоматичної без API. Ruff і синтаксис клітинок notebook перевірено. Реальні Google Drive mount та ZIP користувача в середовищі розробки недоступні.

## RAM під час збереження validation і відновлення

Замінено повний `json.dumps` перед записом на потоковий `json.dump`; atomic replacement та перевірка скінченних значень збережені. SHA-256 fingerprints обчислюються через `iterencode` з тими самими байтами, що й попередній алгоритм. CPU checkpoint після resume та predictions після завершення епохи більше не утримуються до наступної валідації.

Усі 59 тестів пройшли на Python 3.13.17 з CPU Torch 2.6.0/TorchVision 0.21.0. Нові перевірки контролюють додаткову пам'ять запису та хешування, сумісність fingerprints, збереження попереднього JSON при помилці, відновлення після збою між записом checkpoint і predictions та відтворення validation artifact через CLI. Відновлені CPU-ваги й scheduler збігаються з безперервним запуском. Ruff check пройшов.

Окреме вимірювання `tracemalloc` для 10 000 детекцій: додатковий пік Python-алокацій під час старого запису 8,16 MiB, потокового 0,062 MiB; файли по 1 710 055 байтів побайтно ідентичні. Вхідні дані створено до вимірювання. Це вимірювання витрат серіалізації, а не повного RSS тренування або COCO evaluation на VisDrone. Повний GPU-перезапуск перевіряється на машині користувача.

## Повторний OOM під час валідації

COCO evaluation розбито по класах і групах до 32 кадрів. IoU та об'єкти COCO обмежені поточною групою; для глобальної accumulation залишаються лише scores, булеві matches/ignore. Matching один раз для максимального cap використовується для обох AP100/AP500. Вхідні predictions не змінюються через `loadRes`.

Повний набір із 65 тестів пройшов на CPU. Precision-тензори побітно збігаються зі звичайним COCOeval окремо для caps 100/500 і груп розміру 1/3/32. Перевірено crowd regions, однакові scores, різні area ranges, порожні predictions, класи без GT та TP після rank 100. Окремі перевірки підтверджують обмеження COCO batch і відсутність зміни вхідних predictions.

Перед валідацією зберігається `last.pt` зі станом `validation_pending`. Тести змодельованого обриву під час inference та COCO metrics підтверджують: resume не викликає training loss повторно; кінцеві CPU-ваги й scheduler ідентичні безперервному запуску. Додатково перевірено валідацію pending checkpoint із уже вичерпаним бюджетом і відновлення старого checkpoint без нового поля.

Синтетичне вимірювання в окремих Linux-процесах, 256 кадрів / 128 000 детекцій: попередній evaluator — peak RSS 259,7 MiB, 8,39 с; новий — 144,3 MiB, 5,30 с. Обидва дали однакові AP100/AP500. Відтворюваний сценарій: `scripts/benchmark_coco_memory.py`. Це оцінка evaluator на синтетичних даних; повний RSS RetinaNet і VisDrone на користувацькому ноутбуці тут не виміряний.

## Локальна черга моделей

Для `scripts/local_gpu_queue.py` окремо пройшли три тести: послідовність моделей із відновленням last.pt та спільним протоколом; збереження наявної відмінної конфігурації без перезапису; блокування другої черги через реальний Linux flock. Запуск детекторів і очікування процесів у цих тестах замінено тестовими функціями; це перевірка orchestration, а не реального GPU throughput. Ruff check пройшов. Черга на машині користувача ще не запускалася.

## CPU-перевірка стану та submission

19 цільових тестів checker/queue пройшли. Перевірено CSV із правильними predictions, порожній структурно валідний submission, неправильні класи/scores, NaN, координати за межами кадру, вироджені boxes, неправильну довжину груп, cap>500, дублікати/відсутні/невідомі image_id. Smoke-run не зараховується як повний benchmark, а last.pt без метрик — як validated result. Великі prediction JSON не читаються; окремий Python-процес підтверджує імпорт чекера без Torch/NumPy. Додатково перевірено prediction-only режим черги без жодного training job. Ruff check пройшов. Самі ці тести не підтверджують Kaggle score або коректність tensors у checkpoint.
