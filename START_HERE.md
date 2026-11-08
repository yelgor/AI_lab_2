# Початок роботи

Код розміщений у https://github.com/yelgor/AI_lab_2.

## Kaggle — рекомендований перший запуск

1. Створіть Kaggle Notebook, під'єднайте competition dataset, увімкніть GPU та Internet.
2. Імпортуйте [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) з цього репозиторію до Kaggle або скопіюйте його клітинки. Можна скачати notebook на GitHub через кнопку Download raw file.
3. Перша клітинка сама клонує `yelgor/AI_lab_2` у `/kaggle/working/AI_lab_2` і встановить залежності. ZIP не потрібен.
4. Клітинка `DATA_ROOT` автоматично знайде dataset, якщо під'єднано рівно один каталог із `test_images.csv`. Інакше вкажіть потрібний корінь вручну.
5. Виконайте doctor → prepare → перегляд EDA. Після установки перезапустіть kernel за потреби, як вказано в notebook.
6. Спочатку залиште `MODEL='fcos'`, `FOLD=0`, `SMOKE=True`. Це навчання на двох batches із повною валідацією. Потім `SMOKE=False` для повного run в іншому output-каталозі.

## Colab і паралельні експерименти

[Відкрити quickstart у Colab](https://colab.research.google.com/github/yelgor/AI_lab_2/blob/main/notebooks/quickstart.ipynb).

Перша клітинка клонує або оновлює репозиторій у `/content/AI_lab_2` та встановлює downloader. Завантаження вже налаштоване для [вашого змагання](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026):

1. Прийміть правила змагання у своєму Kaggle-акаунті.
2. На https://www.kaggle.com/settings/api натисніть **Generate New Token**.
3. Виконайте нову клітинку **«Дані змагання в Colab»** і вставте token у приховане поле. Також підтримуються Colab Secrets з назвою `KAGGLE_API_TOKEN` або вже налаштовані credentials Kaggle.
4. Клітинка завантажить дані до `/content/drone-data`, розпакує та видалить ZIP після успішного завершення. `DATA_ROOT` визначиться автоматично; повторний запуск пропустить готове завантаження.
5. Далі виконайте prepare → EDA → короткий запуск із `SMOKE=True`.

Токен вводиться лише у Colab; не вставляйте його в код або чат. При HTTP 401 перевірте токен; при HTTP 403 перевірте участь і прийняття правил тим самим акаунтом.

Якщо Colab вже відкритий зі старою копією notebook, відкрийте посилання вище заново. Оновлена перша клітинка виконує `git pull --ff-only` для існуючого clone; локальні зміни примусово не перезаписуються.

Для двох паралельних навчань запустіть різні моделі в незалежних GPU-середовищах: наприклад FCOS у Kaggle, RetinaNet у Colab. Використовуйте той самий каталог `prepared/`, folds і training protocol. Доступність GPU та квоти перевіряйте у своїх акаунтах. Швидкість різних архітектур порівнюйте на однаковому обладнанні.

Якщо доступні два GPU на одному сервері, README містить команди для окремих процесів на `cuda:0` і `cuda:1`. На одному невеликому GPU два навчання можуть сповільнювати одне одного.

## Після першого запуску

Збережіть весь каталог run з checkpoints, history і predictions у стійке сховище до завершення сесії. Після вибору конфігурації потрібні повні folds/OOF, аналіз помилок і генерація submission.

Параметри оптимізації: FP16/BF16, batch, accumulation, CPU workers, prefetch, pinned memory, freeze_backbone. Їхній вплив на AP і швидкість потрібно виміряти на реальному GPU.

Повні команди, протокол ignored regions, метрик та checkpoints — у [README.md](README.md). Результати локальних перевірок — у [VALIDATION.md](VALIDATION.md). Реального навчання на competition dataset у поточному середовищі не було.
