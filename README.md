# AI_lab_2 — Drone Detection

[Відкрити notebook у Colab](https://colab.research.google.com/github/yelgor/AI_lab_2/blob/main/notebooks/quickstart.ipynb) · [Початок роботи](START_HERE.md) · [Перевірки](VALIDATION.md)

Експериментальний pipeline для детекції 8 класів у відеокадрах VisDrone2019-VID. Містить код і конфігурації для відтворюваних експериментів. Навчених competition weights та фінального звіту поки немає.

## Реалізовано

- Читання competition layout: `train|val/images/<clip>/*.jpg`, `labels/`, `ignore/`, `test_images.csv`.
- Перевірка labels, розмірів, класів і тестових ID; EDA та картинки ground truth.
- Три відтворювані folds за цілими clips; офіційний `val` можна зберегти для окремої перевірки.
- Faster R-CNN, RetinaNet, FCOS (TorchVision) і RT-DETR (Transformers), pretrained за замовчуванням.
- Спільний letterbox, horizontal flip для навчання, masking ignored regions.
- AdamW, окремий LR backbone, cosine schedule, gradient clipping, accumulation та FP16/BF16.
- `best.pt`, `last.pt`, продовження на межі епох, RNG states, split fingerprints, конфігурації, версії пакетів, JSONL-історія.
- COCO AP за класами/розмірами, cap=100 та додатковий cap=500, OOF aggregation.
- Час forward і повного inference-конвеєра після warmup, CUDA synchronization, peak allocated VRAM.
- Submission CSV: 0-based класи, вихідні пікселі, максимум 500 boxes.
- Конфігурації ablations: proposals, RoI sampling, assignment, queries і loss.

## Де запускати

Рекомендований старт — **Kaggle GPU**, оскільки dataset можна під'єднати без копіювання 11 GB. Увімкніть GPU та Internet для першого завантаження pretrained weights. Перевірте ресурси командою `doctor`: тип GPU, квоти й тривалість сесії залежать від акаунта та доступності.

Другий незалежний експеримент — **Colab GPU**. Перенесіть той самий пакет, дані та каталог `prepared/`, отриманий у Kaggle. Наприклад: RetinaNet/fold0 у Kaggle, FCOS/fold0 у Colab. Використовуйте ті самі manifests, folds і протокол; одного однакового seed недостатньо.

На одному невеликому GPU запускайте одне навчання. Два процеси конкурують за VRAM і compute. Якщо реально доступні два GPU, можна запустити незалежні процеси на `cuda:0` та `cuda:1`. Один процес тут використовує один GPU; DDP не реалізований. Обидва пристрої витрачають GPU-квоту.

Дані краще читати з локального диска runtime. Mounted Google Drive може сповільнювати читання тисяч JPEG. Зберігайте checkpoints у стійке сховище; для перенесення копіюйте весь run-каталог разом із `history.jsonl`.

## Встановлення

Python 3.11 або 3.12. Сумісна пара Torch/TorchVision 2.6.0/0.21.0 та Transformers 4.51.3 зафіксовані.

```bash
git clone https://github.com/yelgor/AI_lab_2.git
cd AI_lab_2
python -m pip install -e '.[dev]'
python -m cvlab doctor
```

Установка може змінити передвстановлений Torch. Після установки в notebook перезапустіть kernel перед імпортом Torch. Для власного сервера оберіть CUDA wheels, сумісні з драйвером, за офіційною інструкцією PyTorch.

Для CPU-перевірок:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
pytest -q
```

## Завантаження competition data в Colab

Notebook має окрему клітинку авторизації та завантаження для [multiple-object-video-detection-hw-2-ucu-ai-2026](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026). Прийміть правила змагання у своєму акаунті та отримайте API token на https://www.kaggle.com/settings/api. У Colab token вводиться в приховане поле або читається з Secret `KAGGLE_API_TOKEN`; він не записується до репозиторію. Дані розпаковуються до `/content/drone-data`, ZIP після успіху видаляється. Є перевірка місця перед розпакуванням та пропуск уже готового завантаження.

Поза notebook, якщо credentials Kaggle вже налаштовані:

```bash
python -m pip install -e '.[kaggle]'
python -m cvlab.kaggle_data --out /content/drone-data
```

Ключі не додавайте в командні аргументи чи Git. Повторне розпакування після обриву використовує атомарно збережені файли з очікуваним розміром. Реальне завантаження потребує ваших credentials і прав на competition; воно не виконувалося в середовищі розробки.

Джерела: [Kaggle authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication), [competition download](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md#kaggle-competitions-download).

## Перший запуск

Працюйте з кореня пакета. Підставте каталог, усередині якого є `train/`, `val/` і `test_images.csv`:

```bash
export CVLAB_DATA_ROOT=/kaggle/input/ACTUAL_DATASET_FOLDER
python -m cvlab prepare --root "$CVLAB_DATA_ROOT" --out prepared --folds 3 --workers 8
```

Перегляньте `prepared/eda_train/distributions.png`, `summary.json` і `gt_*.jpg`. Параметри 960 px, batch=2, accumulation=2 і `frame_step=1` — стартові гіпотези, а не доведений оптимум. Якщо бракує VRAM, спершу зменште batch до 1. Для порівняння моделей узгодьте протокол.

Коротка перевірка запуску: навчання на двох batches, валідація повна:

```bash
python -m cvlab train --config configs/fcos.yaml --set output=runs/fcos_smoke --set train.epochs=1 --set train.max_batches_per_epoch=2
```

Це не повноцінний benchmark. Новий запуск вимагає нового output-каталогу.

Baseline одного fold:

```bash
python -m cvlab train --config configs/fcos.yaml
```

За замовчуванням максимум 20 епох і м'який ліміт 2 wall-clock години після створення моделі. До часу входить валідація; остання валідація та запис checkpoint можуть перевищити ліміт. Зупинка відбувається на межі accumulation-групи, неповна епоха має `epoch_complete=false`. Підготовка даних і завантаження pretrained weights не входять у цей час. Неконвергований run потрібно так і описувати в аналізі.

Повна cross-validation однієї конфігурації:

```bash
for fold in 0 1 2; do
  python -m cvlab train --config configs/fcos.yaml --set data.fold=$fold --set output=runs/fcos_fold$fold
done
python -m cvlab oof --manifest prepared/train.json --out runs/fcos_oof.json runs/fcos_fold0/val_predictions.json runs/fcos_fold1/val_predictions.json runs/fcos_fold2/val_predictions.json
```

Спершу перевірте один run, потім оцініть реальний бюджет. Чотири моделі × три folds × 2 години — до приблизно 24 годин runtime без tuning і setup.

## Два незалежні запуски

Kaggle:

```bash
python -m cvlab train --config configs/retinanet.yaml
```

Colab, після копіювання тих самих `prepared/` і пакета:

```bash
export CVLAB_DATA_ROOT=/content/drone-data
python -m cvlab train --config configs/fcos.yaml
```

На сервері, тільки якщо підтверджено два GPU:

```bash
python -m cvlab train --config configs/retinanet.yaml --set runtime.device=cuda:0 > retina.log 2>&1 &
python -m cvlab train --config configs/fcos.yaml --set runtime.device=cuda:1 > fcos.log 2>&1 &
wait
```

Не перезаписуйте однакові run-каталоги при об'єднанні результатів. FPS різних GPU не є прямим порівнянням архітектур. Однаковий час навчання на різних GPU також не означає однаковий compute budget.

## Checkpoints, оцінювання і submission

```bash
python -m cvlab train --config configs/fcos.yaml --resume runs/fcos_fold0/last.pt
python -m cvlab evaluate --config configs/fcos.yaml --checkpoint runs/fcos_fold0/best.pt --out runs/fcos_fold0/recheck.json
python -m cvlab benchmark --config configs/fcos.yaml --checkpoint runs/fcos_fold0/best.pt --set loader.batch_size=1 --out runs/fcos_fold0/benchmark_batch1.json
```

Resume вимагає незмінного протоколу та `last.pt` у поточному output. Відновлюється збережена межа епохи; незбережені batches повторно не відновлюються. `persistent_workers=false` збережено для відтворення worker RNG між епохами. Побітова ідентичність на різному обладнанні не гарантується; strict deterministic режим опційний і може відхилити недетерміновану CUDA-операцію. Завантажуйте лише власні `.pt`.

Після вибору конфігурації можна навчити окремий run на всьому `train` із перевіркою на офіційному `val`:

```bash
python -m cvlab train --config configs/fcos.yaml --set data.split=official --set output=runs/fcos_official
python -m cvlab predict --config configs/fcos.yaml --set data.split=official --checkpoint runs/fcos_official/best.pt --out runs/submission.csv
```

CSV зберігає порядок `sample_submission.csv`: `class score x y width height`. Класи 0–7, координати у вихідних пікселях, порожні прогнози для кадрів без детекцій. Код відновлює letterbox, обрізає boxes і обмежує кількість до 500. Confidence threshold обирайте на validation. Файл автоматично не відправляється у змагання.

```bash
python -m cvlab summarize runs/fcos_fold0 runs/retinanet_fold0 --out runs/comparison.csv
```

`history.jsonl` містить mAP проти epoch/elapsed time, loss, LR і timing. `val_predictions.json` — прогнози найкращої епохи для аналізу помилок та OOF.

## Точний поточний протокол ignored regions і метрик

Під час навчання ignored regions замальовуються сірим; прямокутники відомих target boxes зберігаються. Це дозволений умовою masking baseline для всіх чотирьох моделей. Це **не loss-level ignore**: замальовані області можуть брати участь у background loss. Артефакти масок і background всередині збережених прямокутників — обмеження. Маски не застосовуються до validation/test, щоб не використовувати GT при inference. Loss-level ignore можна додати як окремий експеримент.

Локальні метрики використовують COCO crowd matching: безкласова ignored область представлена crowd-анотацією кожного з 8 класів. COCO спочатку зіставляє звичайні GT, а detections у crowd може ігнорувати. Цей локальний протокол **ще не звірений з leaderboard scorer**, якого нам не надали.

`coco100` — стандартний cap=100, `coco500` — додаткова метрика щільних сцен із cap=500. AP_small рахується у вихідних пікселях. Не називайте AP500 стандартним COCO AP100. OOF mAP на об'єднаних predictions відрізняється від середнього fold-mAP. Вибір best epoch на тих самих folds може давати оптимістичну оцінку; окремий офіційний `val` корисний для перевірки.

## Оптимізація

1. Перевірити EDA, GT і submission одного baseline.
2. Порівняти FP32/FP16 на тому самому checkpoint, перевіряючи AP та FPS.
3. Перебрати batch 1/2/4 і workers 0/2/4/8, контролюючи VRAM і pipeline FPS.
4. Порівняти чотири baseline на тих самих folds і бюджетах.
5. Запускати variants з `configs/` як окремі ablations, змінюючи один фактор.
6. Перевірити 640/960/1280 як окремі training-конфігурації: це зміна вартості та якості.
7. Опційно дослідити `model.freeze_backbone=true`: може прискорити навчання, але погіршити AP; inference не обов'язково прискориться.

Pinned memory, prefetch і non-blocking transfers підтримуються; фактичне перекриття копіювання з compute залежить від runtime. Accumulation зменшує VRAM для ефективного batch, але не є прискоренням і не відтворює BatchNorm великого batch. Mixed precision може змінювати результати.

Tiling/SAHI, tracking, ensembles, ONNX/TensorRT, DDP, W&B і автоматичний hyperparameter search ще не реалізовані. Додавати їх варто після вимірювання baseline. Паралельні незалежні навчання прискорюють серію експериментів, а не один inference.

## Здача роботи

Після отримання викладацького GitHub template пакет потрібно перенести до нього. README — інструкція запуску, не фінальний звіт. Ще потрібно виконати реальні навчання й tuning, аналіз failure cases, графіки порівняння, перевершити competition baseline, написати власний звіт і пройти захист. Для бонусу visualization потрібно додати W&B.

## Джерела для атрибуції

- TorchVision: https://pytorch.org/vision/0.21/models.html#object-detection
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- RetinaNet: https://arxiv.org/abs/1708.02002
- FCOS: https://arxiv.org/abs/1904.01355
- GIoU: https://arxiv.org/abs/1902.09630
- RT-DETR: https://arxiv.org/abs/2304.08069
- Transformers: https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/rt_detr
- COCO: https://github.com/cocodataset/cocoapi
- VisDrone: https://arxiv.org/abs/2001.06303
- PyTorch wheels: https://pytorch.org/get-started/previous-versions/

Дотримуйтеся ліцензій бібліотек, pretrained weights і competition data.
