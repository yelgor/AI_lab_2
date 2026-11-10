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

Python 3.11, 3.12 або 3.13. Сумісна пара Torch/TorchVision 2.6.0/0.21.0 та Transformers 4.51.3 зафіксовані.

```bash
git clone https://github.com/yelgor/AI_lab_2.git
cd AI_lab_2
python -m pip install -e '.[dev]'
python -m cvlab doctor
```

Установка може змінити передвстановлений Torch. Після першого встановлення в notebook перезапустіть Python-сеанс і продовжуйте з клітинки `doctor`. Для власного сервера оберіть CUDA wheels, сумісні з драйвером, за офіційною інструкцією PyTorch.

Для CPU-перевірок:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
pytest -q
```

## Готовий ZIP або розпаковані дані

У notebook є optional-клітинка `LOCAL_DATA_SOURCE` перед API-завантаженням. Якщо ZIP уже скачаний через браузер, завантажте його до свого Google Drive та вкажіть повний шлях у цій клітинці. Код під'єднає Drive, збереже вихідний ZIP і розпакує локальну копію. Також підтримуються локальні ZIP і розпаковані папки.

Після встановлення пакета можна використати helper безпосередньо:

```python
from cvlab.kaggle_data import import_local_data
import os
root = import_local_data('/content/drive/MyDrive/AI_lab_2_data/DATASET.zip')
os.environ['CVLAB_DATA_ROOT'] = str(root)
```

У цьому режимі Kaggle API та token не потрібні. Далі виконуються звичайні prepare, навчання та inference.

## Пошук і завантаження competition data

Notebook має клітинку пошуку даних, авторизації та завантаження для [multiple-object-video-detection-hw-2-ucu-ai-2026](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026). Прийміть правила змагання у своєму акаунті та отримайте API token на https://www.kaggle.com/settings/api. У Colab token читається з Secret `KAGGLE_API_TOKEN`, у Kaggle — через Kaggle Secrets; також підтримується приховане введення. Ключ не записується до репозиторію. Під’єднані дані використовуються напряму; за їх відсутності дані завантажуються до `/content/drone-data` у Colab або `/kaggle/working/drone-data` у Kaggle. ZIP після успіху видаляється. Є перевірка місця перед розпакуванням та пропуск уже готового завантаження.

Поза notebook, якщо credentials Kaggle вже налаштовані:

```bash
python -m pip install -e '.[kaggle]'
python -m cvlab.kaggle_data --out /content/drone-data
```

Ключі не додавайте в командні аргументи чи Git. Повторне розпакування після обриву використовує атомарно збережені файли з очікуваним розміром. Реальне завантаження потребує ваших credentials і прав на competition; воно не виконувалося в середовищі розробки.

Джерела: [Kaggle authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication), [competition download](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md#kaggle-competitions-download).

Для діагностики помилки завантаження в notebook запустіть `kaggle_data.check_access()`: буде запитано тільки невеликий `test_images.csv`, а не весь dataset. CLI виконується тим самим Python; його фактичний stdout/stderr і return code доступні в notebook з прихованими token/Authorization. У терміналі з налаштованими credentials: `python -m cvlab.kaggle_data --check-access`.

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

### Черга на одному локальному GPU

`scripts/local_gpu_queue.py` запускає моделі послідовно, використовуючи локальне середовище та вже підготовлені дані. Він бере data/loader/runtime, seed, роздільність та бюджет із `WORK/configs/retinanet_fold0_local.yaml`, а архітектуру й відповідні model settings — із конфігурації кожної моделі. Наявні конфігурації з іншими налаштуваннями не перезаписуються. Один lock захищає від дублювання черги.

```bash
.venv/bin/python -u scripts/local_gpu_queue.py \
  --work "$CVLAB_LOCAL_WORK" --wait-pid "$CVLAB_WAIT_PID" \
  --predict-retina --models fcos fasterrcnn rtdetr
```

Перед запуском задайте `CVLAB_LOCAL_WORK` як шлях до робочого диска, а `CVLAB_WAIT_PID` — PID поточного launcher, який охоплює training і подальше evaluate. Черга дочекається його завершення та відсутності інших CVLab jobs. З `--predict-retina` спочатку генерується початковий `submissions/retinanet_fold0.csv`, потім запускаються вибрані моделі. Checkpoints відновлюються автоматично; відсутній validation artifact відтворюється з best.pt. Скрипт сам не завантажує дані й не надсилає submission до Kaggle. Для фонового виконання використовуйте nohup і systemd-inhibit, як для поточного локального навчання.

Для генерації лише першого submission, без нових training jobs, передайте порожній список `--models`:

```bash
.venv/bin/python -u scripts/local_gpu_queue.py \
  --work "$CVLAB_LOCAL_WORK" --wait-pid "$CVLAB_WAIT_PID" --predict-retina --models
```

### Перевірка артефактів паралельно з навчанням

`scripts/check_lab.py` працює на CPU без імпорту Torch/NumPy та без завантаження tensors. Читаються лише невеликі metadata; великі manifests і validation predictions не завантажуються. Скрипт показує вільну RAM/VRAM, наявні результати моделей, виміряні folds, обмежені smoke-runs і перетин sequences зі split.json. Checkpoint перевіряється лише на наявність ZIP-контейнера; його tensors і фаза без завантаження ваг не підтверджуються. Наявність folds або predictions-файлу не означає перевіреного OOF.

```bash
.venv/bin/python scripts/check_lab.py --work "$CVLAB_LOCAL_WORK"
.venv/bin/python scripts/check_lab.py --work "$CVLAB_LOCAL_WORK" \
  --submission "$CVLAB_LOCAL_WORK/submissions/retinanet_fold0.csv"
```

JSON зберігається у `WORK/checks/status.json`. CSV перевіряється потоково: точний набір image_id, дублікати, шість полів на детекцію, класи 0–7, скінченні scores/координати, межі оригінального кадру, позитивні розміри та cap=500. Порожній submission структурно валідний, але окремо позначається `all_predictions_empty`. Перевищення Kaggle baseline, узгодженість повного CV/OOF, аналіз експериментів та звіт потребують окремої перевірки; чекер не оцінює готовність до здачі автоматично.

## Checkpoints, оцінювання і submission

```bash
python -m cvlab train --config configs/fcos.yaml --resume runs/fcos_fold0/last.pt
python -m cvlab evaluate --config configs/fcos.yaml --checkpoint runs/fcos_fold0/best.pt --out runs/fcos_fold0/recheck.json
python -m cvlab benchmark --config configs/fcos.yaml --checkpoint runs/fcos_fold0/best.pt --set loader.batch_size=1 --out runs/fcos_fold0/benchmark_batch1.json
```

Resume вимагає незмінного протоколу та `last.pt` у поточному output. Відновлюється збережена межа епохи; незбережені batches повторно не відновлюються. `persistent_workers=false` збережено для відтворення worker RNG між епохами. Побітова ідентичність на різному обладнанні не гарантується; strict deterministic режим опційний і може відхилити недетерміновану CUDA-операцію. Завантажуйте лише власні `.pt`.

`last.pt` тепер записується одразу після навчальної частини епохи з полем `validation_pending`, а після успішної валідації оновлюється завершеним станом. Якщо збій стався під час inference або COCO evaluation, resume повторює валідацію, не повторюючи вже збережені training batches і крок scheduler. Це працює й після вичерпання часового бюджету: незавершена валідація повинна завершитися. Старі checkpoints без цього поля також підтримуються; вони відновлюються за попередніми правилами. `best.pt` оновлюється лише після отримання метрик.

Якщо Linux завершив Python через RAM OOM, а `last.pt` збережений, використайте resume з тією самою конфігурацією. Накопичений час бюджету відновлюється з checkpoint. Зменшення batch або зміна workers змінюють протокол і відхиляються під час strict resume; спочатку звільніть RAM від зайвих програм.

JSON записується потоково в тимчасовий файл із подальшим atomic rename. Хеші manifests також обчислюються потоково зі збереженням попередніх fingerprints. Після валідації масив predictions звільняється перед наступною епохою. Повні predictions і COCO evaluation усе ще потребують RAM.

COCO matching виконується окремо для кожного класу групами до 32 кадрів. Зберігаються компактні булеві matches/ignore та confidence scores; AP накопичується глобально за всіма кадрами. Результати matching при cap=500 повторно використовуються для cap=100. Це зберігає стандартні COCO AP і порядок detections при однакових scores. `loadRes` більше не додає службові поля до всього вхідного масиву predictions. Розмір повних predictions і глобальних компактних matches усе ще залежить від кількості детекцій.

Синтетичне порівняння RAM та часу в окремих Linux-процесах:

```bash
python scripts/benchmark_coco_memory.py previous
python scripts/benchmark_coco_memory.py batched
```

Checkpoints записуються до `val_predictions.json`. Якщо процес обірвався при записі JSON, після відновлення навчання відтворіть відсутній або незавершений artifact із `best.pt`:

```bash
if [ -e runs/fcos_fold0/val_predictions.json.tmp ] || [ ! -s runs/fcos_fold0/val_predictions.json ]; then
  python -m cvlab evaluate --config configs/fcos.yaml --checkpoint runs/fcos_fold0/best.pt --out runs/fcos_fold0/val_predictions.json
fi
```

Цей крок повторює inference і підрахунок метрик validation для найкращих ваг. Дані повторно готувати не потрібно.

Після вибору конфігурації можна навчити окремий run на всьому `train` із перевіркою на офіційному `val`:

```bash
python -m cvlab train --config configs/fcos.yaml --set data.split=official --set output=runs/fcos_official
python -m cvlab predict --config configs/fcos.yaml --set data.split=official --checkpoint runs/fcos_official/best.pt --out runs/submission.csv
```

CSV зберігає порядок `sample_submission.csv`, а якщо шаблону немає — порядок `test_images.csv`: `class score x y width height`. Класи 0–7, координати у вихідних пікселях, порожні прогнози для кадрів без детекцій. Код відновлює letterbox, обрізає boxes і обмежує кількість до 500. Confidence threshold обирайте на validation. Файл автоматично не відправляється у змагання.

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

### Роздільність інференсу без перенавчання

Для Faster R-CNN, RetinaNet та FCOS можна перевіряти інший `model.image_size` із тим самим checkpoint. Ваги, backbone topology й losses не змінюються; розмір застосовується і до letterbox, і до внутрішнього transform детектора. RT-DETR також підтримує зміну розміру, якщо збережений `adapter.hf_config` явно містить `eval_size=None` та `anchor_image_size=None`: позиційні кодування й anchors тоді будуються з поточних feature shapes. Відсутня metadata або фіксовані розміри залишають resize забороненим. Кількість queries та інші параметри архітектури змінювати під час відновлення не можна.

```bash
python -u -m cvlab evaluate-resolutions \
  --config "$CVLAB_LOCAL_WORK/configs/retinanet_fold0_local.yaml" \
  --checkpoint "$CVLAB_LOCAL_WORK/runs/retinanet_fold0_local/best.pt" \
  --sizes 960 1280 --max-images 192 --wait-gpu \
  --out "$CVLAB_LOCAL_WORK/experiments/retinanet_resolution_probe.json"
```

Це експрес-порівняння на одних і тих самих validation frames, рівномірно вибраних усередині sequences. JSON містить IDs, fingerprint вибірки, SHA-256 checkpoint, AP та timing кожного розміру. Результат subset не є повним validation mAP або OOF. Для повного validation вкажіть `--max-images` не менше кількості кадрів fold. Покращення роздільності потрібно підтверджувати метриками; воно не гарантується.

`--wait-gpu` чекає завершення поточних CVLab jobs до імпорту Torch та створення моделей. Враховуються фактичні Python module CLI-процеси, включно з `train-overlap`; обгортки `systemd-inhibit` та shell не зараховуються як окремі jobs. Для активного `train-overlap` проба чекає завершення всієї черги. Глобальний lock користувача блокує дублювання таких очікувачів, зокрема з різними output-каталогами.

### Навчання на GPU паралельно з COCO-метриками на CPU

`scripts/start_overlap_queue.py` передає керування режиму `train-overlap`: після GPU-інференсу копіює стан моделі/optimizer/RNG на CPU та запускає один CPU worker для COCO і збереження результатів. GPU тим часом навчає наступну епоху або наступну модель. CPU worker не зберігає посилань на живі GPU-ваги. Перед наступним набором validation predictions попереднє CPU-завдання завершується; це обмежує RAM одним набором прогнозів. При доступній RAM менше 4 GiB pipeline теж чекає завершення CPU-завдання.

```bash
python -u scripts/start_overlap_queue.py \
  --work "$CVLAB_LOCAL_WORK" --take-over \
  --handoff-run "$CVLAB_LOCAL_WORK/runs/fcos_fold0_local" \
  --handoff-log "$CVLAB_LOCAL_WORK/logs/remaining_models_resume.log" \
  --models fcos fasterrcnn rtdetr
```

`--take-over` завершує лише Python-процес старої черги, залишаючи його активний trainer працювати. Пара `--handoff-run/--handoff-log` чекає нового повідомлення `epoch=... loss=... mAP=...`, яке старий код друкує після запису checkpoint та validation artifacts, і завершує trainer на цій границі. Новий процес підхоплює `last.pt`; завершені епохи зберігаються. Кілька щойно розпочатих, ще незбережених батчів наступної епохи можуть повторитися. Без цих двох параметрів launcher чекає завершення всього активного запуску. Якщо стара черга вже перейшла до іншої моделі, launcher не завершує її trainer за помилково вказаним run.

Глобальний для work-каталогу queue lock зберігається. Активний режим має CLI-префікс `train`, тому інші CVLab jobs розпізнають його. DataLoader використовує `spawn`, щоб не робити fork під час роботи CPU thread. Кількість workers, seed, batch, модель та optimizer config зберігаються. Файл `execution.json` позначає режим виконання. Бюджет залишається wall-clock; перекриття може дозволити більше навчання за той самий час. Потрібно виміряти фактичний час і пам'ять на GPU користувача; цей режим не гарантує одночасної роботи двох моделей на GPU.

### CPU-експеримент паралельно з GPU

Збережений `val_predictions.json` дозволяє порівняти додатковий class-specific NMS без нового inference або завантаження моделі. Цей модуль не імпортує PyTorch. Manifest і прогнози читаються частинами; у RAM залишаються розмітка та не більше 500 детекцій на кожен вибраний кадр. Перевіряються IDs повного validation, fingerprint даних і конфігурація моделі. JSON результату містить також SHA-256 файла прогнозів.

```bash
nice -n 15 ionice -c 3 env CUDA_VISIBLE_DEVICES="" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m cvlab.postprocess_experiments \
  --run "$CVLAB_LOCAL_WORK/runs/retinanet_fold0_local" \
  --max-images 192 --nms 0.3 0.4 0.5 \
  --out "$CVLAB_LOCAL_WORK/experiments/retinanet_nms_cpu.json"
```

Спочатку оцінюється вихідний набір детекцій, потім кожен NMS на тих самих кадрах. Це subset-порівняння: кандидата потрібно підтвердити на повному validation та інших folds. Додатковий NMS може лише видаляти вже збережені бокси; він не відновлює бокси, які модель відкинула під час inference. Такий експеримент не замінює навчання інших архітектур або обов'язкові зміни loss. Запуск пише окремий JSON і не змінює checkpoints, вихідні прогнози чи submission.

### Submission на CPU, поки GPU навчає іншу модель

```bash
CUDA_VISIBLE_DEVICES="" python -u -m cvlab.submit_cpu \
  --run "$CVLAB_LOCAL_WORK/runs/fasterrcnn_fold0_local" \
  --out "$CVLAB_LOCAL_WORK/submissions/fasterrcnn_fold0_cpu.csv" \
  --threads 4 --chunk-size 16
```

Модуль копіює поточний `best.pt` у тимчасовий каталог на work-диску та генерує прогнози з цим незмінним snapshot. Використовуються FP32 на CPU, batch 1 та workers 0. Вхідний розмір, класи, пороги та NMS зберігаються. Прогнози тримаються лише для поточної порції кадрів; CSV записується поступово, у порядку `sample_submission.csv`. Прогрес містить оброблені кадри, швидкість та ETA. Фінальний `.csv` з'являється після перевірки IDs, 0-based класів, координат, confidence та ліміту 500 детекцій. Порожній для всіх кадрів результат не публікується. Metadata JSON містить SHA-256 frozen checkpoint та CSV, fingerprint test і конфігурацію inference. Наявний фінальний CSV зберігається, дублювання writers блокується lock. Це CPU-альтернатива з вимірюваним часом; вона не обіцяє швидкості GPU або конкретного leaderboard score.

### Submission на GPU з окремим бюджетом VRAM

Коли поточне навчання залишає достатньо VRAM, можна спробувати паралельний FP16-інференс:

```bash
python -u -m cvlab predict-submission \
  --run "$CVLAB_LOCAL_WORK/runs/fasterrcnn_fold0_local" \
  --out "$CVLAB_LOCAL_WORK/submissions/fasterrcnn_fold0_gpu.csv" \
  --device cuda --vram-gib 1.5 --threads 2 --chunk-size 16
```

Використовуються batch 1 і workers 0, з тими самими правилами frozen checkpoint, потокового CSV та фінальної перевірки. Перед моделлю перевіряються доступні RAM (щонайменше 2 GiB) і VRAM (вказаний бюджет плюс 512 MiB), після чого `torch.cuda.set_per_process_memory_fraction` обмежує allocator лише нового exporter-процесу. CUDA context та сторонні CUDA allocations можуть додавати пам'ять поза цим лімітом. Це не резервує VRAM і не встановлює ліміт іншому trainer; його майбутній пік і спільні GPU compute впливають на результат. Швидкість і ETA виводяться після кожної порції; паралельний exporter може сповільнити навчання. CLI має prefix `predict`, тому очікувачі GPU бачать його. CPU-варіант можна залишити незалежним резервом до готового GPU CSV.

### Кандидати для перевищення Kaggle baseline

Ціль цього змагання — перевищити `0.2216324580` на Kaggle. Наступний сценарій спочатку чекає GPU lease поточної черги, а потім працює на GPU з готовими `best.pt`. Поточне навчання не переривається.

```bash
python -u scripts/start_score_improvement.py \
  --work "$CVLAB_LOCAL_WORK" --models rtdetr fasterrcnn \
  --sizes 960 1280 1536 --max-images 192 --vram-gib 6
```

Спочатку створюються native-resolution CSV обох моделей. Повідомлення `FIRST CANDIDATE READY FOR KAGGLE` означає, що файл уже можна завантажити, поки пошук триває. Копії checkpoint заморожуються з SHA-256 у `score_improvement/`; початкові checkpoints не змінюються. Кожен CSV перевіряється на відповідність тестовим IDs, координатам, класам і ліміту детекцій.

Далі для кожної моделі порівнюються 960/1280/1536 на тих самих 192 кадрах **офіційної val**, яка не використовувалася в навчанні fold0. Перетин training sequences з цим holdout забороняється; fingerprint source split має відповідати checkpoint. Якщо більший розмір перемагає, обидва розміри вимірюються на **всій official val**. Лише після цього зберігається CSV обраного розміру. Для однакового AP обирається менший розмір. Розмір, який не вміщується у GPU budget, пропускається; інші CUDA errors не приховуються.

Результати знаходяться в `score_improvement/summary.json`; готові CSV — в `submissions/<model>_<checkpoint-sha>_<size>.csv`. Повний validation кешується з IDs та fingerprint. Ці official-val scores не замінюють попередні fold0/OOF scores і не порівнюються з ними як приріст. Kaggle score записаний як `null`, доки CSV не оцінено самим змаганням. Скрипт не використовує leaderboard для вибору розміру та не відправляє CSV автоматично. Повторний запуск захищений lock; записи поточної GPU черги захищені окремим work lease, який успадковується активним процесом.

### Плитки для дрібних об'єктів

Наступний inference-експеримент додає перекривні плитки до повного кадру. JPEG читається один раз на кадр, CPU workers готують наступні views паралельно з GPU. Workers передають uint8 tensors, а перетворення до `[0,1]` виконується на пристрої. Бокси повертаються з letterbox у координати crop, потім зміщуються в оригінальний кадр. Детекції, обрізані внутрішньою межею плитки, відкидаються; зовнішня межа оригіналу залишається допустимою. Повний кадр зберігає контекст великих об'єктів.

```bash
python -u scripts/start_tiled_search.py \
  --work "$CVLAB_LOCAL_WORK" --models rtdetr fasterrcnn \
  --tile-sides 768 960 --ious 0.5 0.6 \
  --max-images 192 --workers 2 --vram-gib 6
```

Launcher чекає GPU lease, тому його можна поставити після поточного пошуку роздільності. Він використовує той самий заморожений checkpoint та розмір input, підтверджений попереднім сценарієм. Окремо порівнюються вихідний global inference, global + class-specific NMS та global + tiles + NMS на одних і тих самих 192 official-val кадрах. Далі переможець перевіряється на всій official val проти global inference. Довгий тестовий export запускається лише при прирості не менше `--min-gain 0.01` (1 percentage point mAP); слабші результати зберігаються без нового CSV. Це поріг витрати GPU-часу, не прогноз Kaggle score. Для повільнішого вичерпного пошуку поріг можна явно знизити. Початкова fold0-оцінка не підміняється цим holdout. Реальна швидкість враховує original frames/s, а не кількість плиток.

Дублікати зливаються окремо для кожного класу; ліміт 500 застосовується до оригінального кадру після об'єднання всіх views. Прогнози об'єднуються по одному кадру, а тестовий CSV пишеться частинами з перевіркою порядку/IDs/координат і SHA checkpoint. `score_improvement/tiling_summary.json` містить всі абляції. Невдалі subset/full-validation варіанти теж зберігаються; новий CSV не створюється, якщо глобальний варіант кращий. `experiments/leaderboard_results.json` зберігає лише фактичні оцінки, надані користувачем, окремо від validation scores.

## Здача роботи

Після отримання викладацького GitHub template пакет потрібно перенести до нього. README — інструкція запуску, не фінальний звіт. Ще потрібно виконати реальні навчання й tuning, аналіз failure cases, графіки порівняння, перевершити competition baseline, написати власний звіт і пройти захист. Для бонусу visualization потрібно додати W&B.

## Джерела для атрибуції

- TorchVision: https://pytorch.org/vision/0.21/models.html#object-detection
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- RetinaNet: https://arxiv.org/abs/1708.02002
- FCOS: https://arxiv.org/abs/1904.01355
- GIoU: https://arxiv.org/abs/1902.09630
- RT-DETR: https://arxiv.org/abs/2304.08069
- Slicing inference idea (SAHI): https://arxiv.org/abs/2202.06934 ; the tile implementation here uses PyTorch/Pillow directly.
- Transformers: https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/rt_detr
- COCO: https://github.com/cocodataset/cocoapi
- VisDrone: https://arxiv.org/abs/2001.06303
- PyTorch wheels: https://pytorch.org/get-started/previous-versions/

Дотримуйтеся ліцензій бібліотек, pretrained weights і competition data.

### Warm fine-tuning with native crops

`scripts/start_crop_finetune.py --work "$WORK" --models rtdetr --hours 2`
queues a separate experiment after an already queued tiled search and the current
GPU owner. Duplicate launchers exit. Keep the existing `systemd-inhibit`/`nohup`
wrapper when running overnight; the Python launcher itself stays in the foreground.

The recipe initializes from `runs/rtdetr_fold0_local/best.pt`, with a fresh optimizer,
scheduler and time budget, while preserving the model architecture and input size.
It uses all official train clips, every second frame within each clip, learning rate
`5e-5`, batch 1 with accumulation 4, and two loader workers. Half of the training
examples use a native 768-pixel crop centered near an object sampled uniformly by
present class; the rest use the whole frame. Ignored areas are masked first, and
crop boxes are clipped with their labels. Validation and test frames are never cropped
by this training option. These changes form a combined experiment; attributing a gain
to one component requires separate ablations.

Before any optimizer update, the source weights are measured on the complete official
validation split and saved as the initial best. Fold-validation scores are not reused
as that threshold. Source training-sequence lineage is checked against the new holdout;
legacy checkpoints require their matching `split.json`. Source weights are preserved.
The new output is `runs/rtdetr_crop_finetune`; CPU COCO metrics can overlap the next
training epoch, with the existing host-memory gate. The two-hour budget includes initial
validation and training; final validation and CSV export can extend wall time beyond it.

Repeating the same command resumes the new run's `last.pt`; it does not reset the budget.
The summary is `crop_finetune_summary.json`. A native-resolution CSV is automatically
exported only after at least `0.01` absolute COCO100 mAP improvement over initialization
on the same full official validation split. This threshold saves export work; it does
not predict Kaggle improvement. The output log then prints `CROP CANDIDATE READY FOR KAGGLE`.
The target public baseline remains `0.2216324580`; training this recipe has not yet
established that score. This official-split experiment does not replace sequence-fold
OOF evaluation required for the assignment.

For a custom new experiment, `cvlab train --config ... --init-checkpoint ... --async-metrics`
exposes the same warm initialization. `--init-checkpoint` and `--resume` are mutually
exclusive; resume keeps the saved configuration and experiment protocol.

### BatchNorm diagnostic after the failed crop experiment

The observed crop experiment retained its initialization: official mAP `0.1676`
versus `0.1070` at the last epoch. Tiled inference reached public Kaggle `0.1928780288`,
which remains below `0.2216324580`. Exact public scores and the rounded crop log
observations are recorded under `experiments/`.

`scripts/start_bn_search.py --work "$WORK"` queues an inference-only diagnostic behind
active GPU work. The pinned Hugging Face RT-DETR freezes backbone BatchNorm layers,
but its input projections and encoder still contain mutable BatchNorm. With microbatch
size 1, gradient accumulation does not enlarge the batch used to update these statistics.
Statistics drift is a hypothesis for degradation, not an established cause.

The diagnostic compares original `best.pt`, its train-calibrated version, crop `last.pt`
(if available), the crop weights with original BatchNorm statistics, and crop weights
with recalibrated statistics. Calibration uses 256 clip-balanced official **train**
frames at full-frame inference preprocessing. Only mutable BatchNorm runs in training
mode with cumulative running statistics; there are no gradients, optimizer updates,
crops, flips, dropout or label-based adaptation. Frozen backbone normalization and all
learned weights, including BatchNorm affine parameters, are preserved. Calibration and
source-training clips must be disjoint from official validation clips.

All candidates use the same 192 official validation frames. A winning candidate and
the original are then evaluated on all 2,846 validation frames. CSV export requires
at least `0.01` absolute COCO100 mAP improvement over the stronger of original full-frame
inference and an existing, provenance-matched tiled full-validation result. Public
leaderboard scores are never used in calibration or this selection. Candidates that
fail this check retain their diagnostic metrics but produce no CSV.

Artifacts live under `bn_diagnostic/`; `bn_diagnostic/summary.json` is the latest result.
Derived checkpoints are inference-only; original runs are preserved. Repeated launches
reuse completed cases keyed by source hashes, calibration data, validation data and
configuration. Use the usual `nohup systemd-inhibit ...` wrapper for overnight work.
The log prints `BN CANDIDATE READY FOR KAGGLE` only after a candidate passes confirmation.

### Continue best RT-DETR weights on full frames

`scripts/start_fullframe_finetune.py --work "$WORK" --hours 2 --replace-bn`
queues a separate warm fine-tuning run initialized from the original
`runs/rtdetr_fold0_local/best.pt`. The optional `--replace-bn` stops only actual
Python BN diagnostic processes for the same work root, including their queued
launcher, using Linux process identity handles. Other training jobs and wrappers
are preserved; the new job waits for the GPU lease. Duplicate full-frame launchers
exit without starting another trainer.

The model input size and architecture stay unchanged. Training uses all 24,201
frames from the 56 official train clips, with `frame_step=1`, no crops, learning
rate `1e-5`, backbone factor `0.1`, batch 1 and accumulation 4. The official 2,846
validation frames remain held out. This is a new optimizer/schedule and an additional
time budget initialized from learned weights, not an exact continuation of the
original optimizer or a claim to identify individual previously unseen frames.
Initial weights are reevaluated on the new holdout before training, and a worse
trained checkpoint cannot replace that initial best.

Output: `runs/rtdetr_fullframe_finetune`; configuration:
`configs/rtdetr_fullframe_finetune.yaml`. The same command resumes the new run's
`last.pt` with its remaining budget. Early stopping uses two consecutive completed
validation results without a gain exceeding `0.001` from its running reference.
With asynchronous CPU metrics, it also checks committed results at training
boundaries, so an in-flight next epoch can be saved and stopped. Pending validation
is completed on resume even if the stopping condition is already reached.
The two-hour budget includes initial validation; final validation/export may extend
wall time. Native-resolution CSV export requires a `0.01` full-validation gain over
initialization and prints `FULLFRAME CANDIDATE READY FOR KAGGLE`. Native and tiled
inference must still be compared on the resulting learned weights; this command
performs training and the native export, not a new tiling search.

### Monitored continuation in configurable training rounds

`scripts/start_iterative_finetune.py --work "$WORK" --rounds 3 --batches 6000 --export-tiles`
continues the learned weights from the completed `rtdetr_fullframe_finetune/best.pt`.
It waits for that run's GPU lease, including final CPU metrics and export, before
selecting the source checkpoint. The first round uses a fresh low-LR optimizer;
subsequent rounds and later invocations restore weights, AdamW moments, scheduler,
RNG, elapsed budget and loader state from the new experiment's `last.pt`.

Output: `runs/rtdetr_iterative_finetune`; summary: `iterative_finetune_summary.json`.
Each round processes at most 6,000 randomly shuffled full train frames and then
validates on all official validation frames. This is a bounded round, not a full
pass through the dataset. All train clips and frames remain in the sampling pool.
`--batches` can be changed on the next invocation without resetting weights or
optimizer state; the effective limit is recorded in checkpoint execution metadata
and history. `--rounds 3` runs at most three rounds, with a cumulative 12-hour
experiment cap as a guard, not an instruction to run continuously for 12 hours.
The validation interval, export time and total wall time depend on actual throughput.

The recipe keeps the 960 input/architecture, FP16, batch 1, accumulation 4 and LR
`1e-5`, with five-result patience. During training it freezes BatchNorm running
statistics while leaving affine parameters trainable. This is a training experiment
aimed at batch-1 stability; it does not calibrate normalization on validation/test data
and has no measured GPU/competition gain yet. Original checkpoints remain preserved.

`ROUND COMPLETE` prints current/best/initial official mAP. With `--export-tiles`, a
new best that improves native mAP by at least `0.002` triggers the already chosen
960-pixel tiles / NMS 0.6. The same checkpoint/settings first use 192 validation frames,
then full validation. Export requires a `0.001` full-validation gain over the stronger
of the new native prediction and the previous verified tiled model. Dataset IDs,
fingerprint and previous checkpoint provenance must match. Old tiling reports are
preserved; custom-run results use `score_improvement/tiling_finetune_summary.json`.
Only then does `TILED CANDIDATE READY FOR KAGGLE` announce a CSV. After export, the
next training round continues while the user can upload the completed CSV externally.
GPU validation/export is serialized with training on this single device.

To request a stop after the current round, create
`runs/rtdetr_iterative_finetune/STOP`; the controller preserves checkpoints and skips
further exports/rounds when it sees the file. Remove it before continuing. Repeating
the launcher resumes the experiment. Specific runs can also use the existing tiled
launcher directly: `--runs rtdetr_iterative_finetune --tile-sides 960 --ious 0.6`.

### Audit a regressing tiled submission

The user-observed `rtdetr_cc3d87b1034f_960_tiles_b57dbbdfb3d4.csv` public score
`0.1746973387` regressed from `0.1928780288`. This is not a confirmed improvement.
`scripts/audit_tiled_candidate.py --work "$WORK"` performs a read-only CPU comparison
of those two CSVs: exported checksum, checkpoint SHA, tile options, test fingerprint,
class/score distributions, source frozen run, complete native/tiled COCO100/500 scores,
per-class AP and the recorded export-selection reference. Missing artifacts are reported;
the script does not infer an unseen export provenance from training logs alone.

### Adaptive short trials with rollback

`scripts/start_adaptive_finetune.py --work "$WORK" --attempts 2 --batches 6000 --export-tiles`
starts from the preserved `a7cd89fb349d` RT-DETR checkpoint used by the best observed
public tiled submission (`0.1928780288`). It keeps the original training fold and its
complete held-out fold; the previously added fold-validation clips are not used for
training here. Each trial processes at most the requested number of frames or 30 minutes
of training, then completes validation. Validation/export add wall time beyond that
training cap. There is no instruction to consume a long fixed training block blindly.

A challenger must gain at least `0.001` COCO100 mAP on both the complete original
held-out fold and a fixed 192-frame official confirmation set. The official clips are
checked against source training sequences. A rejected challenger is archived in its
own trial directory, the next trial reloads the champion, and LR is halved. An accepted
challenger becomes the internal champion. This is an internal selection rule, not a
guarantee of a higher public/private leaderboard score. The original weights/CSV remain
preserved. `ACCEPT` / `ROLLBACK` logs show both gains, the next LR and champion hash.

Weights, AdamW moments, AMP scaler and loader/RNG state are initialized from the chosen
champion; LR and the per-trial budget/schedule are explicitly reset. BatchNorm running
statistics remain fixed, while affine parameters can learn. Source validation metrics
are reused only when checkpoint epoch/score, exact train/val fingerprint, model,
precision, TF32 setting, batch size and detection caps match, avoiding repeat evaluation
of the same initial model. Pending confirmation resumes without repeating its training.

State: `adaptive_training/state.json`; trial artifacts:
`runs/rtdetr_adaptive_finetune/attempt_XXX/`. The optional tiled check uses the established
settings and existing full-validation export hurdle before producing a CSV; a following
trial can train while that completed CSV is uploaded. `--reset-to-original` restores the
verified original champion, reduces LR and preserves all previous trials if external
feedback invalidates a branch. A `STOP` file in `runs/rtdetr_adaptive_finetune/` prevents
further trials/exports at the next controller boundary. Repeating the launcher continues
from its saved state, with duplicate-launch and GPU-lease protection.
# Efficient continuation from the verified public-best checkpoint

Use `scripts/start_efficient_search.py` for bounded LR racing rather than a long
unobserved fine-tune. It warm-starts the verified RT-DETR weights and optimizer,
compares three LRs at cumulative 6k/12k/24k-frame budgets, and preserves every
measured checkpoint. A 512-frame probe uses unseen video families; only promising
candidates receive full native and tiled confirmation before CSV export.
The original public-best run is preserved. The GPU launcher waits for current
work, and the default four-hour allowance includes a one-hour confirmation
reserve (a soft limit; active validation/export completes).

See [the mathematical policy and commands](experiments/efficient_search.md).

