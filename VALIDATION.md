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

## Фіксовані ваги та інша роздільність інференсу

Вісім цільових CPU-перевірок пройшли: відтворювана subset-вибірка зі spread усередині clips; очікування інших CLI jobs без очікування власного PID; відновлення ваг трьох CNN-детекторів із розміром 128→160 без зміни tensors та з реальним forward; збереження заборони resize для RT-DETR; наскрізне порівняння двох роздільностей на реальному training checkpoint синтетичного dataset; реальний глобальний flock для запобігання кільком очікувачам у різних output-каталогах. Зміни навчання не вносилися. Реальна користь 960→1280 на VisDrone і швидкість на GPU ще не вимірювалися.
### CPU postprocessing alongside GPU jobs

`tests/test_postprocess_experiments.py`: 15 CPU tests passed. They cover chunked JSON with Unicode and split numeric tokens, rejection of malformed/missing arrays, bounded reader allocation (<2 MiB on 30,000 streamed detections), class/image-specific NMS without input mutation, a known AP degradation from overly aggressive suppression, exact validation fingerprint checks, fold isolation, and a subprocess check that importing the module does not import Torch. Ruff checks pass for the new module and tests. No speed or AP improvement on VisDrone has been measured by these synthetic checks.
### CPU validation overlapped with GPU training

17 targeted pipeline/queue/handoff tests passed. They include real FCOS forward/backward and checkpoint reload on CPU, an event-gated proof that epoch 2 training starts before epoch 1 metrics finish, exact serial/overlapped last and best weights/scheduler/score equality, training of the next queued model while previous metrics are blocked, snapshot isolation, one-prediction-set backpressure, async error propagation and resume without repeated training, DataLoader spawn while the metrics thread is active, legacy queue regression checks, and handoff process filtering/control at the completed-validation log marker. These tests do not benchmark CUDA throughput or validate peak memory on the user's laptop.
### Concurrent progress output

COCO verbosity is suppressed only in the evaluating thread. The previous process-wide `redirect_stdout` could hide GPU training progress during asynchronous CPU evaluation. Tests cover live training output during a blocked metrics task, nested suppression/error cleanup, and overlapping quiet threads. The targeted output/COCO/pipeline/postprocessing regression suite passed 26 CPU tests; reference COCO precision equality is retained. Already running Python processes retain their imported implementation until their next launch.
### Resolution probe launcher wrappers

13 inference-experiment tests passed. GPU-idle detection now inspects `/proc/<pid>/cmdline` for the actual Python `-m cvlab` invocation rather than matching wrapper command text. Coverage includes inhibitor/shell/Python `-c` wrappers, real train/evaluate CLI forms, and a live Python-wrapper subprocess regression that exits instead of waiting on its own parent. Existing real CNN resize/weight-equality, RT-DETR rejection, sampling and global-lock tests still pass. This fixes a queued probe waiting indefinitely under `systemd-inhibit`; GPU inference quality was not measured by these CPU tests.
### CPU submission generation

Five targeted tests passed: chunked output in exact template order, preservation of empty image rows and conversion of internal class 8 to submission class 7, frozen checkpoint isolation while a newer best appears, existing-file and duplicate-writer protection, all-empty rejection, and a real FCOS CPU inference/submission validation in the end-to-end pipeline test. Ruff and CLI help checks pass. The full 3,811-frame VisDrone test inference and its wall time must run on the user's machine; no competition CSV or score was fabricated in the cloud workspace.
### GPU submission export budget

The submission path supports a separate `predict-submission` CLI with FP16 CUDA inference, batch 1, zero loader workers and an explicit per-process PyTorch allocator limit. Seven targeted submission/end-to-end tests passed before the final RAM guard; the six focused submission tests also passed after adding it. Coverage includes allocator fraction, VRAM and RAM rejection before loading, FP16/cap wiring, and the existing actual CPU model/export flow. GPU budget checks use mocked CUDA APIs in this CPU-only workspace; actual concurrent GPU throughput and memory peaks remain to be measured on the user's laptop. No guarantee of reserving memory or protecting another process's future peak is made.
### Indexed CUDA allocator device

Eight focused submission tests pass. CUDA memory configuration resolves an unindexed `torch.device('cuda')` through `torch.cuda.current_device()` and passes an explicit integer index to memory APIs. Regression cases cover default, explicit zero and nonzero devices and invoke the real Torch 2.6 device-argument validator inside the mocked allocator call, reproducing the argument contract without GPU hardware. The earlier test mock accepted an unindexed device and missed this compatibility error.
### Resolution candidates for the Kaggle baseline

31 targeted tests passed across inference experiments, score improvement, and chunked submissions. They include a real RT-DETR forward at a different input size with exact weight preservation, rejection of static/missing RT-DETR metadata, frozen checkpoint provenance, official-holdout leakage rejection, native CSV exports before search, full-validation rejection of a subset winner, OOM fallback, cached full-validation reuse, and inherited GPU-lease / duplicate-launch checks. Existing CPU export and GPU-budget unit tests also pass. These CPU tests do not measure a VisDrone AP gain or a Kaggle score.
### Score-export resolution compatibility regression

38 targeted tests passed across chunked export, score improvement, and inference resolution. Added an integration check that calls `score_improvement.export` through the real `create_submission`, verifies the requested image size reaches loading/prediction and CSV provenance, and confirms the saved training config is unchanged. Invalid sizes fail before model setup, and the `predict-submission --image-size` CLI forwards the override. This covers the previously missed exporter keyword mismatch.
### Full frame plus overlapping tiles

49 targeted tiling, export, score-search and resolution tests passed. The new tiling tests cover full crop-grid coverage on odd dimensions, global-view deduplication, the view-count bound, box restoration with crop offsets, interior-edge filtering, class-specific duplicate removal and the final 500-per-image cap. A real generator/CPU detector path checks inference mode, uint8 worker payloads, a spawn worker, original-image bounds, streamed CSV creation, sample order and provenance reuse. A real randomly initialized RT-DETR also completes global-plus-tile forwards with valid boxes. Search tests exercise the actual tiled predictor and CSV exporter end to end with a deterministic CPU detector, including rejecting a subset winner that loses on full validation. They also check inherited leases and duplicate-launch protection. No VisDrone/Kaggle gain is claimed by these CPU tests.
### Minimum useful tiling gain

51 targeted tests passed after adding `--min-gain` (default 0.01 absolute mAP) before tiled test export. Tests distinguish a loss, a positive but small 0.003 gain, an exact 0.01 boundary and a larger improvement. Metrics/caches remain available when export is skipped. This is a compute-budget rule, not a predicted Kaggle gain.

### Crop fine-tuning and checkpoint initialization

Full CPU regression suite: **175 passed**, with 14 existing Matplotlib/Pyparsing
warnings. Ruff and training CLI help checks pass. New tests execute the real training,
optimizer/checkpoint and asynchronous-finalization paths with a tiny detector: source
weight preservation, fresh optimizer and lower LR, new-holdout initialization score,
completed-run resume without the old source file, interrupted initial-validation
recovery, legacy split fingerprint verification, and rejection of supervised sequence
leakage or model-configuration changes. Additional checks cover native crop geometry,
label alignment, validation isolation, legacy spawned datasets, all-train-clip sampling,
real controller-to-engine integration, no export on a tied score, duplicate launcher
rejection and inherited GPU lease after the tiling dependency. The existing real
TorchVision/RT-DETR inference, resume, submission and overlap tests also passed.

These checks establish software behavior on CPU. No CUDA training throughput, laptop
peak memory or VisDrone/Kaggle improvement for this crop recipe was measured here.

### Train-only BatchNorm diagnostic

**184 CPU tests passed**, with 14 existing Matplotlib/Pyparsing warnings; Ruff, CLI
help and launch-shell syntax checks pass. Nine added tests cover: calibration changing
only running moments/counters, preserving affine and other learned weights, restoring
module modes/momenta after errors, a real pinned Hugging Face RT-DETR forward with
unchanged parameters, source/calibration sequence leakage rejection, full-validation
rejection of optimistic subset results, comparison against the stronger existing tiled
candidate, original-checkpoint preservation, derived inference-only artifacts and
launcher-to-CLI GPU-lease/argument forwarding. Controller score outcomes and CUDA budget
APIs are mocked in CPU tests; this does not establish any VisDrone/Kaggle gain or laptop
CUDA performance. Calibration must be run on the user's dataset to test the hypothesis.

### Full-frame warm fine-tuning and early stopping

**197 CPU tests passed**, with the same 14 Matplotlib/Pyparsing warnings. Ruff, the
training CLI and launch-shell syntax checks pass. Thirteen new checks cover the full
training-frame recipe with disabled crops, source-config preservation, significant-gain
patience with unique completed epochs, serial/asynchronous early stopping while keeping
initial best weights, self-contained stopped-run resume, pending-validation recovery
without repeated training, BN-only process selection for the exact work root, Linux
process-identity pinning before signaling, preserved unrelated training processes,
duplicate-launch protection and actual launcher-to-CLI argument/lease forwarding.
Existing warm-initialization, normalization, real detector inference, submission,
resume and CPU/GPU-overlap regression tests also pass. CUDA training and any score gain
from this full-frame recipe still need measurement on the user's GPU and dataset.

### Configurable monitored continuation and new-weight tile candidates

The full CPU regression run passed **205 tests**. Seven focused iterative tests also
passed after the final allocator-budget guard. Coverage includes real training/checkpoint
round yields, restoration of optimizer step counts and RNG-compatible continuation with
changed round limits, frozen BatchNorm moments with trainable affine gradients, bounded
round counts, STOP handling, source-checkpoint preservation, launcher/CLI inherited leases,
training/export/training handoff with continued optimizer progress, and restoration of
the GPU allocator fraction after successful or failed inference. Four additional custom-run
tiling cases in the full regression suite verify full-holdout confirmation against the
previous tiled reference and preservation of the old report, with real geometry/CSV export.
Ruff, CLI help and shell launch syntax checks passed. GPU allocator APIs use CPU mocks;
actual CUDA throughput and any new VisDrone/Kaggle gains remain unmeasured here.

### Regressing candidate provenance audit

The reported public score `0.1746973387` is recorded as a regression, with its source
checkpoint prefix and screenshot evidence. A standalone standard-library audit was
smoke-checked on temporary CSV/metadata/cache fixtures: class counts, empty rows,
confidence counts, full COCO500 cache extraction, unchanged-CSV checksum equality,
changed-CSV checksum rejection and missing-file reporting. Ruff passes. The actual
candidate's local selection reports have not been supplied; the export reference and
root cause remain unverified. No additional GPU training is required by this audit.

### Adaptive champion/challenger controller

Full CPU regression suite: **220 passed**, with 14 existing Matplotlib/Pyparsing warnings.
The 12 adaptive tests were rerun successfully after adding explicit original-champion
reset and RNG-state assertions. Coverage includes two-holdout acceptance, rejection of
one-sided or insufficient gains, exact source metric-cache compatibility guards, actual
AdamW/scaler/RNG warm restoration with a new LR, no repeated initial inference, interrupted
confirmation recovery without repeated training, accepted/rejected/accepted trial
transitions, optimizer step counts `[2,3,3]` proving the failed branch was not inherited,
LR halving, original-checkpoint preservation, explicit public-best reset, STOP handling,
official confirmation leakage rejection and launcher/CLI lease forwarding. Ruff, CLI and
launch-shell checks pass. These are CPU behavior checks; the policy has no new measured
CUDA training speed or Kaggle gain yet. The observed `3ae5fa3a569c` score `0.182472636`
is recorded as another regression against the original public-best result.
# Efficient-search validation

`tests/test_efficient_search.py` checks paired flight-family evidence, cost-based
allocation, exclusion of related training excerpts, capped validation with the
full training fold retained, optimizer continuation in serial/asynchronous
validation, interruption recovery without repeated training, immutable earlier
checkpoints, reuse across search cycles, and launcher/CLI GPU-lease handling.
These CPU checks validate the controller; they do not establish a Kaggle gain.

