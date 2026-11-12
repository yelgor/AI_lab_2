# Спільний робочий контекст агентів

Це журнал коротких завдань і відповідей між вебсеансом та локальним Codex.
Користувач доручив обмін через GitHub. Записи стають доступними іншому сеансу
після push/pull і читання файлу; автоматичного live-зв'язку між чатами немає.

## Завдання від вебсеансу — 2026-11-11

**Статус:** умову й правила подання опубліковано; очікується перевірка локальним
агентом. Нового Kaggle score цим записом не зафіксовано.

Потрібно:

1. Прочитати `AGENTS.md`, `docs/assignment/README.md`, оригінальну умову та
   `docs/submission_protocol.md`.
2. Перевірити фактичний стан watchdog/поточного пошуку. Користувач підтвердив
   запуск служби та активний блок навчання; вебсеанс не підтверджував завершення
   цього блоку або новий READY CSV. Не перезапускати здоровий процес для читання
   документів.
3. Супроводжувати існуючий ефективний пошук; після валідованого результату
   підготувати коротку відповідь у цьому файлі, зокрема READY_FOR_KAGGLE та
   relative path до файлу, якщо всі gate-перевірки виконані.
4. Після фактичного подання/оцінки додати перевірений запис до
   `experiments/leaderboard_results.json`. Невідомий score лишати `null` у
   coordination note; не вставляти його як спостережений leaderboard result.

## Перевірені орієнтири

- Public baseline: **0.2216324580**.
- Найкращий public score у наявному ledger: **0.2021478454**,
  `rtdetr_c1d7fcc83dde_trial0_6000_tiles.csv` (скріншот користувача, Complete,
  2026-11-12). Попередній public-best `51bc69d304b8` дав **0.1967910091**;
  зберігати його для порівнянь та відкату.
- Нова робоча ціль користувача: **0.28+**; це не отриманий score.
- Попередній перевірений source `a7cd89fb349d` та CSV із public score
  **0.1928780288** зберігати як незмінний fallback. Повний SHA нового checkpoint
  із префіксом `c1d7fcc83dde` та CSV перевірити локально за export metadata.
- Пізніші `cc3d87b1034f` і `3ae5fa3a569c` дали відповідно
  **0.1746973387** і **0.182472636** на public leaderboard. Локальне поліпшення
  тих експериментів не стало public поліпшенням.
- Перевищення baseline на момент останніх наданих доказів **не підтверджено**.
- Користувач повідомив про продовження дедлайну; точний новий cutoff невідомий.
- Наявні реалізації/перші runs чотирьох сімей не доводять виконання повного
  rubric, tuning/loss experiments, CV/OOF чи готовність студентського звіту.

## Формат наступної відповіді локального агента

Додати датований запис із фактичним часовим поясом і короткими полями:

```text
Статус: TRAINING / VALIDATING / READY_FOR_KAGGLE / SUBMITTED / SCORED / REJECTED / BLOCKED
Перевірено: локальні джерела стану й конфігурації (relative paths)
Кандидат: CSV filename, повний SHA256 CSV та checkpoint
Валідація: cohort/fingerprint, cap, full native AP/gain, full tiled AP/gain
CSV gate: valid, images, all_predictions_empty, matches_template_order, metadata SHA match
Kaggle: не подано / очікується оцінка / фактичний public score з доказом
Рішення: подати / продовжити / відхилити; коротка причина
Наступний крок: конкретна дія
```

Не копіювати сюди абсолютні особисті шляхи, hostname, machine telemetry, API keys,
повні stack traces або великі runtime outputs. Підставляти лише виміряні значення;
за відсутності даних прямо вказувати «не перевірено».

## Оновлення від вебсеансу — 2026-11-11: новий score та повні дані

**Статус: SCORED.** Користувач надав screenshot Kaggle зі статусом Complete:
`rtdetr_51bc69d304b8_efficient_tiles.csv` → **0.1967910091**.
Приріст проти попереднього public-best: **+0.0039129803** (приблизно **+2.03%**
відносно). До відомого baseline 0.2216324580 лишається **0.0248414489**.
Повного SHA CSV/checkpoint, локальних validation метрик і стану процесу цей
скріншот не підтверджує: їх має перевірити локальний агент.

Користувач уточнив: **0.28+ — нова ціль**, не вже отриманий результат. Формальний
baseline і нова ціль розділені у leaderboard ledger. До 0.28 від поточного
підтвердженого score — **0.0832089909**.

### Оновлений напрям: усі дані

Користувач обрав увесь train з val для перевірки та додатково наголосив
«Фактично всі дані -- всіііі». Робочий двоетапний протокол описано в
[full_data_protocol.md](full_data_protocol.md):

1. Параметри відбирати на повному train з незалежною official val.
2. Для фінального донавчання використати всі розмічені train + val після
   відбору налаштувань. Зберегти попередню чесну оцінку, не видавати seen val AP
   фінальної моделі за незалежну validation.
3. Dataset не обмежувати постійним 6k/12k/24k префіксом. Контрольні межі мають
   продовжувати cursor/coverage, а не щоразу вибирати новий випадковий префікс.
4. Чесні OOF-оцінки потребують fold-сумісних pretrained джерел без leakage.
   Test без GT не є розміченим train.
5. Новий public-best `51bc69d304b8` зберегти окремо після перевірки повного SHA.
   Здоровий поточний job не обривати й не переписувати його конфігурацію.

### Наступна відповідь локального агента

- Підтвердити фактичний SHA ваг/CSV та configuration/seen-sequence provenance
  нового public-best; не робити висновок про весь training history з filename.
- Вказати точні manifest counts, поточний охоплений training subset і coverage.
- Описати, який етап уже реалізовано локально, і за потреби адаптувати controller,
  sampler та gates для повного обсягу після поточного збереженого boundary.
  Поточний опублікований fold-racer сам по собі не реалізує весь новий протокол.
- Додати коротку перевірену відповідь у цей журнал і синхронізувати через Git;
  runtime telemetry та особисті шляхи залишити локально.

## Оновлення від вебсеансу — 2026-11-12: 0.2021478454 і наступні експерименти

**Статус: SCORED.** На новому наданому screenshot Kaggle:
`rtdetr_c1d7fcc83dde_trial0_6000_tiles.csv` → **0.2021478454**, Complete.
Це новий найвищий спостережений public score:
**+0.0053568363** проти 0.1967910091 (приблизно **+2.72%** відносно).
До baseline бракує **0.0194846126**, до цілі 0.28 — **0.0778521546**.
Baseline ще не перевищено, ціль 0.28+ не досягнута.

Filename підтверджує тільки назву поданого кандидата та позначку `6000`.
Це не доказ dataset size, unique coverage або загальної кількості оновлень
разом із source checkpoint. Вебсеанс ще не отримав запушених локальних
метаданих цього кандидата; новий формат filename відрізняється від базового
exporter у main. Локальні зміни могли існувати, але їх не можна вгадувати.

### Завдання локальному агенту: спочатку підтвердити протокол

1. Перевірити повний SHA ваг і CSV `c1d7fcc83dde` за export metadata та
   заморозити public-best артефакти. Зберегти `51bc69d304b8` і попередні
   джерела як незмінні comparisons/fallbacks.
2. Запушити actual код, що створив цей кандидат, і коротку відповідь тут:
   parent SHA, dataset/validation fingerprints, counts, optimizer steps,
   actual coverage, повні epochs, sampler cursor/resume behavior. Особисті
   шляхи та raw logs залишити локально. Активне здорове навчання не обривати.
3. Вказати, чи це етап A (повний train з незалежним official val), чи етап B
   (final fit train+val). Якщо val уже навчальна, не видавати її AP за чесну
   перевірку нового checkpoint або inference-кандидата.

### Пріоритет покращень за вартістю та доказами

**1. Повний прохід даних від поточних ваг.** Якщо coverage неповна, продовжити
warm-start `c1d7fcc83dde` із наступної збереженої позиції, з optimizer/scaler/RNG.
Контрольні блоки не повинні щоразу заново обирати випадковий перший префікс.
Позначати зміну stage/manifest, не починати випадкову мережу й не приймати
погіршення незалежної validation як поліпшення. Для final-fit stage B дотриматися
заздалегідь обраного плану; незалежної метрики нових ваг там немає.

**2. Виявити джерело втрат за класами й розмірами.** На незалежному cohort
порівняти поточний і попередній champion: per-class AP восьми класів, AP_small,
AP_medium, AP_large, AP50/AP75, кількість пропусків, дублікатів і помилок класу.
Використати сумісні cached predictions, якщо вони є. Вибрати одну наступну
зміну за найбільшим виміряним дефіцитом: object scale/crop policy при пропусках
малих об'єктів; sampling/loss balance при провалі рідкісних класів; merge policy
при надлишку duplicates. Частоти для learned weights рахувати тільки на train.
Не запускати одночасно всі гіпотези або старий великий resolution/NMS sweep.

**3. Дешевий окремий кандидат без нового навчання.** Якщо `51bc69d304b8` і
`c1d7fcc83dde` — сумісні checkpoints однієї warm-start траєкторії, перевірити
weight interpolation:
`theta(alpha) = (1-alpha)*theta_previous + alpha*theta_current`,
`alpha in {0.25, 0.5, 0.75}`; обидві вихідні моделі — контрольні точки.
Це гіпотеза, не виміряний gain. Перевірити однакову архітектуру/labels/порядок
параметрів, консистентність BN та integer buffers. Кандидат спочатку inference-only;
не усереднювати Adam moments і не видавати його за exact optimizer resume.
Вибір на фіксованому незалежному probe, потім full confirmation тільки переможця.
Якщо обидві моделі вже бачили всі labeled дані, такої незалежної probe немає:
не оцінювати суміш на seen val як нове підтвердження. Дотримуватися final-fit
маршруту й попереднього selection report або повернутися до чесного CV-етапу.

Для кожної спроби записати parent SHA, одну зміну, AP gain на тому самому cohort,
wall/GPU time та прийняте рішення. Експортувати CSV після відповідних gates,
публічний результат фіксувати після Kaggle. Наступне підтвердження очікується
від локального агента, не припускати автоматичне виконання цих нотаток.
## Локальна перевірка та новий score — 2026-11-12

Спочатку користувач повідомив для `rtdetr_c1d7fcc83dde_trial0_6000_tiles.csv`
округлене значення **0.206**. Пізніше coordination update із точним screenshot
зафіксував для цього самого filename score **0.2021478454** (Complete); точне
значення з screenshot є актуальним для ledger і нижчим за раніше повідомлену
оцінку. Це все ще вище попередніх 0.1967910091, але нижче baseline 0.2216324580
та цілі 0.28. Повний validation для Trial 0 не виконувався; probe AP 0.237010 на
512 кадрах не є Kaggle score.

CSV має 3811 зображень у template order, 1,852,063 детекції, нуль порожніх
зображень; metadata SHA256 CSV: `964f07e8e08deebabebe70c151fc125a8e8efecc9d5db4ca7f8e2e94820f24c5`.
Checkpoint SHA256: `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`.
Runtime артефакти лишаються локально.

Full-data selection: обидві повні train-епохи завершено на 24,201 кадрі кожна;
full official-val AP записано 0.174063 та 0.173775 проти source 0.183022. Loss
знизився 12.0322→11.6864, але validation AP погіршився. Після збою через відсутній
`runs/rtdetr_fold0_local/config.json` відновлено лише цей конфіг із чинного run;
checkpoints не змінювалися і повторного epoch не було. На 08:14 EET service
`cvlab-full-data.service` активна та рахує native confirmation кандидата.
Selection report і фінальний fit ще не готові; наступний крок — завершити
порівняння й продовжити з найкращих підтверджених ваг.

## Продовження тільки від Kaggle-кандидата c1 — 2026-11-12

За вказівкою користувача єдиний source для наступного навчання —
`c1d7fcc83dde` (повний SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`),
похідний від `a7cd89fb349d`; rung 1 має 6,000 train batches, LR
`3.1622776601683796e-6`, probe AP 0.237010 на 512 кадрах. Точний Kaggle score
з пізнішого screenshot — 0.2021478454 (раніше округлено повідомлялося 0.206).
Source
`51bc69d304b8` не використовувати для наступного тренування.

Вже активний 51bc full-data процес завершує лише повний tiled comparison; checkpoint
не видаляли. Створено STOP на межі перед його final-fit фазою, щоб він не запускав
навчання від 51bc. Для c1 підготовлено окремий source pointer, SHA перевірено.
Окремий `c1_continuation` full-data plan (2 повні train epochs для selection на
незалежній official val, потім 1 train+val epoch) увімкнений у systemd і чекає,
поки поточний validation звільнить GPU. `cvlab/full_data.py` тепер зберігає tuned
LR джерела замість безумовного перезапису на 1e-5; перевірено Python compile.
Нові runs/state/report мають окремий namespace, попередні checkpoints і score ledger
залишаються недоторканими. Після старту підтвердити effective source SHA, LR,
coverage, validation та готовність CSV; final fit не матиме незалежного AP.

### Перемикання єдиного кандидата — 2026-11-12, 08:55 EET

На уточнення користувача непотрібний tiled inference `51bc` було зупинено; він не
записав/не змінив checkpoint, а його final-fit не запускався (межовий `STOP` збережено).
Запущено окремий `c1_continuation` service від точного checkpoint SHA
`c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`. Перша повна
selection-епоха стартувала на 24,201 train-кадрах; source optimizer state збережений,
effective LR `3.1622776601683796e-6`, sampler `flight_time`. Dataset roots у
runtime copy конфігу виправлено на фактичний mounted volume; checkpoint і первісний
source-config лишені незмінними. Service має systemd sleep/lid inhibitor та спільний
GPU lease; треба контролювати лог/coverage/RAM/swap/диск і selection AP. Поточний
status — TRAINING; до завершення validation не робити висновків і не експортувати.

### Відновлення `c1` після suspend/CUDA Xid31 — 2026-11-12

`cvlab-full-data-c1.service` стартувала, але 08:59 EET logind заснув систему на
подію `Lid closed`; power-guard стежив лише за старою назвою unit. Після resume
NVIDIA Xid31 зробив CUDA недоступною; перший checkpoint `last.pt` лишився валідним
source boundary (`epoch=-1`, 812 model tensors, optimizer 2 groups, c1 source SHA).
Train epoch не був committed. Виправлено `cvlab-power-guard` для c1 plan; вимкнено
автозапуск застарілих `cvlab-full-data` і дубльованого trial0 export. Після reboot
CUDA tensor preflight пройшов; volume знову доступний read-only. Коротка невдала
спроба service на boot була через mount до появи UDisks auth agent; після монтування
c1 service вручну піднята повторно.

Стан на 09:41 EET: тільки c1 job активний; initial full official-val inference
перед першим epoch-ом триває. Checkpoint не втрачено; тренування ще не дійшло до
першого gradient epoch.

### Перша full-data епоха `c1_continuation` — 2026-11-12, 11:40 EET

Повний train pass: 24,201/24,201 унікальних кадрів. Initial official-val AP
0.182219; після epoch 0 — 0.191111 (gain +0.008892), AP50 0.368334→0.387005,
AP75 0.161274→0.167546, AP_small 0.072797→0.087055, AP_medium 0.173838→0.177193,
AP_large 0.345070→0.351036. Train loss 12.3002. За класами truck +0.04724 і bus
+0.05125; person −0.00751, bicycle −0.00581, car −0.01208, motor −0.00627.
Перевірка по семи val videos неоднорідна: п’ять зросли, два впали. Це локальна
validation, не Kaggle score; precise public c1 screenshot показує 0.2021478454.

Checkpoint `best.pt` і `last.pt` збережені; coverage підтверджує всі 24,201 кадри.
Після оцінки знято boundary STOP і відновлено лише epoch 1/2 того самого c1 run;
на 11:41 EET служба активна. Перевірити наступний full-val AP; фінальний train+val
не запускати без перевірки selection report. Source c1 незмінний.

### Друга selection-епоха `c1_continuation` — 2026-11-12, 13:18 EET

Повторний full pass: ще 24,201 кадр; cumulative coverage 48,402 presentations.
Official-val AP зростав послідовно: 0.182219 → 0.191111 → 0.196215; epoch-1
loss 12.0753. Від initial до epoch 1: AP50 +0.02636, AP75 +0.01125,
AP_small +0.01501, AP_medium +0.00692, AP_large +0.01213. Class AP delta:
truck +0.0543, bus +0.0560, tricycle +0.0139, motor +0.0013; person −0.0054,
car −0.0093, bicycle +0.0019, van −0.0006. На 7 val families 4 покращились,
3 погіршились. Це не Kaggle score; precise Kaggle score c1 лишається 0.2021478454
(попередньо повідомлялося округлено як 0.206).

На 13:18 EET active job виконує повне native confirmation для 2,846 official-val
кадрів. Epoch checkpoints збережені. `runs/rtdetr_c1_continuation_final/STOP`
залишається перед фінальним train+val, щоб завершити native+tiled gates та підготувати
CSV для користувацького подання перед будь-яким фінальним all-data fit.

### Аналіз зміни AP за класами — 2026-11-12

Фактичні annotations пояснюють частину різниці в сигналах. На train 24,201 кадрів:
car 505,301 boxes, person 328,701, bus лише 9,653, truck 30,498. Вихідні c1
weights бачили 32 train flight-families (16,872 кадри); full pass додав 16 нових
family (7,329 кадрів), у яких було 1,938 bus і 4,835 truck boxes. Перша selection
епоха знизила person/car AP на 0.0055/0.0093, але додала truck/bus 0.0543/0.0560.
Це узгоджується з користю від додаткових family/класового покриття та спільними
компромісами ваг; кореляція не доводить причинність.

Validation також нерівномірна: 2,846 кадрів лише із 7 відеосімейств, а всі 264 bus
boxes зосереджені в одному family. Тому bus AP має високу міжсценову нестабільність;
чотири family зростання проти трьох зниження — не надійний незалежний тест. Наразі
confirmation native ще рахується; tiled gate і рішення про selection report не готові.
Наступний діагностичний крок — локалізувати miss/classification/localization errors
за повними predictions до будь-яких class-weight змін.

### Selection confirmation і проміжний Kaggle export `c1_continuation` — 2026-11-12

Завершено два повні train passes по 24,201 кадру (48,402 presentations). На тій
самій повній official validation (2,846 кадрів, fingerprint
`22cd17773d2567a6d2189bd385478eb7a6ed73c1f2217cdc3ed842dc8858addd`) checkpoint
`224c8cc6f9877a4cb4196c2ec514527964cf18c4f9a8d85031fc09cc234c8f9d` має native
AP 0.1962149 проти source 0.1821884 (gain +0.0140264) і tiled AP 0.1988935 проти
0.1879994 (gain +0.0108940). Обидва full-data selection gates пройдені. Tiled
inference: tile 960, overlap 0.2, NMS 0.6, edge margin 2. Ваги збережені у
`runs/rtdetr_c1_continuation_train_selection/best.pt`; джерело `c1d7fcc83dde`
незмінне. Final train+val не запускався через наявний boundary STOP.

Після системного reboot о 14:11 EET (причину й ініціатора наявні журнали не
встановлюють; чистої shutdown-послідовності в попередньому boot journal немає)
service відновив confirmation із кешованого стану, без повторного навчання.
Перевірка завершилась і selection report збережений у
`full_data/c1_continuation/selection_report.json`. Окремий test tiled export для
checkpoint `224c8cc6f987` завершено. Готовий файл
`submissions/rtdetr_224c8cc6f987_c1_epoch1_selection_tiles.csv`; checkpoint SHA256
`224c8cc6f9877a4cb4196c2ec514527964cf18c4f9a8d85031fc09cc234c8f9d`, CSV SHA256
`6385f4951d2a012320c844b765a7d4badd588a2ffa447c4a34fd4a24fe13abb8`. CPU checker
підтвердив 3,811/3,811 IDs у template order, 1,856,650 detections, жодного
порожнього кадру, валідний формат. Metadata звіряє checkpoint і обидва full-val
gains. Початково CSV позначено готовим до ручного Kaggle upload; фактична
подальша оцінка користувача зафіксована нижче.

### Public regression і корекція selection policy — 2026-11-12

Користувач повідомив Kaggle score `0.1919` для
`rtdetr_224c8cc6f987_c1_epoch1_selection_tiles.csv`, приблизно `-0.0141` від
попереднього c1 CSV із пізніше уточненим score `0.2021478454` (раніше округлено
до `0.206`). Регресія становить приблизно `-0.01025`. Новий результат додано до
`experiments/leaderboard_results.json`; попередній c1 лишається public-best.
Checkpoint regression збережений як експеримент і не буде source для наступного
навчання.

На тому самому повному official-val cohort pooled native/tiled AP зріс на
`+0.0140/+0.0109`, але class AP змінився нерівномірно: person `-0.0055`, car
`-0.0093`, van `-0.0006`; truck `+0.0543`, bus `+0.0560`, tricycle `+0.0139`,
motor `+0.0013`. На семи val families покращились чотири, погіршились три. На
test CSV прогнозовані person/car кількості змінились `582,034/301,173 →
465,081/275,312`, тоді як bicycle/bus/motor змінились `175,112/72,491/290,701 →
288,199/94,113/332,890`; загальна кількість зросла лише на 4,587. Це counts
прогнозів, не виміри TP/FP, але вони показують великий класовий зсув, який pooled
AP приховав.

Обидва CSV мають однакові tiled options і test IDs/order; локальна перевірка не
виявила export-format чи inference-config mismatch. Цей випадок показує, що pooled
COCO AP на 7 families недостатній як public-score proxy. Наступне навчання не
починати від регресивного checkpoint; перед ним підготувати family-grouped OOF
оцінювання та відбирати з урахуванням class/family deltas, не pooled AP окремо.
Точна Kaggle metric implementation лишається непідтвердженою.

### Діагностика напрямку оновлення ваг — 2026-11-12

Завершено inference-only line search уздовж одного фактичного вектора між source
`c1d7fcc83dde` і checkpoint `224c8cc6f987`: `theta(alpha)=theta_source +
alpha*(theta_candidate-theta_source)`. На фіксованому збалансованому probe із 512
кадрів і тих самих 7 validation families pooled COCO100 mAP був `0.20747`,
`0.21445`, `0.22063`, `0.22243`, `0.22249` для alpha `0`, `.25`, `.5`, `.75`,
`1`. Macro family AP ішов `0.20908 → 0.21516 → 0.21914 → 0.22094 → 0.22099`.
Тобто спостережене оновлення спрямоване на покращення цього локального AP proxy,
але майже насичується біля `.75`; це не випадкове блукання параметрів.

Водночас alpha 0→1 погіршив person AP `0.25088→0.24350` і car
`0.47901→0.46928`, підвищив truck `0.24587→0.31731` і bus
`0.04020→0.09540`; 4 із 7 family покращились, 3 погіршились. Probe не незалежний
від cohort, на якій обирали endpoint. Головне: endpoint підняв full local AP,
але отримав Kaggle `0.1919` проти точнішого source score `0.2021478454`. Отже, цей sweep доводить
лише напрямність щодо локального COCO objective, а не щодо Kaggle. AdamW і далі
оптимізує train detection loss; AP/Kaggle напряму не диференціюються.

Звіт/CSV діагностики зберігаються локально у
`$CVLAB_LOCAL_WORK/diagnostics/c1_weight_line_search/`. На перевірці 15:45 EET
активного training-процесу не було; зовнішній data volume був read-only, тому
навчання не відновлювали. Checkpoint-и збережені; регресивний кандидат не
використовувати як source. Наступне надійне дослідження напрямку — family-grouped
OOF для source та парних кандидатів, окремо аналізуючи класи/families; без нового
Kaggle score не робити висновок, що локальний alpha максимум покращить leaderboard.

### Напрямне уточнення ваг і новий кандидат — 2026-11-12

Алгоритм: зберегти public-scored source `c1d7fcc83dde` та continuation
`224c8cc6f987`; оцінити точки `theta(alpha)=theta_source + alpha *
(theta_candidate-theta_source)`; вибрати проміжний alpha; вимагати повні native і
tiled gains щонайменше `0.001` проти source; лише тоді експортувати test CSV.
Початковий probe на 512 кадрах дав зростання mAP `0.20747→0.22249` для alpha
`0→1`, з plateau біля `.75`. Через невідповідність локального AP і Kaggle на
endpoint, alpha `.5` наперед обрано як shrinkage-кандидат для повної перевірки,
не як гарантоване leaderboard покращення.

Відтворюваний runner: `scripts/export_c1_interpolated_candidate.py`; він тримає
shared GPU lease, пише immutable interpolated checkpoint/report, рахує full native
та tiled validation на тому самому official cohort і створює CSV лише якщо обидва
gains проходять gate. Команда наведена в `experiments/c1_direction_search.md`.
Фактичний результат alpha `.5` додати після завершення; не позначати CSV як
SUBMITTED/SCORED без відповіді Kaggle. Runner не підміняє family-grouped OOF, а
цей крок є контрольованим тестом гіпотези про часткове повернення до public-best.

### Web review of local commit e6fbd76 — 2026-11-12

Reviewed the full-data continuation and direction search; 69 targeted CPU tests
passed in an isolated checkout. Detailed findings and ordered actions:
[training efficiency review](reviews/2026-11-12-training-efficiency.md).

The confirmed inefficiency is selection: 48,402 presentations improved official
native/tiled AP while the recorded public result fell from 0.2021478454 to
approximately 0.1919. Truck+bus explain about 98% of pooled native AP gain, with
all validation bus boxes in one family. Exact Kaggle/local metric parity remains
unverified; do not infer it from the custom metric template name.

Keep c1d7fcc83dde as the user's approved training source and retain the final-fit
STOP barrier. Before another long run: bind promotion to approved source/outcome,
verify interpolation baseline identities, reuse exact evaluation receipts, and
align monitoring/power protection with the actual active plan. Preserve a healthy
current job. This review changes no laptop process and triggers no GPU launch or
automatic submission. Local agent: report the active plan/source SHA, completed
phase timings/cache hits, and which review items were applied before the next
training launch.
