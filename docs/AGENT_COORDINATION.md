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

### Stronger selection and lab audit implemented — web Codex, 2026-11-12

Implemented the user's requested replacement of pooled-AP/weight-direction
promotion. See [selection protocol](selection_policy.md) and
[laboratory evidence checklist](LAB_PROGRESS.md). Full-data `best.pt` now needs
distributed flight-level improvement, positive approximate family-bootstrap
lower gain, class/size regression limits and leave-one-family-out support.
Efficient-search confirmation and interpolation export use the same guards.
Known public regressions are blocked by checkpoint identity. Legacy AP-only
reports cannot authorize new export/final fit; preserve their evidence.

Shared evaluator receipts reuse committed native training validation, including
the original source identity, and avoid repeated GPU inference only when exact
weights/records/inference/software match. Interpolation measures its actual
baseline, checks inherited exposure and uses immutable endpoint-specific output.
Full-data harmful epoch updates write a persistent `selection_halt.json`; a
restart cannot quietly continue the rejected recipe. The old auto-launch
watchdog monitors healthy work but cannot fall back to the previous search when
a named full-data plan has recorded status. The dedicated approved-plan service
must own c1 continuation. Lightweight monitoring accepts the actual log/state.

Local agent: apply at the next safe saved boundary, preserving active healthy
work, GPU lease, original c1 weights and prior CSVs. Inspect the current owner,
actual plan/source SHA and storage first. Do not blindly rerun the old two-epoch
plan or clear its STOP/halt after a quality rejection. Reassess existing committed
metrics under the new guards before spending another epoch. A new controlled
recipe gets a new plan, retaining learned c1 initialization and compatible
optimizer state. Do not launch a broad alpha grid from the rejected endpoint.
Exact competition metric parity is still unknown. These guards do not guarantee
0.28+; record actual independent measurements and later observed Kaggle scores.

Local audit command: `python3 scripts/check_lab.py --work "$CVLAB_LOCAL_WORK"
--out "$CVLAB_LOCAL_WORK/checks/status.json"`. It reads metadata/current process
argv without tensor loading and flags inherited flight-family leakage. Report
which baseline/tuning/loss runs, comparable metrics, failure examples and actual
OOF outputs exist; configs alone do not fulfill those requirements. Keep raw
runtime status local. The academic report must be written by the students.

Validation: final full CPU suite passed, 307 tests; Ruff and diff whitespace
checks passed. Tests cover real tiny warm-start cache reuse, rejection of
one-flight/class regressions, identity-separated interpolation, source rejection,
restart halt, inherited leakage and watchdog ownership. No live laptop training
or GPU execution was performed by the web session. Mid-epoch sampler recovery
and hard full-epoch deadlines remain unimplemented.

### Training loop optimization — web Codex, 2026-11-12

The user asked whether training itself is becoming more efficient. The earlier
commit optimized selection and duplicate evaluation; this change additionally
replaces the per-minibatch loss readback path with a shared on-device detached
loss accumulator. Finiteness is checked before each optimizer update; clipping,
AMP, accumulation normalization and learned optimizer state are preserved.
The per-batch reference remains available for measurement. Training history and
pending validation now carry pipeline frames/s, host loader wait, update-boundary
and loss-readback counts, with explicit timing scope. No actual GPU speedup or
new AP is claimed from this CPU-only web environment.

See [training performance](training_performance.md). The source-preserving
`scripts/benchmark_training.py` compares reference/optimized readback and physical
batch sizes using the actual shared training loop. It holds the GPU lease,
checks RAM/other compute, bounds CUDA allocation, stress-tests dense GT, restores
source state per case, alternates order and reports measured throughput only.
It saves no trained model and submits nothing. The full-data launcher can apply
a measured `--batch-size 2` under a new plan, preserving the established effective
batch via accumulation. Validation stays batch one; cache identities include
the actual inference batch; full-epoch checks count frames correctly.

Local agent: keep the healthy owner until a safe saved boundary; pull this code
then run the documented short benchmark from immutable c1 when the GPU is free.
If the measured gain/VRAM headroom is absent, keep batch one. A physical-batch
change can alter RT-DETR loss normalization/denoising despite equal effective
batch, so it still needs the robust held-out gates in a new c1-initialized plan.
Record the benchmark summary, actual training frames/s and held-out class/family
deltas next. Do not overwrite source config/weights, clear a rejected plan's
halt/STOP, infer public gain, or start a second concurrent training model.

Validation for this change: final full suite 317 passed, 1 CUDA equivalence test
skipped because no GPU is attached to the web executor. CPU tests verify exact
reference/optimized parameter and AdamW-state equality, nonfinite-loss rejection
before updates, accumulation tails, source-preserving real benchmark execution,
physical-batch coverage and validation-batch/cache identities. Actual GPU frames/s
and AP improvement remain unmeasured. Local benchmark results must supply them.

### Локальна перевірка плану та benchmark throughput — 2026-11-12, 18:16 EET

Виконано CPU artifact audit, власник GPU звірений безпосередньо. Новий alpha=.5
процес не завершився: збережено лише повний native результат `0.1916993` проти
source `0.1821884` (gain `+0.0095108`); процес зник до tiled gate, тому CSV немає
і цей результат не приймається за selection. До цього стартував batch benchmark;
два виправлення в `scripts/benchmark_training.py` (CUDA device index і копія
adapter metadata на кожен repeat) опубліковані окремими commits.

Завершений source-preserving benchmark використав checkpoint
`c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`, effective
batch 4, 20 optimizer-update boundaries на вимір і 2 repeats. Median pipeline
throughput: batch1/accumulation4 `3.9981` frames/s з readback per minibatch;
batch1/accumulation4 `4.0125` з readback per update (`+0.36%`); batch2/accumulation2
`4.3627` per minibatch (`+9.12%`); batch2/accumulation2 `4.5668` per update
(`+14.22%`) проти reference. Report позначає batch2/per-update як measured faster,
але якість не вимірювалася. Source checkpoint перевірено тим самим SHA до/після;
`models_saved=false`. Це пропонує окремий c1-initialized batch2 plan для robust
class/family validation, але не дозволяє перемикати поточний plan або оголосити
поліпшення AP/Kaggle. Loss `batch2` також відрізнявся від `batch1`; фізичний batch
може міняти RT-DETR loss/denoising.

Локальний JSON лишився у `WORK/benchmarks/training_20261112T161141_8e1c6551.json`;
machine telemetry/checkpoint-и не комітяться. Перевірка після benchmark: активного
training/evaluation процесу немає; `cvlab-full-data-c1` та sampling service inactive,
watchdog active з pause marker, final-fit STOP лишився. Work data volume read-only;
final train+val не запускався. Наступне рішення: зберігати c1 source й окремим
plan наздогнати throughput batch2, але запускати лише з robust source-bound gates,
збереженим STOP для final-fit і writable storage. OOF/public score не виводити з
цього benchmark.


### Локальний стан, аудит і failure diagnostics — 2026-11-12, 18:25 EET

Перевірено безпосередньо на локальній машині: навчальних/validation-процесів
немає; CUDA доступна, compute GPU вільний. User-служби `cvlab-watchdog.service`
і `.timer` встановлені та активні; watchdog продовжує внутрішню перевірку раз
на 60 с. Його поточний стан — `paused` через checkpoint-boundary STOP;
збережено STOP для `rtdetr_c1_continuation_final` і `rtdetr_full_data_final`.
Старий загальний часовий бюджет завершився 2026-11-11. Новий запуск не
допускається політикою watchdog: runtime work має близько 7.25 GiB вільного
місця при мінімумі 8 GiB. Підготовлений data volume досі read-only. Checkpoints
не змінювалися й не видалялися.

CPU audit `scripts/check_lab.py` для фактичного data work підтвердив
`train/val/test/folds` під `prepared/`; окремий runtime audit очікувано не має
копії `prepared/`. EDA summaries уже існують: train — 24,201 кадр / 1,105,516
об'єктів, validation — 2,846 / 114,132; класи нерівномірні (validation bus —
264 об'єкти, train bus — 9,653). Поточні локальні таблиці failure diagnostics
для checkpoint `224c8cc6f9877a4cb4196c2ec514527964cf18c4f9a8d85031fc09cc234c8f9d`
збережені в `WORK/diagnostics/c1_native_failure_audit/`: `class_counts_iou50_score025.csv`,
`examples_by_error_and_class.csv`, `README.json`. Вони покривають усі 2,846
validation кадрів і дають фіксований поріг score=0.25, IoU=0.5; це описова
діагностика, не AP і не незалежна оцінка, бо ця cohort брала участь у відборі.
Найбільші помилки за таблицею: motor має 5,346 wrong-class FP, van — 3,131,
bicycle — 2,631; bus recall при цьому порозі 0.136. Є до трьох різних family
прикладів на клас/тип помилки. Prediction cache — повний, 2,846 image IDs.

Останній записаний tiled confirmation у watchdog/log — local mAP `0.198893`;
це не Kaggle score. Не створено нового READY CSV і не подано нічого на Kaggle.
Далі: звільнити місце до admission-порогу та відновити writable data/work
storage; зберегти STOP до нового source-bound плану з class/family gates.
Фінальний train+val лишається без незалежної validation-оцінки.


### Новий c1 batch-2 selection — 2026-11-12, 19:25 EET

Після виміряного GPU throughput запущено окремий `c1_batch2` plan від незмінного
source SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`:
24,201 train frames на кожен із двох повних selection epochs, physical batch=2,
accumulation=2 (effective batch 4), full official validation і далі tiled
source-bound gates. Фінальний `train+val` output має свій STOP, тож цей запуск
його не починає. Раніше завершений `c1_continuation` і public-best артефакти
залишені без змін.

На 19:25 служба `cvlab-c1-batch2` активна, CUDA preflight і читання ваг/data
пройшли. Йде full-validation baseline перед першою епохою; перших train batches,
loss і measured FPS ще немає. GPU був активно зайнятий, температура безпечна;
RAM доступна вище 4 GiB тренера, диск вище 8 GiB admission порогу. NTFS data
volume лишається read-only, але модель читає його; робочі виходи — на writable
Linux work. Watchdog перейшов у `monitor`; початковий quiet-process сигнал
викликаний його старим log path. Запущено локальне журналювання systemd unit у
`WORK/logs/c1_batch2_training.log`, аби наступні stall/progress перевірки бачили
новий процес. Після baseline стежити за frame/batch прогресом, VRAM/RAM/swap,
температурою, вільним місцем і checkpoint-ами; оцінювати тільки повну validation
та tiled family/class gates. Немає нового CSV чи Kaggle score.


### Негайна пауза c1 batch-2 — 2026-11-12, 20:50 EET

На пряме прохання користувача service `cvlab-c1-batch2` негайно заморожено через
systemd cgroup freezer; `FreezerState=frozen`, unit лишається active, stdout
collector і watchdog не зупиняв. STOP-файл selection-run збережений, тому після
відновлення процесу він доробить цю епоху/validation і не почне наступну. На
момент паузи останній рядок — epoch 0, batch 11,500/12,101, mean loss 12.5804.
Після замороження GPU utilization впав до 0%, VRAM 5.2 GiB лишається зарезервованою;
checkpoints c1 та public-best не змінені.

Важливе обмеження відновлення: `last.pt` у `WORK/runs/rtdetr_c1_batch2_train_selection/`
ще збережений на старті (до першого epoch update). Точний mid-epoch optimizer/RNG
контекст є лише в RAM замороженого процесу й не є reboot-safe. Якщо процес/машина
буде завершена, наступний запуск повторить цей epoch від останнього дискового
checkpoint; source c1 залишається неушкодженим. Не видаляти STOP або заморожений
unit, доки користувач явно не відновить роботу.


### Відновлення після shutdown і новий c1 selection — 2026-11-13, 21:51 EET

Перевірено journal і поточну систему після reboot: попередній frozen unit було
зупинено системою 2026-11-12 о 20:53 EET. На диску немає завершеного epoch
checkpoint або `history.jsonl`; `last.pt` лишився стартовим. Незбережені 11,500
batch-ів не відновлюються після reboot. Старий `c1_batch2` план/STOP збережені як
історія; public-best source `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`
перевірений і незмінний. Старий увімкнений batch-1 c1 service вимкнено, бо його
плановий deadline вже минув.

За чинним full-data протоколом незавершений epoch доводиться повторити від
останнього committed checkpoint, оскільки mid-epoch cursor recovery не
реалізований. Створено окремий bounded `c1_batch2_retry1` plan/service від c1:
2 full selection epochs, batch=2/accumulation=2, full official validation,
source-bound native+tiled family/class gates, новий восьмигодинний deadline.
Service увімкнений для відновлення після boot і має sleep inhibition; auto-retry
обмежений systemd start limit. Фінальний train+val має окремий STOP. Work storage
проходить 8 GiB admission, підготовлені manifests читаються з read-only NTFS.

О 21:51 служба пройшла CUDA/data preflight, завантажила c1 і рахує початкову
повну validation оцінку; train batches і новий AP ще не записані. Watchdog active
і бачить процес як `monitor`. CSV не готовий, на Kaggle нічого не подано.
Наступний критерій — завершений повний epoch, збережений optimizer/checkpoint,
повна validation і robust gates; loss окремо не вважати доказом якості.


### Початкова validation і train progress — 2026-11-13, 22:02 EET

У новому `c1_batch2_retry1` завершено повну native validation для того самого
official cohort із 2,846 кадрів: initial local COCO100 mAP `0.1822042`, pipeline
inference `14.7098` кадр/с. Це baseline до gradient updates, не Kaggle score.
Epoch 1/2 почалася на всіх 24,201 train frames; на 22:02 записано batch
250/12,101 з cumulative mean loss `12.9210`. Ще немає завершеного epoch
checkpoint, post-training AP чи нового CSV. Batch=2 має інші batch dynamics;
висновок лише після повної validation та source-bound gates.

### Результат c1 batch-2 retry — 2026-11-14, 16:18 EET

Завершено один повний selection epoch на всіх 24,201 train-кадрах (batch=2, accumulation=2); training loss `12.6019`, measured pipeline `4.96` кадр/с, потім повна native validation на 2,846 official-val кадрах дала mAP `0.1917016` проти source baseline `0.1822042` (pooled gain `+0.0094974`). Це локальна native-метрика, не Kaggle score.

Selection відхилив epoch: `class_regression`, `family_lower_bound_not_positive`, `family_regression`. Зокрема bicycle −0.01883, car −0.00521, person −0.00597; сімейний bootstrap lower bound −0.00630, 5/7 сімейств позитивні. Plan `c1_batch2_retry1` перейшов у `selection_halted`; другий epoch, tiled confirmation і final train+val не запускались, CSV не створено. Служба вимкнена й inactive; watchdog timer active; GPU вільний. Source SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4` збережений. Run `WORK/runs/rtdetr_c1_batch2_retry1_train_selection/`: `best.pt` source-best SHA256 `cbe10f0089d6475eb2eef0879a61ad4d2b8f80f7bf48946e73f5ea0e8823606c`; `last.pt` epoch candidate SHA256 `7feda78d6de7197b5598c996d980188b591b46001980d90bba77847f21973ec3`.

Рішення: не подавати epoch candidate і не повторювати цей самий recipe. Наступний крок — аналіз class/family error evidence та заздалегідь визначена змінена гіпотеза з тими самими robust gates. Поточний Kaggle public-best залишається `0.2021478454`; baseline `0.2216324580` і ціль `0.28+` не досягнуті.

### Новий повнодатасетний VFL coefficient experiment — 2026-11-14, 16:29 EET

Після зупинки попереднього recipe створено окремий source-bound план `c1_vfl15_fulltrain` від незмінного c1 SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`. Змінено один параметр loss: RT-DETR `weight_loss_vfl` 1.0→1.5; matcher, bbox/GIoU coefficients, LR `3.1622776601683796e-6` та архітектура лишаються як у source. Warm-start зберігає сумісні optimizer/scaler/RNG. План має два повні selection epochs (24,201 кадрів кожен), batch=1/accumulation=4, повну незалежну official validation на 2,846 кадрах із сімома невиданими сімействами та робастні class/family gates. Лише після їх проходження дозволено один final-fit epoch на всіх 27,047 розмічених train+val кадрах; seen validation тоді не буде незалежною оцінкою. Дані прочитані з доступного read-only NTFS mount; checkpoint-и пишуться у локальний Linux work storage.

CUDA preflight успішний; service `cvlab-c1-vfl15` enabled/active із sleep inhibition, watchdog і power-guard timers active. На 16:29 процес завантажив c1 і рахує повний baseline validation перед оновленнями ваг. Навчальний batch ще не почався, тому AP/loss gain, прийняття recipe і CSV поки не підтверджені. Дивитися `WORK/logs/c1_vfl15_fulltrain.log` та `WORK/full_data/c1_vfl15_fulltrain/state.json`. Результат попереднього batch-2 recipe лишається відхиленим і не використовується як initialization.

### Результат c1 VFL coefficient experiment — 2026-11-14, 18:28 EET

`c1_vfl15_fulltrain` завершив один повний selection epoch: усі 24,201 train frames,
6,051 optimizer-update boundaries, training loss `15.3078`, pipeline `4.04` frames/s
(5,990 s). Початковий native baseline на повній official val (2,846 frames) був
`0.1822121`; кандидат із `weight_loss_vfl=1.5` отримав `0.1891387`, pooled gain
`+0.0069266`. AP50 gain `+0.01440`, AP75 `+0.00502`, AP_small `+0.01519`. Це локальні
COCO metrics, не Kaggle score.

Кандидат відхилений: `class_regression`, `family_lower_bound_not_positive`,
`too_few_improving_families`, `family_regression`, `gain_depends_on_one_family`.
Позитивні 4/7 сімейств; equal-family bootstrap lower bound `−0.010556`,
leave-one-family-out minimum `−0.003765`. Class AP deltas: person `−0.00457`,
bicycle `−0.00934`, car `−0.01213`, van `+0.00071`, truck `+0.04107`,
tricycle `+0.00011`, bus `+0.04696`, motor `−0.00739`. Source c1 лишився
незмінним (SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`).
Candidate `last.pt` SHA256:
`00a4a9e0fc6b340304714f1b6d8601468dab069805b2fbee8bf9da83b29bade9`;
source-best receipt `best.pt` SHA256:
`e301d659fcd1bf3222fa797d468ece105e98dd9e2703ec66fdac820e35e1d742`.
Plan status `selection_halted`; second epoch, tiled gate, final train+val та CSV
не запускались. Service `cvlab-c1-vfl15` disabled/inactive, CUDA idle; periodic
watchdog та power-guard timers active. Disk admission margin близький до мінімуму.
Kaggle score не змінювався; CSV не готовий.

### Наступний source-bound запуск від c1 з нижчою LR — 2026-11-14, 18:38 EET

Попередній `c1_vfl15_fulltrain` зупинено robust selection після epoch 0; його
checkpoint не використовується як source, halt збережено, service вимкнено.
Створено новий окремий `c1_lrhalf_fulltrain` plan від public-best c1 з тим самим
повним SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`;
`source_c1_lrhalf/best.pt` є посиланням на незмінний checkpoint. Єдина зміна проти
c1-рецепта — LR знижено удвічі з `3.1622776601683796e-6` до
`1.5811388300841898e-6`; VFL повернуто до source значення 1.0. Batch=1,
accumulation=4, три повні selection epochs по 24,201 кадру максимум; кожна має
повну official validation із 2,846 кадрів і ті самі class/family gates. Один
final train+val epoch на 27,047 мітках можливий лише після успішного відбору.

На 18:38 preflight пройшов, `cvlab-c1-lrhalf` active із sleep inhibition,
CUDA/data manifests доступні, runtime state `full_train_selection`; завантажено c1
і розпочато початкову validation. Train batches та оцінка цього recipe ще не
підтверджені. Watchdog та power-guard timers active; service має bounded 12h
budget. Вільне Linux work storage невелике, тому перевіряти його протягом запуску.
Runtime: `WORK/logs/c1_lrhalf_fulltrain.log`,
`WORK/full_data/c1_lrhalf_fulltrain/state.json`.

### Зміна основної стратегії: YOLO11s та continuous control — 2026-11-14

За прямим дорученням користувача змінити стратегію для підвищення якості,
наступний головний експеримент — **YOLO11s COCO на 1280 px**, один фіксований
курс на всіх 24,201 train frames із повною незалежною official val. Це інша
архітектура, а не новий одноепоховий VFL/LR trial від c1. Джерело нового YOLO —
офіційні COCO weights; не брати довільні VisDrone-trained weights із невідомим
перетином із competition val/test. RT-DETR c1 і його public 0.2021478454
зберігаються як контроль. Baseline 0.2216324580 і ціль 0.28+ не досягнуті.

У коді підготовлено `scripts/start_yolo_training.py`, optional extra `[yolo]`,
streamed conversion із symlinks замість копіювання архіву, training-only mask
ignored regions зі збереженням positives, stock Ultralytics optimization/EMA,
full common COCO100/500 validation серіалізованих ваг після кожної епохи.
При `--export-every 2` проміжний common-best CSV формується кожні дві епохи;
однакові hashes не переекспортуються. `QUALITY_REVIEW_REQUIRED` не означає
promotion, Kaggle submission або public gain. Потрібне повне порівняння з c1,
class/family діагностика й структурна перевірка. Автоматичного submit немає.

Виявлена помилка попереднього allocation: batch-2 retry покращив native AP
0.1822→0.1917, проте class/family promotion gate завершив навчання після першої
епохи. Новий continuous policy не обриває курс за одним таким діагнозом:
зберігає той самий optimizer/scheduler і last.pt, чекає кількох validation checks
без прогресу; фінальні promotion gates не прибрано. Для RT-DETR наявний strict
plan переходить лише явно через `--training-policy continuous`; його LR,
batch, кількість запланованих epochs і deadline при цьому не змінюються.

**Локальному агенту:** перевір фактичні процеси, service, storage і стан
`c1_lrhalf_fulltrain`; Git не є live-з'єднанням. Якщо healthy epoch працює,
дочекайся збереженої межі та повної val. Для пріоритету YOLO створи intentional
STOP у selection run до початку наступної епохи, збережи checkpoint і не
перезапускай fallback trials. CPU-only `--prepare-only` можна виконати паралельно.
Далі встанови optional backend у чинний CUDA venv і запусти курс за
[точними командами](yolo_strategy.md) під shared GPU lease та sleep inhibition.
Не запускай обидва GPU jobs одночасно; не вбивай активний процес заради оновлення.
На старті виміряй фактичні frames/s, RAM/VRAM, після першої повної епохи — AP;
перший low AP після зміни 80→8 classes не є автоматичною причиною reset.

Перевірено тут на CPU: реальні stock forward/backward, збереження optimizer,
resume наступної епохи без зміни recipe, common validation, ignored masking,
class mapping і CSV. GPU throughput/AP та новий Kaggle score не виміряні.
Інструкції: [YOLO strategy](yolo_strategy.md),
[continuous RT-DETR](continuous_training.md). Після запуску додай датовану
відповідь із фактичними вимірюваннями та станом, без приватних mount paths.

### Відновлення LR-half курсу після power-button shutdown — 2026-11-14, 20:52 EET

Після `git pull --ff-only` застосовано нові правила `continuous_training.md` до
вже створеного `c1_lrhalf_fulltrain`, без нового recipe: той самий c1 source SHA
`c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`, config LR
`1.5811388300841898e-6`, batch=1/accumulation=4, три selection epochs і попередній
deadline `2026-11-15 06:38 EET` збережені. Явно прийнято `--training-policy
continuous`: повна validation залишається після кожної епохи, class/family evidence
і promotion gates зберігаються, але окрема regression після першої епохи більше не
обриває fixed course. State фіксує перехід `strict`→`continuous`; checkpoint c1
архівовано, user STOP не створювався й не знімався.

Journal показав `systemd-logind: Power key pressed short` о 19:48:43 EET та штатний
poweroff о 19:49:13, який завершив active service під час epoch 0 на batch
17,250/24,201. Це було до checkpoint межі; `last.pt` лишився source-initialization
checkpoint (epoch −1), `history.jsonl` не має завершеної епохи. Mid-epoch cursor
відсутній, тому невідcommitована частина втрачена і epoch 1 повторюється від c1;
це не випадковий restart ваг і не нова модель. Після boot auto-start тричі не
пройшов preflight: NTFS data volume був unmounted, потім disk admission був нижче
8 GiB. Нічого не repair-или. Volume змонтовано read-only; manifests доступні.
Для відновлення 8 GiB admission видалено лише rebuildable VS Code C/C++ browse DB
та precompiled-header cache; CVLab checkpoints/data/evaluation evidence не чіпали.

Service `cvlab-c1-lrhalf` знову enabled/active, тепер sleep/lid та short power-key
inhibitor-и встановлені. Watchdog і power-guard timers active. На 20:52 новий
process уже виконував epoch 1 повторно, batch `250/24,201`, mean loss `12.7319`;
після reboot ще немає завершеної епохи чи нового AP. GPU активний, доступно понад
8 GiB work disk; нового CSV/Kaggle score немає. Runtime лог і state лишаються в
`WORK/logs/c1_lrhalf_fulltrain.log` та
`WORK/full_data/c1_lrhalf_fulltrain/state.json`.

### Continuous c1 LR-half: epoch 1 validation and epoch 2 training — 2026-11-14, 23:53 EET

У resumed continuous plan завершився повний перший train epoch після перезапуску:
24,201 frames, mean loss `12.40237`, pipeline `4.6541` frames/s (5,199.9 s).
Повна native official validation на 2,846 frames дала mAP `0.19109594` проти
початкових `0.18221211` (gain `+0.00888383`, локальна метрика, не Kaggle score).
Робастний promotion decision позначив class/family regression; позитивні лише
4/7 сімейств, bootstrap lower `−0.008386`, однак згідно з adopted continuous
policy це діагностика, не зупинка курсу. Public-best c1 не змінюється.

Друга повна train epoch також завершена: log loss `12.2018`; на 23:53 виконується
її повна validation. State лишається `full_train_selection`, policy `continuous`,
checkpoint boundary збережений, service/watchdog/power-guard active. План ще має
до трьох selection epochs; final promotion gates, tiled confirmation і train+val
fit не пройдені/не запускались. CSV немає. Поточний log/state:
`WORK/logs/c1_lrhalf_fulltrain.log`,
`WORK/full_data/c1_lrhalf_fulltrain/state.json`.

### Continuous c1 LR-half: epoch 2 result — 2026-11-15, 00:55 EET

Другий completed epoch у fixed continuous course: усі 24,201 train frames, loss
`12.21188`, pipeline `4.7107` frames/s (5,137.4 s). Full official native validation
(2,846 frames) дала mAP `0.19618139`, gain `+0.01396928` проти c1 baseline
`0.18221211`; це локальний AP, не Kaggle. Відносно попереднього epoch local mAP
підріс з `0.19109594`; loss зменшився з `12.40237` до `12.21188`.

Class/family promotion gate знову не прийняв кандидата: positive families `4/7`,
family-bootstrap lower `−0.003531`, class/family regression. За новим continuous
policy курс не обривається на цьому gate. На 00:55 epoch 2 (третя selection epoch)
активна, batch `15,250/24,201`, cumulative `mean_loss=12.1959`; повну validation
ще не завершено. Service, watchdog і power-guard active; CSV та новий Kaggle score
відсутні. Runtime details у `WORK/logs/c1_lrhalf_fulltrain.log` і
`WORK/full_data/c1_lrhalf_fulltrain/state.json`.

### Continuous c1 LR-half: third epoch complete — 2026-11-15, 01:39 EET

Третій/останній predeclared selection epoch завершився на всіх 24,201 train frames:
loss `12.1392499`, training pipeline `4.3998` frames/s (5,500.4 s). Native
mAP на всіх 2,846 official-val кадрах — `0.19759198`, pooled gain `+0.01537987`
від c1 baseline `0.18221211`. Candidate `best.pt` SHA256
`7f910f3517e948c5784063136d90f1f6134a5a6bc39968f73a6235f6772645b5`;
`last.pt` SHA256 `8aa28789ac98f4c43efdf840be36bfb52420c198cd2ee943e161910bf2282b09`.

Навчальний selection gate має `class_regression`, `family_lower_bound_not_positive`,
`too_few_improving_families`, `family_regression`: позитивні 4/7 сімейств,
family bootstrap lower `−0.001965`. Continuous policy дозволив завершити три
епохи; AP приростав `0.19109594`→`0.19618139`→`0.19759198`, але це локальний
native результат, не public Kaggle. Після epoch 3, о 01:38, почалась окрема full
native confirmation (`CONFIRM native`, 2,846 кадрів) перед остаточним promotion
рішенням. На час запису confirmation ще працює; tiled gate, train+val final fit,
CSV та новий Kaggle score відсутні. Source c1 SHA збережений. Service і періодичний
watchdog активні. Цю перевірку опубліковано окремою гілкою
`experiment/c1-lrhalf-continuous`, відгалуженою від поточного `main`.

### YOLO11s full-data course started — 2026-11-15, 12:08 EET

Звірено локальні `main` (на `2f9d8ca`) і `experiment/c1-lrhalf-continuous` (на `e81adac`); код YOLO є від `f9303b6`. RT-DETR процесів більше немає. У log зафіксовано epoch 3 native mAP `0.197592` та початок окремої native confirmation, але її підсумкової метрики/receipt не знайдено; отже підтвердження не зараховується. Public-best c1 з Kaggle `0.2021478454` збережений.

Data-volume доступний тільки read-only. YOLO dataset CPU-підготовку завершено з усіма 24,201 train та 2,846 official-val кадрами; сімейства польотів незалежні між split-ами (49 train, 7 val). `ultralytics==8.3.250` встановлений у чинному середовищі; Torch/TorchVision CUDA `2.6.0+cu118 / 0.21.0+cu118` після встановлення лишилися працездатними. Навчання стартувало з офіційних COCO `yolo11s.pt` за зафіксованим рецептом у `cvlab-yolo11s.service` із sleep/lid/power-key inhibition. Watchdog та power-guard timers активні.

Перший вимір ранньої стабілізованої ділянки: близько `7.7 batches/s` (фізичний batch 2, тобто приблизно `15.4 frames/s`); проєкція train-частини епохи — близько 26 хв, до повної validation ще не виміряно. Це попередня швидкість за частковою епохою; фактичний повний час/throughput запишеться після epoch 1. На час запису epoch 1 працює, метрик AP/CSV ще немає. Проміжний перерваний asset-download не використовувався як initialization; фінальний процес завантажив і прочитав офіційні COCO ваги. `c1_lrhalf` STOP збережений; наступні RT-DETR епохи/final-fit не запускалися.

### YOLO11s first full epoch and independent validation — 2026-11-15

Перший повний курс epoch (Ultralytics index `0`) пройшов усі 24,201 train frames на офіційних COCO `yolo11s.pt`, imgsz=1280, physical batch 2 / nominal effective batch 16. Виміряний train pipeline — 23:44, `~17.0 frames/s`. Train loss components: box `1.6065`, cls `1.0952`, dfl `1.1602`. Чинний CUDA/Torch environment залишився незмінним. Після checkpoint виконано full independent official validation на 2,846 кадрах; family split train/val не перетинається.

Common COCO results (локальні, не Kaggle): COCO100 mAP `0.181069`, AP50 `0.364102`, AP75 `0.156348`, AP_small `0.106895`; COCO500 mAP `0.181879`, AP_small `0.107342`. Inference pipeline на full val: `10.61 frames/s`; окреме CPU COCO matching/aggregation займає додатковий час. Per-class COCO100 AP: person `0.24665`, bicycle `0.17882`, car `0.36904`, van `0.18143`, truck `0.15169`, tricycle `0.19320`, bus `0.00248`, motor `0.12524`. Per held-out family AP: `uav0000086 0.22665`, `uav0000117 0.17471`, `uav0000137 0.27800`, `uav0000182 0.17725`, `uav0000268 0.08246`, `uav0000305 0.20390`, `uav0000339 0.25922`.

COCO100 результат на `0.00114` нижчий за c1 local native reference `0.182212`; це очікувано фіксується як рання перевірка нової 80→8-class голови, не Kaggle score і не stop-сигнал. Continuous YOLO course триває без зміни recipe/optimizer; status `training`. CSV ще немає (export кожні дві повні епохи), автоматичного подання немає; public-best c1 `0.2021478454` незмінний. Артефакти — у локальному run `runs/yolo11s_1280`; ваги, CSV, fingerprints і приватні шляхи в Git не публікувалися.

### YOLO11s epoch 2, export review, and course continuation — 2026-11-15

Другий повний epoch (index `1`) завершив усі 24,201 train frames за 27:39; checkpoint і common validation збережені. Independent official validation — 2,846 кадрів, ті самі 7 held-out families. COCO100 mAP `0.189767`, AP50 `0.380292`, AP75 `0.164265`, AP_small `0.119958`; COCO500 mAP `0.190573`, AP_small `0.120458`. Проти c1 initial full-val reference mAP `0.182212`, це `+0.007555`; AP_small gain `+0.047224`. Усе це — local AP, не public/Kaggle score.

COCO100 per-class AP: person `0.26828`, bicycle `0.18573`, car `0.40972`, van `0.20374`, truck `0.12679`, tricycle `0.20680`, bus `0.00093`, motor `0.11614`. Relative to c1, 6/8 classes improved; truck `−0.07690` and bus `−0.01805` regressed. Per-family AP: `uav0000086 0.18089`, `uav0000117 0.19124`, `uav0000137 0.30533`, `uav0000182 0.16723`, `uav0000268 0.14441`, `uav0000305 0.24245`, `uav0000339 0.23207`; compared with c1, 4/7 families improved. These are diagnostics, not a promotion decision.

The two-epoch export completed. Local structural validation reports `valid=true`, exactly 3,811 test IDs in template order, 1,374,267 detections, no empty predictions, and metadata bound to the selected checkpoint. Review status is `QUALITY_REVIEW_REQUIRED`; reference comparison and public score remain unset. The CSV/weights/metadata are local and were not committed or submitted. Course continues with the same recipe; epoch 3 is active. Watchdog reports monitor, power/sleep inhibition remains active, and c1 public-best `0.2021478454` is unchanged.

Застосовано `paired-family-class-v1` до epoch 2 проти c1 на тих самих 2,846 official-val records: pooled gain `+0.007555`, AP_small gain `+0.047224`, але candidate відхилений для promotion/export readiness через `class_regression`, `family_lower_bound_not_positive`, `too_few_improving_families`, `family_regression`, `gain_depends_on_one_family`. Family bootstrap 90% lower `−0.021572`, positive families `4/7`, leave-one-family-out minimum `−0.008267`. CSV лишається тільки `QUALITY_REVIEW_REQUIRED` попри валідну структуру; курс не зупинено й продовжується тим самим recipe.

### Наступний вимір до baseline: фіксований YOLO+c1 route — 2026-11-15

За прямим дорученням користувача працювати над перевищенням baseline додано
`scripts/class_ensemble.py` і [точний порядок виконання](baseline_next.md).
Гіпотеза одна: YOLO для person/bicycle/car/van/tricycle/motor, незмінний c1
для truck/bus. Дефіцити цих двох класів у записаній epoch 2 відповідають близько
0.01187 macro-AP, але фактичний ансамбль ще не виміряний; це не прогноз Kaggle.
Усі ваги зберігаються; нового навчання або зміни чинного YOLO recipe немає.

Локальному агенту: спочатку звір найсвіжішу повну YOLO val. Якщо новіший single
вже має готовий належно перевірений кандидат — повідом його одразу. Інакше на
найближчій збереженій межі тимчасово передай GPU двом frozen val-cache jobs:
canonical c1 із його 960 global+tiled pipeline та current validated YOLO best
1280 global. Використовуй нормальний STOP-boundary, не kill/freeze посеред епохи.
Після кешування продовж той самий YOLO course; CPU compare може працювати
паралельно, якщо є RAM. Не чекай завершення всіх 20 епох, щоб отримати перший
контрольований результат комбінації. Історію RT-DETR лишити без автоперезапуску.

Повна comparison вимірює обидва singles і route на всіх official-val records,
COCO100/500, class/family AP; готові prediction/metric caches повторно
використовуються. Strict promotion не послаблено. Окремий явний `--manual-review`
дозволяє лише підготовку CSV за pooled gain >=0.001 проти обох моделей,
class/size/evidence/minimum-family guards, із відкритими family warnings.
Class/size regression або недостатній gain не обходяться цим прапорцем.
Нічого не подається на Kaggle автоматично.

Лише після eligible full-val result витрачай GPU на два test-cache jobs із
тими самими frozen SHAs, потім CPU export/CSV validation та повідомлення
користувачу. Не підмінюй frozen YOLO source новішим live best під час test.
Додай фактичні повні метрики, timings, SHAs і CSV path/hash у відповідь сюди.
Перевірено 57 CPU tests суміжних компонентів, Ruff/compile/diff checks;
actual ensemble AP/GPU cost/public score тут не виміряні. Public-best
0.2021478454 і ціль >0.2216324580 залишаються незмінними.

### Frozen YOLO+c1 full-val comparison — 2026-11-15

За `baseline_next.md` закешовано всі 2,846 official-val кадрів із незмінного
c1 `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4` та
YOLO common-best `42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8`.
Використано c1 960 global+tiled pipeline і YOLO 1280 global; inference cache
зайняв відповідно 1,654.24 s і 115.17 s. Дані й checkpoints лишилися локально.

Повний local comparison COCO100/500: c1 `0.188016/0.188566`, YOLO
`0.189765/0.190570`, fixed route `0.202613/0.203428`; route AP_small `0.129118`
проти c1 `0.072257` і YOLO `0.119960`. Route бере truck/bus із c1, решту класів
із YOLO. Pooled gain становить `+0.014596` проти c1 і `+0.012848` проти YOLO.
Проти c1 покращились 6/7 families, але `uav0000305` впав на `0.065414`; paired
family 90% interval `[-0.015933, 0.032703]`. Проти YOLO route пройшов строгий
gate (6/7 positive families, lower `+0.003765`). Загальний статус
`FAMILY_UNCERTAINTY_MANUAL_REVIEW`, `accepted=false`, `review_eligible=true`;
це дозволяє лише локальний CSV для ручного розгляду, не promotion чи Kaggle
score. Test inference з frozen checkpoints ще не виконано.

YOLO course продовжено тим самим recipe після тимчасового GPU cache slot.
Під час resume loss meter став `nan` від batch 74, але EMA, optimizer state та
`last.pt` на збереженій межі були скінченні; точний checkpoint збережено окремим
SHA snapshot. Повторений batch відтворив це значення в накопичуваному epoch
meter. Поточний запуск доводиться до наступної full-validation межі, яка
визначить стан ваг. Для test CSV потрібен окремий normal STOP-boundary slot;
course після нього буде відновлено. Public c1 score `0.2021478454` не змінено;
жоден локальний AP не є Kaggle score.

### Frozen route CSV ready for manual Kaggle review — 2026-11-15

На точній frozen-парі з попереднього comparison виконано test cache для всіх
3,811 кадрів: c1 pipeline 960 global+tiled (`853.20 s` на test) і YOLO11s
1280 global (`100.71 s` на test). Ні новіший live-best, ні пошкоджений epoch-4
checkpoint не використовувалися. Ідентичності кешів і CSV metadata збігаються
з перевіреною парою.

Локальний CSV `yolo_c1_pair_epoch_004.csv` має SHA256
`1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562`.
Структурна перевірка: 3,811 IDs у template order, 1,485,721 detection,
0 порожніх кадрів, класи 0–7, не більше 500 записів/кадр, координати в межах
оригінальних зображень; `valid=true`, `all_predictions_empty=false`,
`matches_template_order=true`. Metadata має `manual_review=true`,
`strict_promotion_accepted=false`, `public_score=null`. У data package відсутній
окремий `sample_submission.csv`, тому використано канонічний порядок
`test_images.csv`, звірений з test manifest. CSV не подавався.

На всіх 2,846 незалежних official-val records (local AP, не Kaggle score):

| Pipeline | COCO100 mAP / AP50 / AP75 / APsmall | COCO500 mAP / APsmall |
|---|---:|---:|
| c1 tiled | 0.188016 / 0.383383 / 0.162947 / 0.072257 | 0.188566 / 0.072344 |
| YOLO global | 0.189765 / 0.380315 / 0.164253 / 0.119960 | 0.190570 / 0.120459 |
| class route | 0.202613 / 0.401829 / 0.180093 / 0.129118 | 0.203428 / 0.129623 |

Маршрут бере truck/bus із c1, інші шість класів із YOLO. Route pooled gain:
`+0.014596` проти c1 і `+0.012848` проти YOLO. COCO100 per-class AP для
route (person/bicycle/car/van/truck/tricycle/bus/motor):
`0.268288 / 0.185553 / 0.409673 / 0.203740 / 0.224090 / 0.206805 / 0.006704 /
0.116050`. Проти c1 сімейні gains: `+0.001035, +0.009522, +0.054217,
+0.005339, +0.054674, −0.065414, +0.006674`; 6/7 позитивні, bootstrap 90%
interval `[-0.015933, 0.032703]`. Проти YOLO 6/7 позитивні, lower bound
`+0.003765`. Тому `accepted=false`, `review_eligible=true` через сімейну
невизначеність; строгий promotion не дозволений. Manual-review CSV готовий до
перевірки користувачем. Baseline `0.2216324580` буде порівняно лише з реальною
відповіддю Kaggle.

Виправлено JSON serialization для накопичувального `tloss`: NaN/Inf стають
`null` із діагностичним прапорцем; повний receipt серіалізується до promotion
`common_best.pt`. Новий callback зупиняє course на нечисловому batch loss або
невалідному scaler та пише локальну діагностику; на train-start/epoch boundary
перевіряються model, EMA, optimizer і scaler. Targeted test пройшов (`1 passed`),
`compileall` і `git diff --check` чисті. Історично точний loss batch 74 і
кількість пропущених updates не були записані, тому ці значення невідомі.
Перевірений epoch-4 checkpoint має 9 нечислових EMA BatchNorm buffers і scaler
`0.0`, хоча model/optimizer tensors скінченні; відновлений epoch-3 `last.pt`
має скінченні model/EMA/optimizer і scaler `8192`. YOLO service лишається
зупиненим, STOP збережений, нові updates не виконуються.

### YOLO horizontal-flip A/B against the scored public-best route — 2026-11-15

Збережений контроль `yolo_c1_pair_epoch_004.csv` і його SHA
`1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562` не
змінювалися. Frozen c1/YOLO inputs ті самі canonical SHAs із попереднього
запису. Новий `HorizontalFlipOnlyPredictor` рахує тільки flip view; x-boxes
віддзеркалюються у 1280 canvas перед geometry restore. Увесь prediction merge
після inference працює на CPU: class-aware NMS IoU 0.6 і global cap 500.
Native cache identity лишилась біт-в-біт сумісною; новий cache окремо позначає
adapter, його SHA та flip transform. І c1 inference не повторювався.

Flip-only YOLO inference: official val 2,846 кадрів за `199.54 s`
(`14.26 frames/s`); test 3,811 за `134.32 s` (`28.37 frames/s`). A/B з
точним parent comparison SHA `e66e9a6b6d7604dd3ae7ddaf799a2b0391fb22ccf92182303803e62eb4aff60f`
і тим самим validation fingerprint завершив CPU metrics за `355.10 s`.

Усі метрики — local COCO, не Kaggle score:

| Route | COCO100 mAP / AP50 / AP75 / APsmall | COCO500 mAP / APsmall |
|---|---:|---:|
| Scored parent route | 0.202613 / 0.401829 / 0.180093 / 0.129118 | 0.203428 / 0.129623 |
| YOLO native + flip route | 0.216622 / 0.425163 / 0.192607 / 0.135170 | 0.217520 / 0.135717 |

Gain проти public-best route: COCO100 `+0.01400946`, COCO500 `+0.01409233`,
APsmall `+0.00605210`. COCO100 per-class gains
(person/bicycle/car/van/truck/tricycle/bus/motor):
`+0.005644 / +0.025316 / +0.005922 / +0.053631 / 0 / +0.011949 / 0 / +0.009614`;
усі class/size guards пройдені. Усі 7 held-out families покращились; їхні gains
`+0.006448, +0.009223, +0.013677, +0.008264, +0.021656, +0.008318, +0.011516`;
paired family-bootstrap 90% interval `+0.008621…+0.014662`. Статус local gate
`accepted=true`; це не обіцяє Kaggle результат, а official val уже використовувався
для відбору.

Новий локальний CSV `yolo_c1_pair_epoch_004_fliptta.csv`, SHA256
`2dee65d0f257ba61569d48ba94ad95e74e740802b7e430cf10556a1fe2ebdce0`.
Незалежний submission checker підтвердив 3,811 IDs у template order, 1,624,942
детекції, нуль порожніх кадрів, класи 0–7, координати в межах кадру, cap 500,
`valid=true`, `matches_template_order=true`; metadata SHA збігається та зв’язує
точні val/test cache identities. `manual_review=true`, `strict_promotion_accepted=true`,
`public_score=null`. CSV не подавався автоматично. Поточний public-best лишається
підтверджений parent `0.2176009680`; новий Kaggle score невідомий. Пошкоджений
epoch-4 checkpoint не використовувався, YOLO STOP лишився на місці, навчання не
відновлювалось.

Цільові flip tests: `3 passed` (odd dimensions/asymmetric padding, tensor-space
unflip before restore, same-class duplicate suppression, class preservation,
500 cap і окрема cache identity). Ruff, `compileall` та `git diff --check` пройшли.

## Стабільне FP32-продовження champion і перевірки кожні дві епохи — 2026-11-15

**Статус: TRAINING.** Public-best `0.2237335225` збережений: контрольний CSV
`yolo_c1_pair_epoch_004_fliptta.csv` має SHA256
`2dee65d0f257ba61569d48ba94ad95e74e740802b7e430cf10556a1fe2ebdce0`; frozen
c1 SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4`,
YOLO SHA `42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8`.
Пошкоджений epoch-4 лишається виключеним; новий run почався з frozen YOLO
checkpoint, FP32, batch 1, AdamW LR `1e-4`, frozen BatchNorm, `imgsz=1280`,
повний train manifest `24,201` кадрів; `2,846` official-val кадрів незалежні.
Для нового рецепта optimizer навмисно свіжий; це не resume старих optimizer
moments.

Перевірена початкова full-val оцінка source checkpoint на тому самому cohort:
COCO100 mAP `0.1897116407`, AP50 `0.3803219992`, AP75 `0.1640890306`,
APsmall `0.1198247004`; COCO500 mAP `0.1905167524`, APsmall `0.1203154464`.
Це початкова native-оцінка моделі, не новий ансамблевий кандидат і не Kaggle
score. Навчання активне; повного epoch receipt ще немає. Shared GPU lease і
sleep-inhibition активні. Дані змонтовані read-only; вихідні checkpoints пишуться
у work-каталог. Під час першої епохи зафіксовано `1,550/24,201` batch при
поточному темпі `7.7 batch/s`; рання оцінка часу до межі — близько 49 хвилин,
не гарантований ETA. VRAM було `2,819/8,188 MiB`, GPU util `74%`, host RAM
available `7.2 GiB`, swap usage близько `0.3 MiB`, вільно на output filesystem
`4.5 GiB`. Контролер `yolo_tta_controller.py` очікує дві повні епохи,
зупиняє тільки на збереженій межі, рахує native+horizontal-flip на повній val,
порівнює з точним champion report/CSV, повторно використовує c1 cache та
експортує лише кандидата, що пройшов class/size/family guards. Потім продовжує
той самий 20-epoch course через FP32 recovery state. Test inference і Kaggle
submission не запускаються для кандидата до проходження val gate; auto-submit
вимкнений.

Код generic champion comparison, перевірки parent/cohort/view identities і
run-specific controller локально перевірено: `6 passed`, Ruff, compile та
`git diff --check` пройшли. Додатковий self-control на тих самих frozen
champion caches точно відтворив class route (Δ COCO100 mAP `0.000000`); all-YOLO
route мав Δ `-0.0105588778`. Обидва закономірно відхилені; це не новий кандидат,
CSV не створювався. CPU comparison тривав `706.62 s`; GPU inference не запускався.
Перший навчений checkpoint A/B ще очікує двох повних епох; метрики/CSV буде
записано окремо. Жодного нового public score цим записом не заявлено.

Оновлення стану о 2026-11-15 19:48 EET: продовження досі на epoch 1,
`15,017/24,201` batches, `10.2 batch/s`; GPU `93%`, VRAM `3,299/8,188 MiB`,
RAM available `7.2 GiB`, swap free `4,095 MiB`, work filesystem вільно `4.46 GiB`.
CUDA активна, numeric-failure receipt немає. Контролер перезавантажено окремо,
щоб активував відновлення вже збережених comparison/CSV за їхніми fingerprints
і до трьох повторів невдалої перевірки; навчальний процес не переривався.
Цільові champion/flip тести повторно пройшли: `6 passed`. Двоепоховий кандидат
ще не готовий; останній вимір — проміжний статус, не результат якості.
