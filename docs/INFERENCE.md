# Фінальний ансамбль: RT-DETR c1 + YOLO11s + flip

## Зафіксовані артефакти

| Артефакт | SHA256 |
| --- | --- |
| RT-DETR c1 | `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4` |
| YOLO11s | `42e4c916c45a262d1e29863dc7edb919679528fa673241eef4641a25606285a8` |
| Parent CSV без flip | `1d4ac45ca8f1b121d40c65ddea376a498b845ab1249cd0f00063eb2f8ddf2562` |
| Фінальний CSV | `2dee65d0f257ba61569d48ba94ad95e74e740802b7e430cf10556a1fe2ebdce0` |

Фінальний файл — `yolo_c1_pair_epoch_004_fliptta.csv`, public score **0.2237335225**. Parent CSV — `yolo_c1_pair_epoch_004.csv`, score **0.2176009680**. Повні записи містяться в [журналі](../experiments/leaderboard_results.json).

Ваги, точні runtime-конфігурації та результати перевірок зберігаються локально й не включені в Git. Для відтворення саме цієї пари потрібні ці артефакти; заміна їх новішим live-best змінює експеримент.

## Послідовність

1. **RT-DETR c1:** input 960; повний кадр і плитки 960, overlap 0.2, NMS IoU 0.6, edge margin 2. Координати плиток повертаються у вихідний кадр.
2. **YOLO11s:** input 1280; повний кадр в оригінальному та горизонтально віддзеркаленому вигляді. Повернення flipped-боксів виконується у координатах квадратного canvas до скасування letterbox.
3. Передбачення двох YOLO views об’єднуються class-aware NMS з IoU 0.6.
4. `truck` і `bus` беруться з c1; `person`, `bicycle`, `car`, `van`, `tricycle`, `motor` — із YOLO. Координати й ваги моделей не усереднюються.
5. Об’єднаний список сортується за score; залишається до 500 записів на кадр.
6. CSV пишеться в порядку тестового шаблону. Перевіряються IDs, класи, координати й хеш файла.

## Налаштування шляхів

Визначте `WORK`, `C1_CONFIG`, `YOLO_CONFIG`, `C1_WEIGHTS`, `YOLO_WEIGHTS`, `PARENT_REPORT`, `PARENT_CSV`. Конфігурації мають відповідати зафіксованим вагам та містити шляхи до тих самих manifests. Parent report — повне порівняння одиночних моделей і class route до додавання flip. Parent CSV і його `.metadata.json` потрібні для перевірки походження.

Створіть новий каталог для відтворення, зберігаючи попередні артефакти:

```bash
export PAIR="$WORK/ensemble/reproduction_flip"
mkdir -p "$PAIR"
```

## Validation-кеші та порівняння

```bash
.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$C1_CONFIG" cache \
  --backend c1 --checkpoint "$C1_WEIGHTS" --split val \
  --output "$PAIR/c1_val"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" cache \
  --backend yolo --checkpoint "$YOLO_WEIGHTS" --split val \
  --output "$PAIR/yolo_val"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" cache \
  --backend yolo --checkpoint "$YOLO_WEIGHTS" --split val \
  --view horizontal_flip --output "$PAIR/yolo_flip_val"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" compare-flip \
  --c1 "$PAIR/c1_val" --yolo "$PAIR/yolo_val" \
  --yolo-flip "$PAIR/yolo_flip_val" \
  --parent-report "$PARENT_REPORT" --parent-csv "$PARENT_CSV" \
  --output "$PAIR/comparison_flip.json"
```

Команда звіряє точні джерела, усі 2 846 validation records, параметри views та результати parent comparison. Зміна ваг, cohort або pipeline потребує нового експерименту. Метрики включають COCO100/500, AP за класами, AP_small та результати за відеосімействами.

## Test і експорт

Після успішної повної перевірки створіть test-кеші з тими самими вагами:

```bash
.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$C1_CONFIG" cache \
  --backend c1 --checkpoint "$C1_WEIGHTS" --split test \
  --output "$PAIR/c1_test"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" cache \
  --backend yolo --checkpoint "$YOLO_WEIGHTS" --split test \
  --output "$PAIR/yolo_test"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" cache \
  --backend yolo --checkpoint "$YOLO_WEIGHTS" --split test \
  --view horizontal_flip --output "$PAIR/yolo_flip_test"

.venv/bin/python scripts/class_ensemble.py \
  --work "$WORK" --config "$YOLO_CONFIG" export-flip \
  --c1-val "$PAIR/c1_val" --yolo-val "$PAIR/yolo_val" \
  --yolo-flip-val "$PAIR/yolo_flip_val" \
  --c1-test "$PAIR/c1_test" --yolo-test "$PAIR/yolo_test" \
  --yolo-flip-test "$PAIR/yolo_flip_test" \
  --parent-report "$PARENT_REPORT" --report "$PAIR/comparison_flip.json" \
  --output "$WORK/submissions/reproduction_flip.csv" --manual-review
```

`--manual-review` записує явний ручний експорт; він не обходить вимоги якості та не відправляє CSV на Kaggle. Повторний inference може відрізнятися чисельно між GPU/runtime; звірте конфігурації, метрики та metadata, а не лише назву вихідного файла.
