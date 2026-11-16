# Перевірка коду та результатів

## Тести

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check cvlab scripts tests
```

Тести охоплюють перетворення координат, ignored regions, COCO-метрики, групування послідовностей, відновлення стану, поєднання кешів та перевірку CSV. Проходження тестів перевіряє реалізацію, а не результат навченої моделі.

## Оцінювання моделі

```bash
.venv/bin/python -m cvlab evaluate \
  --config /path/to/config.yaml \
  --checkpoint /path/to/best.pt
```

Використовуйте ту саму конфігурацію даних, роздільність, правила інференсу та ліміт детекцій для варіантів, які порівнюєте. Основні метрики: COCO AP при IoU 0.50–0.95, AP50, AP_small та AP за класами. Варіанти з maxDets 100 і 500 записуються окремо.

Для повторного використання передбачень перевіряються SHA256 ваг, список кадрів і параметри інференсу. Validation-кадри не додаються до навчання кандидатів, які на них порівнюються. Кадри однієї відеопослідовності не розбиваються випадково між train і val.

## Submission CSV

```bash
.venv/bin/python scripts/check_lab.py \
  --work "$WORK" --data-root "$DATA_ROOT" \
  --submission /path/to/submission.csv
```

Окрім відсутності помилок, звірте `valid=true`, `matches_template_order=true`, `all_predictions_empty=false`. Порожній окремий кадр допускається. Формат одного передбачення: `class score x y width height`; класи 0–7, координати у вихідних пікселях, максимум 500 детекцій/кадр.

Значення public score береться з відповіді Kaggle. У [журналі](experiments/leaderboard_results.json) є точні назви оцінених CSV та відомі хеші. Короткий probe, зниження train loss або наявність checkpoint-файла не замінюють повну перевірку якості.
