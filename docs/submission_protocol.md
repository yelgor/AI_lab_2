# Коли можна подавати результати

Це операційна інструкція для локального Codex і користувача. Оригінальна умова:
[assignment/kaggle-original.txt](assignment/kaggle-original.txt). Документ не є
фінальним студентським звітом.

## Проміжний Kaggle CSV можна подати до кінця всього навчання

Не потрібно чекати завершення всіх моделей, циклів або восьмигодинного плану.
Можна подати окремий заморожений checkpoint після його повної валідації й
закінченого перевіреного test export, поки інші експерименти продовжуються.
Неповна епоха допускається, якщо її реальний обсяг задокументовано та валідація
цього checkpoint завершена. Саме зниження train loss не є підставою для подання.

## Gate 1 — якість і відсутність leakage

Для поточного `train-efficient`:

1. Швидкий probe — до 512 фіксованих held-out кадрів. Локальний контролер
   вимагає щонайменше чотири незалежні video families, додатну нижню межу
   приблизного family-bootstrap інтервалу та global probe AP gain >=0.001.
   Це фільтр витрати compute, а не гарантія Kaggle score.
2. Кандидат проходить **повну native validation оригінального held-out fold**
   з AP gain >=0.001 проти поточного локального лідера.
3. Далі він проходить **повну official validation із фіксованим tiled inference**
   з AP gain >=0.001 проти того самого лідера. Налаштування: input 960,
   tile 960, overlap 0.2, NMS 0.6, edge margin 2.
4. У `WORK/efficient_search/state.json`, у `cycles[].confirmations[]`, має бути
   фактичний запис `accepted: true` з `native_gain`, `tiled_gain`, `run`, повним
   `checkpoint_sha256` та `submission` для потрібного файлу. Експорт повинен
   закінчитися, а не лише початися.

Для нового протоколу потрібна явно задокументована еквівалентна перевірка.
Не переносити AP з іншого cohort/fingerprint, іншої кількості кадрів, cap чи
inference settings як прямий приріст. Не оцінювати all-training fine-tune на
fold, кадри якого вже були використані в його навчанні. Після promotion
контролер перебазовує probe evidence відносно нового лідера: поточний нульовий
gain самого лідера не скасовує його попередньої успішної перевірки.

`last.pt` може вже існувати при `validation_pending`; це ще не завершена оцінка.
Окремий рядок `READY` від старого exporter не доводить проходження всіх gates.
Сильний probe із невдалою full validation зберігається як експеримент, але не
витрачає submission quota в поточному режимі відбору.

## Gate 2 — закінчений CSV та походження

Обов'язково перевірити:

- Фінальний `.csv`, не `.tmp`; поруч є `.csv.metadata.json`.
- SHA256 frozen checkpoint відповідає confirmation record і metadata; SHA256
  CSV відповідає `metadata.validation.sha256`.
- Точні колонки `image_id,PredictionString`, усі test IDs рівно по одному,
  правильний порядок sample/test manifest. Очікувана кількість береться з даних
  (у відомому наборі — 3811), не підставляється навмання.
- На детекцію: `class score x y width height`; class 0–7, score у [0,1], скінченні
  числа, додатні width/height; координати у вихідних пікселях і в межах кадру.
  Максимум 500 детекцій на зображення після об'єднання всіх views.
- Порожні окремі кадри допустимі. Повністю порожній submission не подається.
- Не подавати повторно той самий SHA256 CSV під іншою назвою; не експортувати
  знову вже відхилений checkpoint/inference variant як нібито нове поліпшення.

Загальна CPU-перевірка (шляхи задає локальний агент за реальною конфігурацією):

```bash
python3 scripts/check_lab.py \
  --work "$CVLAB_LOCAL_WORK" \
  --data-root "$DATA_ROOT" \
  --submission "$CSV"
```

Окрім exit code перевірити `valid=true`, `all_predictions_empty=false` та
`matches_template_order=true`. Чинний checker повертає порядок і порожність
окремими полями; один успішний exit code їх не гарантує.

Строгий gate для CSV саме з поточного efficient-search (тільки CPU, без
unpickle або запуску моделі):

```bash
python3 - "$CVLAB_LOCAL_WORK" "$CSV" <<'PY'
import hashlib, json, sys
from pathlib import Path
from scripts.check_lab import validate_submission

work, csv_path = map(lambda p: Path(p).resolve(), sys.argv[1:])
state = json.loads((work / "efficient_search/state.json").read_text())
entries = [c for cycle in state["cycles"] for c in cycle["confirmations"]
           if c.get("accepted") is True and c.get("submission")
           and Path(c["submission"]).resolve() == csv_path]
assert entries, "No accepted full-validation confirmation for this CSV"
entry = entries[-1]
assert entry["native_gain"] >= 0.001 and entry["tiled_gain"] >= 0.001
run = Path(entry["run"])
config = json.loads((run / "config.json").read_text())
root = Path(config["data"]["root"])
assert csv_path.suffix == ".csv"
result = validate_submission(csv_path, root / "test_images.csv",
                             root / "sample_submission.csv")
assert result["valid"] and result["matches_template_order"]
assert not result["all_predictions_empty"]
meta = json.loads(csv_path.with_suffix(".csv.metadata.json").read_text())
with (run / "best.pt").open("rb") as stream:
    checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert checkpoint_sha == entry["checkpoint_sha256"] == meta["checkpoint_sha256"]
assert result["sha256"] == meta["validation"]["sha256"]
print("READY_FOR_KAGGLE", csv_path.name)
print(json.dumps(result, indent=2))
PY
```

Не вважати це підтвердженням повного OOF, academic report або private score.

## Gate 3 — подання та фактичний score

Перевірити, що змагання зараз приймає submissions і в акаунті лишилася квота.
Поточні deadline/quota беруться з актуального Kaggle/оголошення, не зі старого
знімка або вигаданого «5 на день». Подання можливе одразу після gates 1–2.

Точне перевищення baseline **не можна вимагати від невідомого public score до
першого подання кандидата**. Не порівнювати local AP з public baseline як одну
метрику на тих самих даних. Ціль public score — строго **>0.2216324580**.
`READY_FOR_KAGGLE` означає готовність файлу; `SUBMITTED` — фактичну відправку;
`SCORED` — отриману оцінку Kaggle. Наявність CSV не означає жодного з останніх
двох станів.

Поточний watchdog сам не викликає Kaggle submit. Агент повідомляє готовий шлях
і короткі вимірювання; спосіб подання застосовується в межах явно дорученого
користувачем режиму. Після відповіді Kaggle записати:

- filename, CSV SHA256, checkpoint SHA256 та inference settings;
- full-validation cohort/fingerprint і метрики, дату фактичного подання;
- справжній public score та доказ (сторінка submissions, відповідь API/CLI або
  наданий користувачем screenshot).

Перевірені public спостереження додавати до
`experiments/leaderboard_results.json`, коротку відповідь — у
`docs/AGENT_COORDINATION.md`. До отримання score залишати його невідомим.
Не оголошувати baseline пройденим за локальними метриками або прогнозом.

За збереженими доказами на дату цієї інструкції (2026-11-11), public-best —
**0.1928780288**, source SHA
починається на `a7cd89fb349d`; baseline ще не перевищено. Новіші оцінки перевіряти
в актуальному `experiments/leaderboard_results.json`. Якщо новий public score
гірший, зберегти невдалий експеримент, не перезаписувати перевірений public-best
і не продовжувати довге донавчання від погіршеного checkpoint автоматично.
Корекцію/відкат виконувати через сумісний механізм контролера на збереженій межі,
зі збереженням історії й виключенням повторних подань вже відхиленого варіанта.

## Фінальна здача лабораторної — окремий етап

Готовий проміжний CSV не замінює виконання рубрики. До фінальної здачі потрібні
чотири сім'ї детекторів, обґрунтовані tuning/loss experiments, CV/OOF, метрики та
failure cases, відтворюваний final pipeline, власноручний звіт у викладацькому
template/GitHub Pages та repository link у LMS. Перевірити деталі за оригіналом.

Два фінальні рішення вибирати за внутрішньою оцінкою/узагальненням, а не лише
максимумом public leaderboard. Public baseline gate важливий, але private
leaderboard і повнота роботи також впливають на оцінку. Актуальний продовжений
deadline має бути встановлено окремо; не покладатися на старе «a day to go»,
не пушити за запланованим розкладом після cutoff і не backdate commits.
