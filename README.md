# Детекція об’єктів на відео з дрона

Лабораторна робота №2, Artificial Intelligence, UCU, 2026.

**Автори:** Гордій Єлісєєв та Володимир Легчило Янішевський.

[Звіт на GitHub Pages](https://yelgor.github.io/AI_lab2/) · [Преобробка](docs/PREPROCESSING.md) · [Фінальний ансамбль та експорт](docs/INFERENCE.md) · [Оцінки Kaggle](experiments/leaderboard_results.json)

У репозиторії є підготовка даних VisDrone2019-VID, навчання Faster R-CNN, RetinaNet, FCOS, RT-DETR і YOLO11s, оцінювання та формування submission CSV. Найкращий зафіксований public score — **0.2237335225**, competition baseline — **0.2216324580**.

Фінальний варіант поєднує RT-DETR c1 для `truck` і `bus` та YOLO11s для інших шести класів. YOLO використовує оригінальне й горизонтально віддзеркалене зображення. Кожен клас надходить від однієї моделі; координати чи ваги різних моделей не усереднюються.

## Вміст

| Каталог | Призначення |
| --- | --- |
| `cvlab/` | Дані, моделі, навчання, оцінювання, інференс |
| `configs/` | Конфігурації чотирьох базових детекторів та експериментів |
| `scripts/` | Запуски навчання, кешування передбачень, ансамбль, перевірки CSV |
| `tests/` | Тести геометрії, метрик, відновлення навчання та експорту |
| `experiments/` | Зафіксовані результати експериментів |
| `docs/assignment/` | Збережена умова змагання |

Ця гілка підготовлена для перегляду та відтворення. Код узято з `experiment/c1-lrhalf-continuous`, commit `21f49d1ba14c34924a49c54e133e6cac069364ac`. Навчальні модулі, конфігурації, scripts і тести залишені без змін. Оперативні нотатки зберігаються в історичній робочій гілці.

## Встановлення

Потрібні Python 3.11–3.13 і сумісний NVIDIA driver для GPU-запуску. Основні версії зафіксовані в `pyproject.toml`: PyTorch 2.6.0, TorchVision 0.21.0, Transformers 4.51.3, Ultralytics 8.3.250, pycocotools 2.0.11.

```bash
git clone --branch submission/lab2 https://github.com/yelgor/AI_lab_2.git
cd AI_lab_2
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,yolo]'
.venv/bin/python -m cvlab doctor
```

Перевірте відповідність встановленого PyTorch драйверу за [офіційною інструкцією](https://pytorch.org/get-started/previous-versions/).

## Дані

Завантажте дані [змагання](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026) у своєму акаунті. Очікувана структура:

```text
DATA_ROOT/
  train/{images,labels,ignore}/<clip>/...
  val/{images,labels,ignore}/<clip>/...
  test/images/<clip>/...
  test_images.csv
```

`DATA_ROOT` — шлях до розпакованих даних; `WORK` — окремий каталог для вихідних файлів. Підготовка не змінює початкові зображення та розмітку:

```bash
export DATA_ROOT=/path/to/dataset
export WORK=/path/to/work
mkdir -p "$WORK"
.venv/bin/python -m cvlab prepare \
  --root "$DATA_ROOT" --out "$WORK/prepared" \
  --folds 3 --seed 42 --workers 2
```

Команда створює train/val/test manifests, опис fold-розбиття та EDA-файли. Підготовка fold-файла сама по собі не виконує навчання всіх folds.

## Навчання

Для базового детектора:

```bash
.venv/bin/python -u -m cvlab train \
  --config configs/retinanet.yaml \
  --set "data.root=$DATA_ROOT" \
  --set "data.prepared=$WORK/prepared" \
  --set data.split=official \
  --set "output=$WORK/runs/retinanet"
```

Для іншого сімейства виберіть відповідний YAML у `configs/` та новий `output`. Продовження з відновленням стану використовує `--resume /path/to/last.pt`; новий експеримент з готових ваг — `--init-checkpoint /path/to/best.pt`.

Для YOLO потрібен `SOURCE_CONFIG`: JSON/YAML із коректними `data.root` та `data.prepared`, наприклад локальна копія `configs/rtdetr.yaml` із виправленими шляхами. YOLO launcher використовує ці шляхи для власної підготовки даних:

```bash
export SOURCE_CONFIG=/path/to/config.yaml
.venv/bin/python scripts/start_yolo_training.py \
  --work "$WORK" --config "$SOURCE_CONFIG" --prepare-only
.venv/bin/python -u scripts/start_yolo_training.py \
  --work "$WORK" --config "$SOURCE_CONFIG" \
  --name yolo11s_1280 --epochs 20 --hours 10 \
  --batch 2 --imgsz 1280 --workers 2
```

Це команда навчального рецепта, а не гарантія повторення конкретних ваг або Kaggle score. Параметр `--epochs 20` задає верхню межу курсу. Фактичні результати потрібно читати зі збережених повних перевірок.

## Фінальні передбачення

Для точного складу обраного ансамблю потрібні заморожені ваги c1 та YOLO, їхні конфігурації й файли результатів валідації. Вони зберігаються окремо від Git. SHA256, налаштування плиток, TTA та команди експорту наведено в [docs/INFERENCE.md](docs/INFERENCE.md).

Оцінки на validation та Kaggle зберігаються окремо. CSV перевіряється на повноту 3 811 test IDs, порядок, класи, координати, скінченність чисел і ліміт 500 детекцій на кадр.

## Перевірки

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check cvlab scripts tests
```

Детальніше: [VALIDATION.md](VALIDATION.md).

## Використані реалізації та джерела

- [TorchVision](https://pytorch.org/vision/0.21/models.html#object-detection): Faster R-CNN, RetinaNet і FCOS.
- [Transformers RT-DETR](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/rt_detr), pretrained `PekingU/rtdetr_r50vd`.
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), pretrained `yolo11s.pt`.
- [VisDrone](https://arxiv.org/abs/2001.06303), [COCO evaluator](https://github.com/cocodataset/cocoapi), [SAHI: slicing inference](https://arxiv.org/abs/2202.06934).

Код інтеграції, контролю даних, спільних метрик, кешування та маршрутизації міститься в цьому репозиторії. Ліцензії бібліотек, pretrained weights і даних застосовуються окремо; базові архітектури не заявляються власними розробками.
