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
