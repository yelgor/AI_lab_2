undefined

### Результат c1 batch-2 retry — 2026-11-14, 16:18 EET

Завершено один повний selection epoch на всіх 24,201 train-кадрах (batch=2, accumulation=2); training loss `12.6019`, measured pipeline `4.96` кадр/с, потім повна native validation на 2,846 official-val кадрах дала mAP `0.1917016` проти source baseline `0.1822042` (pooled gain `+0.0094974`). Це локальна native-метрика, не Kaggle score.

Selection відхилив epoch: `class_regression`, `family_lower_bound_not_positive`, `family_regression`. Зокрема bicycle −0.01883, car −0.00521, person −0.00597; сімейний bootstrap lower bound −0.00630, 5/7 сімейств позитивні. Plan `c1_batch2_retry1` перейшов у `selection_halted`; другий epoch, tiled confirmation і final train+val не запускались, CSV не створено. Служба вимкнена й inactive; watchdog timer active; GPU вільний. Source SHA `c1d7fcc83dde045578514e64c85e04b215fca988c804985f1a5a313ebe6daab4` збережений. Run `WORK/runs/rtdetr_c1_batch2_retry1_train_selection/`: `best.pt` source-best SHA256 `cbe10f0089d6475eb2eef0879a61ad4d2b8f80f7bf48946e73f5ea0e8823606c`; `last.pt` epoch candidate SHA256 `7feda78d6de7197b5598c996d980188b591b46001980d90bba77847f21973ec3`.

Рішення: не подавати epoch candidate і не повторювати цей самий recipe. Наступний крок — аналіз class/family error evidence та заздалегідь визначена змінена гіпотеза з тими самими robust gates. Поточний Kaggle public-best залишається `0.2021478454`; baseline `0.2216324580` і ціль `0.28+` не досягнуті.