# Dataset & Model Audit

Результаты аудита датасета и модели. Обновляется после каждого цикла улучшений.

## Как запустить

```bash
make audit            # Провести аудит, сохранить snapshot
make audit-compare    # Сравнить последний snapshot с предыдущим

# Сравнить конкретные snapshots:
make audit-compare OLD=audit_20261013_224053.json NEW=audit_20261014_120000.json
```

Snapshots сохраняются в `ml/results/audits/` (gitignored). `latest.json` → последний.

## Текущее состояние — после итерации 1 (2026-10-13)

### Val accuracy

| Класс | Windows | Accuracy | Δ vs baseline |
|---|---|---|---|
| **siren** | 3579 | **79.8%** | **+3.4%** ✓ |
| background | 8700 | 81.4% | -8.1% (ожидаемо) |
| explosion | 1395 | 71.3% | -7.5% |
| **UAV** | 658 | **65.8%** | **+8.4%** ✓ |

### Файлы: aug/real соотношение

| Класс | Всего | Real | Aug | Aug% | Ratio | Статус |
|---|---|---|---|---|---|---|
| siren | 249 | 83 | 166 | 66.7% | 2.0× | OK |
| background | 674 | 674 | 0 | 0% | 0× | OK |
| explosion | 610 | 258 | 352 | 57.7% | 1.4× | OK |
| UAV | 240 | 80 | 160 | 66.7% | 2.0× | OK |

### Confusion matrix

```
                  siren  background  explosion  UAV
  siren           2856        353        128        242
  background       910       7085        507        198
  explosion        100        273        994         28
  UAV       132         24         69        433
```

### Состояние train_classifier.py

- ✅ Class weights: `sqrt(N_total / N_class)`
- ✅ Sample weights: real=1.0, aug/external=0.4
- ✅ Val split по source (с учётом aug — data leakage = 0)
- ✅ Dead window filtering (UDP drops)
- ✅ Background undersampling (MAX_BG_TRAIN_WINDOWS=2000)
- ✅ Embedding cache
- ❌ Mixup на embeddings — ещё не реализовано

---

<details>
<summary>Baseline — до оптимизации (2026-10-13)</summary>

### Val accuracy (baseline)

| Класс | Windows | Accuracy |
|---|---|---|
| siren | 2691 | 76.4% |
| background | 7272 | 89.5% |
| explosion | 312 | 78.8% |
| UAV | 176 | **57.4%** |

### Файлы (baseline)

| Класс | Всего | Real | Aug | Aug% | Ratio | Статус |
|---|---|---|---|---|---|---|
| siren | 875 | 83 | 792 | 90.5% | 9.5× | ⚠ критично |
| background | 674 | 674 | 0 | 0% | 0× | OK |
| explosion | 610 | 258 | 352 | 57.7% | 1.4× | OK |
| UAV | 1016 | 80 | 936 | 92.1% | 11.7× | ⚠ критично |

### Confusion matrix (baseline)

```
                  siren  background  explosion  UAV
  siren           2055        360        113        163
  background       142       6505        426        199
  explosion          6         43        246         17
  UAV         5         55         15        101
```

### Проблемы (baseline)

- Aug/real ratio: siren 9.5×, UAV 11.7× (рекомендация ≤ 2×)
- Class imbalance: background 41.3× vs UAV
- Data leakage: 295 aug файлов от val sources в train
- Нет sample weights, нет background undersampling

</details>

## Итерация 1 — 2026-10-13

### Что сделано

1. ✅ **Data leakage fix** — `source_name()` теперь strip'ит `aug_`/`aug_q_` prefix и `_XXXXXXXX` hash suffix → aug файлы попадают в тот же split что оригинал
2. ✅ **Урезаны aug**: siren 792→166 (2×), UAV 936→160 (2×)
3. ✅ **Sample weights**: real=1.0, aug/external=0.4, умножаются на class weights
4. ✅ **Class weights**: формула `sqrt(N_total / N_class)` (TRAINING_GUIDE)
5. ✅ **Undersample background**: 36748→2000 windows в train
6. ✅ **Пересоздан val split** (удалён старый `val_sources.json`)

### Результаты vs baseline

| Класс | Baseline | После | Дельта |
|---|---|---|---|
| **siren** | 76.4% | **79.8%** | **+3.4%** ✓ |
| **UAV** | 57.4% | **65.8%** | **+8.4%** ✓ |
| background | 89.5% | 81.4% | -8.1% (ожидаемо) |
| explosion | 78.8% | 71.3% | -7.5% |

Background accuracy снизилась потому что модель перестала «лениться» предсказывать
доминирующий класс. siren и UAV — главные цели — значительно выросли.

Data leakage: **0** (было 295 файлов в 3 классах).

### Текущее состояние train_classifier.py

- ✅ Class weights: `sqrt(N_total / N_class)`
- ✅ Sample weights: real=1.0, aug/external=0.4
- ✅ Val split по source (с учётом aug)
- ✅ Dead window filtering (UDP drops)
- ✅ Background undersampling (MAX_BG_TRAIN_WINDOWS=2000)
- ✅ Embedding cache
- ❌ Mixup на embeddings — ещё не реализовано

## Следующие шаги

1. **Набрать реальных данных** — UAV и explosion нуждаются в real recordings
2. **Mixup на embeddings** — смешивание двух embeddings, λ ~ Beta(0.2, 0.2)
3. **Noise profile** — записать 2-3 мин тишины с INMP441 для `--noise-profile`
4. **Задеплоить** — `make deploy-model` чтобы проверить на живом потоке
