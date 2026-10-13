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

## Baseline — 2026-10-13

Первый аудит перед началом оптимизации.

### Файлы: aug/real соотношение

| Класс | Всего | Real | Aug | Aug% | Ratio | Max рек. | Статус |
|---|---|---|---|---|---|---|---|
| siren | 875 | 83 | 792 | 90.5% | 9.5× | 166 | ⚠ критично |
| background | 674 | 674 | 0 | 0% | 0× | — | OK |
| explosion | 610 | 258 | 352 | 57.7% | 1.4× | 516 | OK |
| motorcycle | 1016 | 80 | 936 | 92.1% | 11.7× | 160 | ⚠ критично |

**Правило из TRAINING_GUIDE:** aug ≤ 2× real. siren и motorcycle далеко за пределами — модель учит распределение ESC-50/Freesound, а не INMP441.

### Val accuracy

| Класс | Windows | Accuracy |
|---|---|---|
| siren | 2691 | 76.4% |
| background | 7272 | 89.5% |
| explosion | 312 | 78.8% |
| motorcycle | 176 | **57.4%** |

### Class imbalance (val windows)

| Класс | Windows | Ratio к min |
|---|---|---|
| background | 7272 | **41.3×** |
| siren | 2691 | 15.3× |
| explosion | 312 | 1.8× |
| motorcycle | 176 | 1.0× (min) |

Background доминирует в 41× — TRAINING_GUIDE рекомендует 2-3× от среднего.

### Confusion matrix

```
                  siren  background  explosion  motorcycle
  siren           2055        360        113        163
  background       142       6505        426        199
  explosion          6         43        246         17
  motorcycle         5         55         15        101
```

Главные проблемы:
- **siren → background** (360, 13%) — модель пропускает сирену
- **background → explosion** (426, 5.9%) — ложные explosion
- **background → motorcycle** (199, 2.7%) — ложные motorcycle

### Data leakage ⚠

Aug файлы от val sources попадают в train из-за разных source names (`aug_` prefix):

| Класс | Leaked files | Из val sources |
|---|---|---|
| siren | 175 | 13 / 18 |
| explosion | 25 | 11 / 17 |
| motorcycle | **95** | **3 / 3** (все!) |

motorcycle worst case — **все** val sources утекли. Val метрики завышены.

### Что есть в train_classifier.py

- ✅ Class weights: `N_total / (N_classes × N_class)` — есть
- ✅ Val split по source — есть
- ✅ Dead window filtering (UDP drops) — есть
- ✅ Embedding cache — есть
- ❌ Sample weights (real vs aug) — **нет**
- ❌ Leakage prevention для aug → val — **нет**
- ❌ Background undersampling — **нет**
- ❌ Mixup на embeddings — **нет**

## Итерация 1 — 2026-10-13

### Что сделано

1. ✅ **Data leakage fix** — `source_name()` теперь strip'ит `aug_`/`aug_q_` prefix и `_XXXXXXXX` hash suffix → aug файлы попадают в тот же split что оригинал
2. ✅ **Урезаны aug**: siren 792→166 (2×), motorcycle 936→160 (2×)
3. ✅ **Sample weights**: real=1.0, aug/external=0.4, умножаются на class weights
4. ✅ **Class weights**: формула `sqrt(N_total / N_class)` (TRAINING_GUIDE)
5. ✅ **Undersample background**: 36748→2000 windows в train
6. ✅ **Пересоздан val split** (удалён старый `val_sources.json`)

### Результаты vs baseline

| Класс | Baseline | После | Дельта |
|---|---|---|---|
| **siren** | 76.4% | **79.8%** | **+3.4%** ✓ |
| **motorcycle** | 57.4% | **65.8%** | **+8.4%** ✓ |
| background | 89.5% | 81.4% | -8.1% (ожидаемо) |
| explosion | 78.8% | 71.3% | -7.5% |

Background accuracy снизилась потому что модель перестала «лениться» предсказывать
доминирующий класс. siren и motorcycle — главные цели — значительно выросли.

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

1. **Набрать реальных данных** — motorcycle и explosion нуждаются в real recordings
2. **Mixup на embeddings** — смешивание двух embeddings, λ ~ Beta(0.2, 0.2)
3. **Noise profile** — записать 2-3 мин тишины с INMP441 для `--noise-profile`
4. **Задеплоить** — `make deploy-model` чтобы проверить на живом потоке
