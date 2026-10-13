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

## План улучшений

1. **Починить data leakage** — `source_name()` должна возвращать оригинальный source для aug файлов
2. **Урезать aug**: siren max ~166, motorcycle max ~160
3. **Добавить sample weights**: real=1.0, aug=0.3–0.5
4. **Undersample background** до ~1500–2000 windows
5. **Переобучить и сравнить**: `make retrain` → `make audit` → `make audit-compare`
