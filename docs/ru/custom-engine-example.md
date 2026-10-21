# Пример стороннего движка

В репозитории есть минимальный движок
`detector/examples/energy_threshold_engine.py`. Он показывает границы
интеграции, а не качество распознавания: считает RMS-энергию кадра и использует
её как score класса `siren`, чтобы результат был виден в текущем UI. Это **не
детектор сирены**.

## Путь сообщения

```text
UDPSourceAdapter → AudioFrame → Runtime Hub → EnergyThresholdEngine
                                      ↑                 │
                                      └── score/event ──┘
                                                ↓
                                   EventBus → WebSocket / MQTT
```

Движок подписывается на нормализованные кадры `audio-frame.v1`. Для каждого
кадра он публикует score через `RuntimeHub`; при переходе через порог формирует
event, а для карточки UI просит Hub сохранить короткий клип через настроенный
storage adapter. Сам движок не открывает UDP и не публикует напрямую в UI или
MQTT.

## Запуск

Из корня репозитория запустите Detector, указав dotted path примера:

```sh
python -m detector.main \
  --engine detector.examples.energy_threshold_engine.EnergyThresholdEngine \
  --energy-event-threshold 0.08 \
  --energy-release-threshold 0.05
```

Остальные параметры запуска и настройки аудиовхода остаются такими же, как у
обычного Detector. Пример включает manifest, который можно посмотреть так:

```sh
python -m detector.main \
  --engine detector.examples.energy_threshold_engine.EnergyThresholdEngine \
  --show-engine-manifest
```

Если в `settings.json` заданы `energy_event_threshold`,
`energy_release_threshold` или `energy_clip_seconds`, они имеют приоритет над
CLI значениями. Текущий Settings UI пока не строится из manifest, поэтому эти
параметры не появятся в форме автоматически.

При достаточной амплитуде UI score-bar `siren` поднимется, а карточка события
появится после сохранения WAV-клипа. Та же score/event рассылка проходит через
существующие WebSocket и MQTT выходы. Подпись engine укажет
`energy-threshold-example`.

## Основа собственного движка

Класс движка должен реализовать `start(runtime, settings, args)`. Он может
объявить `ENGINE_MANIFEST`, реализовать `add_arguments(parser)` для параметров
CLI и подписаться на аудио:

```python
runtime.subscribe_audio(on_audio_frame)
```

Публикация результата:

```python
from detector.runtime.contracts import RuntimeMessage

runtime.publish(RuntimeMessage.from_legacy({
    "type": "score",
    "source_id": frame.source_id,
    "source": "my microphone",
    "top": "My model",
    "siren_score": 0.82,
    "siren_state": "possible",
    "siren_triggers": 1,
    "siren_possible_needed": 1,
    "siren_confirmed_needed": 3,
}))
```

Для события используется `type="event"` с `event`, `state`, `confidence`,
`timestamp`, `source_id`, `source`, `engine` и `clip_url`. В текущем интерфейсе
карточка события отображается только если `clip_url` указывает на сохранённый
клип. Для совместимости отправляйте имена классов, которые знает текущий UI
(`siren`, `explosion`, `uav`); произвольные классы и настройки из manifest
пока не настраиваются в веб-интерфейсе.

> Импорт движка выполняется в основном процессе Detector. Подключайте только
> локальный код, которому доверяете; установка Python-плагина из Web UI не
> поддерживается.
