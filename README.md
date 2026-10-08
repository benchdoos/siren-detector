# Sound Detector

Масштабируемая open-source платформа для обнаружения звуковых событий с упором на дешёвые edge-устройства, локальную обработку и возможность подключения внешних движков распознавания.

## 1. Концепция проекта

### Идея

Сделать не отдельный «детектор сирены», а универсальную платформу **Audio Event Detection**.

Первичные классы для проекта:

- 🚨 `siren` — сирены, в первую очередь системы гражданского оповещения;
- 💥 `explosion` — взрывы и похожие импульсные события;
- 🛵 `motorcycle` — мопеды/мотоциклы;
- 🔇 `background` / `unknown` — обычные и неуверенно классифицируемые звуки.

Ключевой принцип:

> **ESP32 — это универсальный Audio Node, а не обязательный ML-движок.**

Один и тот же Audio Node должен уметь работать:

1. полностью автономно с локальной TinyML-моделью;
2. с Raspberry Pi;
3. с обычным Linux/PC;
4. с другим сервером или edge-компьютером;
5. в будущем — с облачным или специализированным inference engine.

### Целевая архитектура

```text
                    ┌───────────────────┐
                    │      INMP441      │
                    └─────────┬─────────┘
                              │ I2S
                              ▼
                    ┌───────────────────┐
                    │     ESP32-S3      │
                    │                   │
                    │   Audio Node      │
                    │                   │
                    │ capture / buffer  │
                    │ optional TinyML   │
                    └─────────┬─────────┘
                              │
                    Audio Transport
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
       Raspberry Pi          PC              Server
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    Detection Engine
                              │
                       Event Protocol
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Home Assistant       MQTT          REST/API
```

## 2. Архитектурные принципы

### Разделение ответственности

**Audio Node**
- получает звук;
- буферизует PCM;
- передаёт аудио;
- при необходимости выполняет локальное распознавание;
- может выдавать локальное голосовое/звуковое оповещение.

**Detection Engine**
- извлекает признаки;
- запускает ML-модель;
- оценивает confidence;
- объединяет несколько временных окон;
- принимает решение о событии.

**Integration**
- Home Assistant;
- MQTT;
- REST;
- WebSocket;
- webhook;
- другие потребители событий.

Ни один из этих компонентов не должен быть жёстко связан с конкретным производителем железа.

### Interchangeable Engine

ESP32 не должен знать, какая именно нейросеть находится на сервере.

Возможные движки:

- YAMNet;
- собственная TensorFlow/TFLite-модель;
- LiteRT/TensorFlow Lite Micro;
- ONNX Runtime;
- PyTorch-based inference;
- Edge Impulse;
- специализированный серверный inference engine.

### Локальная автономность

В перспективе ESP32-S3 должен иметь возможность работать без Raspberry Pi:

```text
INMP441
   ↓
ESP32-S3
   ↓
TinyML
   ↓
event
   ↓
local audio notification
```

Внешний движок при этом остаётся опциональным.

---

# 3. Технологический стек

## Hardware

### Audio Node

- ESP32-S3 DevKitC-1 N16R8
- 8 MB PSRAM
- Wi-Fi
- INMP441 I2S MEMS microphone
- в будущем — небольшой динамик/усилитель для локального голосового оповещения

### Server / Edge

- Raspberry Pi 5
- Linux
- Docker / Docker Compose

## Firmware / device layer

- ESPHome 2026.9.1 — текущий этап разработки
- ESP-IDF
- I2S audio
- mDNS
- ESPHome API
- OTA

ESPHome используется сейчас для быстрого bring-up железа и управления устройством. Для универсального Audio Node в дальнейшем аудиотранспорт может быть вынесен в отдельный firmware/application layer.

## ML / Audio

Первый baseline:

- YAMNet
- AudioSet

Будущие направления:

- собственная компактная CNN;
- MFCC;
- log-mel spectrogram;
- INT8 quantization;
- LiteRT / TensorFlow Lite Micro;
- ESP-NN;
- Edge Impulse.

## Transport

Первый планируемый транспорт:

- WebSocket
- PCM audio stream

Возможные дополнительные варианты:

- UDP;
- TCP;
- MQTT для событий/команд;
- Serial/USB;
- другие протоколы.

## Integrations

Первый целевой consumer:

- Home Assistant

Дополнительно:

- MQTT;
- REST API;
- WebSocket;
- webhook;
- Node-RED;
- собственные приложения.

---

# 4. Текущий стенд

## Сеть

| Устройство | Адрес |
|---|---|
| MikroTik | `192.168.88.1` |
| Raspberry Pi 5 | `192.168.1.100` |
| ESP32-S3 | `192.168.88.49` |
| ESP32 hostname | `siren-detector.local` |
| Wi-Fi SSID | `Batcave` |

Секреты Wi-Fi и пароли в README не хранятся.

## Raspberry Pi

Проект ESPHome:

```text
~/Projects/esphome/
├── docker-compose.yml
├── .env
└── esphome/
    ├── secrets.yaml
    └── siren-detector.yaml
```

Текущий Docker Compose:

```yaml
services:
  esphome:
    image: ghcr.io/esphome/esphome:2026.9.1
    container_name: esphome
    restart: unless-stopped
    network_mode: host
    environment:
      ESPHOME_USERNAME: ${ESPHOME_USERNAME}
      ESPHOME_PASSWORD: ${ESPHOME_PASSWORD}
    volumes:
      - ./esphome:/config
      - /etc/localtime:/etc/localtime:ro
```

## Текущая ESPHome-конфигурация

Базовые возможности:

- ESP32-S3;
- ESP-IDF;
- Wi-Fi;
- ESPHome API;
- OTA;
- captive portal;
- Wi-Fi signal sensor;
- uptime sensor;
- status/online sensor.

### Первый аудиотест: INMP441

В `esp32/siren-detector.yaml` добавлен I²S-захват без ML:

| INMP441 | ESP32-S3 |
|---|---|
| `VDD` | `3V3` |
| `GND` | `GND` |
| `SCK` / `BCLK` | `GPIO4` |
| `WS` / `LRCLK` | `GPIO5` |
| `SD` / `DOUT` | `GPIO6` |
| `L/R` | `GND` (левый канал) |

Параметры захвата: mono, `16 kHz`, данные микрофона в `32-bit I²S`-слотах. Recorder преобразует поток в WAV `16-bit PCM`.

Для проверки в Home Assistant появился переключатель `Microphone capture`:

- включить — начать чтение I²S;
- выключить — остановить чтение;
- через 5 секунд после загрузки ESP32 захват автоматически включается.

При активном захвате ESP32 примерно раз в полсекунды пишет размер блока и уровни сигнала (`rms` и `peak`). Это проверка наличия реального аудиосигнала, а не детектор событий.

Перед первой прошивкой нужно проверить конфиг на Raspberry Pi:

```bash
cd ~/Projects/esphome
docker compose run --rm esphome config /config/siren-detector.yaml
```

Затем прошить по OTA:

```bash
docker compose run --rm esphome upload /config/siren-detector.yaml --device siren-detector.local
docker compose run --rm esphome logs /config/siren-detector.yaml --device siren-detector.local
```

Через 5 секунд после загрузки `Microphone capture` включается автоматически. В Home Assistant его можно выключить вручную; после следующей перезагрузки захват снова запустится.

Ожидаемый формат строки в логах:

```text
INMP441 data: 1024 bytes, chunks=32, samples=256, rms=0.00..., peak=0.00...
```

## UDP recorder: первый рабочий прототип

ESP32 отправляет I²S-пакеты по UDP на Raspberry Pi 5 (`192.168.1.100:9000`). Захват включается при запуске устройства. Каждый пакет получает порядковый номер: если UDP потеряет пакет, recorder вставит на его месте тишину, а не укоротит WAV. Рекордер сохраняет поток как WAV `16 kHz / mono / 16 bit` и не требует сторонних Python-библиотек.

Recorder вынесен в отдельную папку `recorder/`:

```text
recorder/
├── docker-compose.yml
├── recorder-files/
│   ├── Dockerfile
│   └── udp_recorder.py
└── dataset/
    ├── siren/
    ├── explosion/
    ├── motorcycle/
    ├── background/
    ├── unknown/
    └── inbox/
```

Скопировать папку на Raspberry Pi:

```bash
scp -r recorder doos@192.168.1.100:~/Projects/esphome/
ssh doos@192.168.1.100
cd ~/Projects/esphome/recorder
mkdir -p dataset/{siren,explosion,motorcycle,background,unknown,inbox}
docker compose build
LABEL=background DURATION=10 docker compose run --rm recorder
```

После изменений в формате UDP-пакетов нужно заново собрать и прошить ESP32, а затем пересобрать recorder:

```bash
cd ~/Projects/esphome/recorder
docker compose build --no-cache
```

`LABEL` выбирает подпапку, а `DURATION` задаёт длину записи в секундах. Например, команда выше создаст 10-секундный WAV в `dataset/background/` и завершится автоматически. Допустимые метки: `siren`, `explosion`, `motorcycle`, `background`, `unknown`, `inbox`. Если `DURATION` не задан или равен `0`, запись идёт до `Ctrl-C`.

Примеры:

```bash
LABEL=siren DURATION=10 docker compose run --rm recorder
LABEL=background DURATION=30 docker compose run --rm recorder
docker compose run --rm recorder
```

Последняя команда сохраняет непрерывную запись в `dataset/inbox/`.

## Готовые звуки для быстрого старта

Чтобы быстрее проверить будущий классификатор, можно использовать внешний набор
[ESC-50](https://github.com/karolpiczak/ESC-50): 2 000 пятисекундных записей,
в том числе `siren`, `rain`, `wind`, `washing_machine`, `keyboard_typing`,
`engine` и `fireworks`. В нём нет отдельных классов «посудомойка» и «3D-принтер».

Важно: ESC-50 распространяется под **CC BY-NC 3.0** (некоммерческое
использование); отдельное подмножество ESC-10 — CC BY 3.0. В архиве есть
атрибуция авторов исходных клипов. Перед коммерческим использованием нужно
проверить права на каждый материал. Поэтому аудио не хранится в Git и не
смешивается с собственными записями.

Структура данных:

```text
dataset/
├── siren/                 # собственные записи с recorder
├── explosion/
├── motorcycle/
├── background/
├── unknown/
├── inbox/
└── external/
    └── esc50/             # скачивается отдельно; не коммитить аудио
        ├── audio/
        ├── meta/esc50.csv
        ├── LICENSE
        └── SOURCE.md
```

Из корня репозитория запусти:

```bash
python3 dataset/fetch_esc50.py
```

Скрипт скачает официальный архив и извлечёт только выбранные категории
(примерно 280 пятисекундных клипов), сохранив оригинальные имена, полный CSV
метаданных и LICENSE. Архив временный; сами WAV останутся локально в
`dataset/external/esc50/audio/` и не попадут в Git. Не переименовывай файлы:
CSV связывает их с классом, fold и источником. Соответствия для sanity-check:

| ESC-50 class | Использование при проверке | Наша целевая метка |
|---|---|---|
| `siren` | сирена | `siren` |
| `rain`, `wind`, `washing_machine`, `keyboard_typing`, `engine` | разнообразный фон | `background` |
| `fireworks` | только похожий импульсный звук, не подтверждённый взрыв | `unknown` |

Не копируй внешние клипы в папки `dataset/siren/` или `dataset/background/`:
это скроет происхождение и может привести к утечке одинаковых исходных записей
между обучением и проверкой. При сравнении разделяй данные по `fold` из CSV;
результат на ESC-50 — только быстрый sanity-check, не оценка качества на
микрофоне INMP441. Для реального качества обязательно нужны собственные записи
с устройства. У ESC-50 возможны особенности исходной обработки, влияющие на
обучение и метрики.

Лицензия и карточка набора:
[ESC-50 README и лицензия](https://github.com/karolpiczak/ESC-50).

Перед прошивкой версии с UDP-транспортом на Raspberry Pi выполнить:

```bash
cd ~/Projects/esphome
docker compose run --rm esphome config /config/siren-detector.yaml
docker compose run --rm esphome compile /config/siren-detector.yaml
docker compose run --rm esphome upload /config/siren-detector.yaml --device siren-detector.local
```

Устройство успешно подключается к ESPHome API:

```text
siren-detector.local:6053
```

OTA:

```text
siren-detector.local:3232
```

Текущий Wi-Fi уровень в момент проверки был около `-35 dB`.

## Home Assistant

ESPHome-интеграция устройства уже подключена.

Текущие сущности включают:

- Wi-Fi Signal;
- Uptime;
- Online.

---

# 5. INMP441

Планируемое подключение:

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| SCK / BCLK | GPIO4 |
| WS / LRCLK | GPIO5 |
| SD / DOUT | GPIO6 |
| L/R | GND |

Целевой аудиоформат первого прототипа:

```text
PCM
16 kHz
mono
16 bit
```

Первый этап — не распознавание, а проверка качества захвата.

---

# 6. План реализации

## Phase 0 — Bring-up

**Статус: завершён**

- [x] ESP32-S3 определён;
- [x] прошит;
- [x] подключён к Wi-Fi;
- [x] доступен по mDNS;
- [x] ESPHome API работает;
- [x] OTA подготовлен;
- [x] Home Assistant подключён;
- [x] подключён INMP441;
- [x] получить стабильный I²S-поток;
- [x] передать аудио по UDP на Raspberry Pi;
- [x] сохранить и прослушать WAV `16 kHz / mono / 16 bit`.

## Phase 1 — Audio Capture

**Статус: базовый захват и прослушивание подтверждены; проверка качества продолжается.**

Цель:

```text
INMP441
   ↓
ESP32-S3
   ↓
I2S
   ↓
PCM 16 kHz / mono / 16 bit
```

Проверено на стенде: запись `20261008-130016.wav` длительностью `16.864 s`, `dropped=0`. Дальше проверить шумовой фон и собрать размеченные записи в разных условиях.

Дополнительно проверить:

- уровень сигнала;
- собственный шум;
- clipping;
- стабильность I2S;
- качество записи;
- влияние положения устройства;
- работу через окно/балкон.

## Phase 2 — Audio Node

**Статус: впереди.**

Сделать ESP32 независимым от конкретного backend.

Функции:

- audio capture;
- ring buffer;
- chunking;
- timestamp;
- transport;
- heartbeat;
- configuration;
- optional local inference.

## Phase 3 — Audio Transport

**Статус: UDP-прототип работает; WebSocket остаётся следующим вариантом транспорта.**

Текущий прототип использует UDP с sequence number. Recorder показывает wall-clock и статистику потерь. Следующая версия транспорта — WebSocket:

```text
ESP32 ── WebSocket ──> Raspberry Pi
```

Передавать PCM chunks.

Предварительные типы сообщений:

```text
HELLO
CONFIG
AUDIO
PING
PONG
EVENT
MODEL_INFO
```

Протокол должен оставаться независимым от конкретной ML-модели.

## Phase 4 — Recording / Dataset

**Статус: recorder умеет сохранять собственные WAV по категориям; внешний ESC-50 можно добавить отдельно для sanity-check.**

Raspberry Pi должен уметь сохранять реальные записи:

```text
dataset/
├── siren/
├── explosion/
├── motorcycle/
├── background/
├── unknown/
├── inbox/
└── external/esc50/  # сторонний набор с отдельной лицензией и метаданными
```

Особое внимание:

- реальные бытовые шумы;
- ветер;
- дождь;
- речь;
- телевизор;
- автомобили;
- обычные двигатели;
- музыка;
- птицы/животные;
- случайные хлопки.

Большой `background`-датасет необходим для снижения false positive.

## Phase 5 — Baseline Engine

Первый backend:

```text
ESP32
  ↓
WebSocket
  ↓
Raspberry Pi
  ↓
YAMNet
  ↓
classification
```

Цель — получить объективный baseline до обучения собственной модели.

Проверяем:

- siren;
- explosion;
- motorcycle;
- false positives;
- false negatives;
- latency;
- устойчивость к шуму.

## Phase 6 — Event Engine

Сырые predictions не должны сразу превращаться в тревогу.

Например:

```text
window 1 → siren 0.71
window 2 → siren 0.83
window 3 → siren 0.91
window 4 → siren 0.94
             ↓
        CONFIRMED EVENT
```

Нужны:

- confidence threshold;
- temporal smoothing;
- debounce;
- minimum event duration;
- cooldown;
- `unknown/background`.

## Phase 7 — Integrations

Первый вариант:

```text
Detector Engine
      ↓
MQTT
      ↓
Home Assistant
```

Событие должно быть стандартизировано.

Пример:

```json
{
  "event": "siren",
  "confidence": 0.94,
  "timestamp": "2026-10-07T20:00:00Z",
  "source": "siren-detector-01",
  "engine": "yamnet"
}
```

## Phase 8 — собственная модель

После получения реальных данных:

```text
dataset
   ↓
training
   ↓
evaluation
   ↓
quantization
   ↓
TFLite
```

Целевые классы:

```text
siren
explosion
motorcycle
background
unknown
```

## Phase 9 — TinyML на ESP32

Финальная цель:

```text
INMP441
   ↓
ESP32-S3
   ↓
feature extraction
   ↓
TinyML
   ↓
local event
```

Используем:

- LiteRT / TensorFlow Lite Micro;
- INT8 quantization;
- ESP-NN;
- ESP32-S3 hardware capabilities.

При этом внешний Detection Engine остаётся доступным.

---

# 7. Безопасность и качество распознавания

Это проект, связанный с потенциально опасными событиями, поэтому главная метрика — не просто accuracy.

Особенно важны:

- false negative rate;
- false positive rate;
- latency;
- confidence;
- устойчивость к шуму;
- качество на реальных записях;
- стабильность при нескольких источниках звука.

### Не делать

Не превращать одно слабое prediction в категоричное сообщение:

```text
"ВЗРЫВ!!!"
```

Правильнее иметь состояния:

```text
possible
probable
confirmed
```

и учитывать несколько последовательных окон.

### Unknown — обязательный класс

Модель не должна быть вынуждена выбирать один из трёх опасных классов для любого звука.

```text
siren
explosion
motorcycle
background
unknown
```

---

# 8. Почему не привязываемся к Raspberry Pi

Raspberry Pi — удобный и дешёвый inference host для домашней установки, но не архитектурное требование.

Возможны:

```text
ESP32 → Raspberry Pi
ESP32 → PC
ESP32 → NAS
ESP32 → mini-PC
ESP32 → server
ESP32 → cloud
ESP32 → local TinyML
```

Таким образом пользователь может выбрать уровень сложности:

### Минимум

```text
ESP32 + microphone
```

### Домашний

```text
ESP32 + Raspberry Pi
```

### Продвинутый

```text
ESP32 + powerful PC/server + custom model
```

### Автономный

```text
ESP32 + TinyML
```

---

# 9. Non-goals

Проект не должен:

- быть привязан к Home Assistant;
- требовать Raspberry Pi;
- требовать облака;
- требовать конкретной ML-модели;
- требовать ESPHome в качестве конечного firmware;
- ограничиваться только сиренами;
- пытаться классифицировать абсолютно любой звук.

---

# 10. Текущий статус

**Проект:** `Sound Detector`

**Фаза:** Audio Node bring-up.

Уже работает:

```text
ESP32-S3
   ↓
Wi-Fi
   ↓
ESPHome API
   ↓
Home Assistant
```

Ближайшая задача:

```text
INMP441
   ↓
I2S
   ↓
ESP32-S3
   ↓
проверка PCM
```

После этого:

```text
ESP32
   ↓
WebSocket
   ↓
Raspberry Pi
   ↓
WAV
```

И только затем подключаем первый Detection Engine.

---

# 11. Долгосрочная цель

Сделать доступную и расширяемую платформу:

> **дешёвый микрофонный Audio Node + сменный inference engine + стандартизированные события + локальные интеграции.**

Первый практический сценарий — обнаружение потенциально опасных звуков:

```text
🚨 Siren
💥 Explosion
🛵 Motorcycle
```

Но архитектура должна позволять добавлять новые модели и классы без изменения базового устройства.

---

## Project Philosophy

**Cheap hardware.  
Local-first.  
Open architecture.  
Replaceable intelligence.  
Real-world data.  
Safety over hype.**
