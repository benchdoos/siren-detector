# Переход на новую архитектуру и обновление

Инструкция для обновления существующей установки Detector/Hub и включения
новых возможностей Runtime Hub, контрактов и нескольких движков.

## Что происходит при обновлении

- Формат аудио по сети не изменён: ESP32 и USB-сервис продолжают передавать
  пакеты SDR1/UDP. Перепрошивать ESP32 не нужно.
- Данные Detector — клипы и `settings.json` — остаются в Docker-томе `clips`.
  Пересборка контейнера этот том не удаляет.
- Старые плоские ключи настроек сохраняются и остаются defaults/fallback.
  Пустой `engine_configs` означает прежний режим с одним движком; переносить
  существующие настройки вручную для обычного обновления не нужно.
- Граф движков включается только после его сохранения в Settings → Engines
  (раздел решений).
  Тогда граф и маршруты источников становятся источником выбора движков;
  одновременно передавать `--engine` нельзя.
- Контракты `audio-frame.v1`, `inference-result.v1` и `detector-event.v1` —
  внутренние/выходные версии Detector. Они не меняют SDR1 wire format.

## Перед обновлением

1. Убедитесь, что у вас есть актуальные `hub/.env`, `audio_source/.env`,
   корневой `.env`, Compose-файлы и файлы моделей. Не заменяйте production
   `hub/.env` локальной копией,
   если значения в них отличаются: цель `make deploy` копирует локальный
   `hub/.env` на сервер, когда такой файл существует.
   Для быстрого отката сохраните текущий образ Detector под отдельным тегом и
   запишите его Compose image reference:

   ```bash
   OLD_IMAGE_ID="$(docker inspect siren-detector-engine --format '{{.Image}}')"
   docker inspect siren-detector-engine --format '{{.Config.Image}}'
   docker image tag "$OLD_IMAGE_ID" siren-detector:pre-migration
   ```

   Запишите значение image reference из второй команды: Compose может
   использовать имя, отличающееся от `siren-detector`.
2. Чтобы сохранить только текущие настройки, на сервере выполните из папки,
   куда нужно положить копию:

   ```bash
   docker cp siren-detector-engine:/clips/settings.json ./settings-before-upgrade.json
   ```

   Файл содержит как старые настройки, так и `engine_configs`/маршруты, если
   граф уже настроен. Для сохранения также клипов и sidecar-метаданных сделайте
   архив всего volume `clips`. Узнать имя тома можно так:

   ```bash
   docker inspect siren-detector-engine \
     --format '{{range .Mounts}}{{if eq .Destination "/clips"}}{{.Name}}{{end}}{{end}}'
   ```

   Создайте архив, подставив выведенное имя вместо `CLIPS_VOLUME`:

   ```bash
   docker run --rm \
     -v CLIPS_VOLUME:/data:ro \
     -v "$PWD:/backup" \
     alpine tar czf /backup/clips-backup.tgz -C /data .
   ```

   Также сохраните `hub/.env`, `audio_source/.env` и текущие Compose-файлы вне
   каталогов, которые обновляются. Цели `make deploy` и `make deploy-audio`
   используют `rsync --delete`; перед запуском проверьте, что локальные env-файлы
   содержат production-значения, иначе удалённые копии могут быть заменены или
   удалены. Не удаляйте Docker volume при обновлении.

## Обновить код

**С компьютера разработчика.** Настройте корневой `.env` с `RPI`, `RPI_BASE`
и `CONTAINER`, затем из корня репозитория выполните:

```bash
make deploy
```

Цель синхронизирует `detector/`, Compose-файл, при наличии копирует локальный
`hub/.env`, затем пересобирает и запускает Detector. Чтобы обновить только
USB-сервис после изменения `audio_source/`, отдельно выполните:

```bash
make deploy-audio
```

**На сервере, где уже размещён checkout репозитория.** Сначала обновите файлы
репозитория выбранным способом, затем из каталога `hub/` выполните:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build detector
docker compose logs -f detector
```

Hub и прошивку ESP32 обычно пересобирать не нужно: их сетевой протокол не
менялся. После запуска откройте веб-интерфейс и проверьте источники, Settings → Runtime
и последние логи Detector.

## Проверить старые настройки

`settings.json` остаётся в volume `clips`. При обычном обновлении продолжает
работать существующий одиночный `yamnet` и прежние пороги, сглаживание,
калибровки и имена источников. Не создавайте `engine_configs`, если хотите
оставить прежний режим.

Для нескольких решений откройте Settings → Engines и задайте движки и маршруты.
Эквивалентная часть сохраняемого JSON выглядит так:

```json
{
  "engine_graph_contract": "engine-graph.v1",
  "engine_configs": {
    "yamnet": {
      "selector": "yamnet",
      "enabled": true,
      "settings": {"siren_threshold": 0.1}
    }
  },
  "engine_routes": {
    "default": ["yamnet"],
    "by_source": {"1": ["yamnet"]}
  }
}
```

`settings` — необязательные переопределения defaults для этого решения. Если
не указать их, используются сохранённые legacy-настройки как общие defaults.
Не редактируйте файл, пока Detector работает: Settings API сохраняет его
самостоятельно. После сохранения графа перезапустите Detector. При активном
графе уберите явный `--engine` из собственной команды запуска; Compose-файл
проекта не задаёт этот аргумент по умолчанию.

Для отката остановите Detector, восстановите предыдущий образ/код и
`hub/docker-compose.yml`. Если Compose настроен на предыдущий image reference,
верните ему сохранённый образ перед запуском:

```bash
docker tag siren-detector:pre-migration <IMAGE_REF>
docker compose up -d detector
```

Старый Detector игнорирует неизвестные поля нового
`settings.json`, сохраняя старые ключи. Если нужно полностью вернуть состояние
до обновления, восстановите также резервную копию volume `clips`.

## USB-микрофон и ALSA

USB уже работает как отдельный сервис, без прямого захвата внутри Detector:
`audio_source/` использует `sounddevice`/PortAudio, при необходимости ALSA,
преобразует звук в SDR1/UDP и отправляет его в Hub. На Linux проверьте микрофон
через `arecord -l`, настройте `AUDIO_SOURCE_ID`, `AUDIO_TARGET` и, если нужно,
`AUDIO_DEVICE` в `audio_source/.env`, затем запустите `docker compose up -d`
в `audio_source/`. См. [настройку нескольких микрофонов](multi-microphone.md).

Новый `FileAudioAdapter` принимает mono WAV 16 kHz для локальных тестов и
подключается программно через `RuntimeHub.add_input`; он не заменяет USB-сервис.
Прямой ALSA input adapter внутри Detector пока не реализован и для текущей
USB-схемы не требуется. Реальное обнаружение USB-устройства зависит от Linux,
Docker-доступа к `/dev/snd` и выбранного backend; этот репозиторий не может
проверить наличие конкретного микрофона на вашей Raspberry Pi.

## Новые интерфейсы после обновления

- `GET /api/runtime/status` — состояние движков, очередей, ошибок и выходов.
- `GET /metrics` — Prometheus-метрики (при включённой авторизации нужен viewer).
- `GET /events` — SSE-поток для односторонних HTTP-клиентов.
- `/ws`, MQTT и существующие retained Home Assistant топики сохраняют
  совместимость; события дополнены идентификаторами источника и движка.
- Webhooks выключены, пока не заданы `WEBHOOK_URLS` в окружении Detector.
