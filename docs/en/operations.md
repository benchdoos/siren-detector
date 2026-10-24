# Operations

For an existing installation, see the [upgrade and migration guide](migration.md)
for backing up `clips`, preserving production `.env` files, and enabling the
multi-engine graph.

## Services and ports

| Service | Default port | Notes |
|---|---:|---|
| Hub UDP | `9000` | Receives ESP32 and audio-source packets |
| Recorder UDP | `9001` | Hub destination when recording is enabled |
| Detector UDP | `9002` | Hub destination for inference |
| Detector web UI/API | `8080` | Dashboard, settings, and HTTP API |

Ports can be changed in `hub/.env`. The detector runs with host networking in the provided Compose file.

## Deploy from a development machine

After configuring the root `.env`, common Makefile targets are:

```bash
make deploy       # sync detector and update/rebuild the service
make deploy-ui    # sync only web assets
make logs         # follow detector logs on the configured host
make restart      # restart the detector container
```

For Compose-managed updates on the host:

```bash
cd hub
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build detector
docker compose logs -f detector
```

## Web UI and settings

The web UI is served at `http://<host>:8080`. It displays live scores, sources, event history, and saved clips. The settings page controls class thresholds, smoothing, cooldowns, audio sources, and inference mode. Persistent detector settings and clips are stored in the Docker `clips` volume. Defaults, valid ranges, and parameter descriptions are listed in the [settings reference](settings.md).

The dashboard and settings page support English and Russian. Use the language selector in the page header to switch languages; the choice is saved in this browser. On first visit, the UI uses the browser's preferred language when it is Russian and English otherwise.

Authentication is controlled by `AUTH_ENABLED` and `ADMIN_TOKEN` in `hub/.env`. If authentication is enabled and no token is set, the detector generates a token at startup; check container logs. Keep the token private.

## MQTT and Home Assistant

MQTT is optional. Set `MQTT_HOST`, `MQTT_PORT`, and, if needed, `MQTT_USER`/`MQTT_PASSWORD` in `hub/.env`, then recreate the detector container. The integration publishes detector state and event information under the `sound_detector` topic namespace and supports Home Assistant discovery when a broker is available. `sound_detector/events` carries JSON event updates with `engine_id`, `source_id`, source name, event, state, confidence, and timestamp; existing retained Home Assistant state topics remain available.

## Webhooks

Set `WEBHOOK_URLS` to a comma-separated list of HTTP(S) endpoints to receive POST requests containing `detector-event.v1` JSON. `WEBHOOK_TOKEN` optionally sends a shared Bearer token, and `WEBHOOK_TIMEOUT_SECONDS` controls the per-endpoint timeout (default 2 seconds). Delivery uses a bounded queue of 128 events and drops the oldest pending event when full. Failed requests are logged and counted in Settings → Runtime; they are not retried. Webhooks receive events, not score updates.

## HTTP and WebSocket API

The [API reference](api.md) lists routes and access rules. FastAPI's interactive REST docs are at `http://<host>:8080/docs`; the OpenAPI specification is at `http://<host>:8080/openapi.json`.

The API shares the web UI port. Key routes include:

| Route | Method | Purpose |
|---|---|---|
| `/api/settings` | GET/POST | Read or update settings (POST requires admin access) |
| `/api/sources` | GET | List known audio sources |
| `/metrics` | GET | Scrape Prometheus runtime counters and gauges (requires viewer auth) |
| `/api/metrics` | GET | Read validation metrics when present |
| `/api/audits` | GET | Read model/dataset audit snapshots |
| `/api/calibrate` | GET/POST | Check or start background calibration |
| `/api/calibrate/stop` | POST | Finish and save calibration |
| `/api/calibrate/cancel` | POST | Cancel calibration |
| `/feedback` | POST | Label a saved clip |
| `/ws` | WebSocket | Live events, scores, and status |
| `/ws/audio` | WebSocket | Live PCM audio stream |

Viewer routes require a valid token when authentication is enabled; state-changing routes require admin access. See the [API reference](api.md) and the running server's Swagger UI for routes and request/response schemas.

## Makefile reference

All targets assume the root `.env` is configured with `RPI`, `RPI_BASE`, `CONTAINER`, and `WEB_UI`.

### Deploy

| Target | Description |
|---|---|
| `make deploy` | Full rebuild: sync detector code + Compose file to RPi, rebuild container. |
| `make deploy-ui` | Sync only `detector/static/` (no container rebuild). |
| `make deploy-audio` | Sync `audio_source/` to RPi and rebuild its container. |
| `make deploy-firmware` | Copy ESP32 YAML to RPi, validate, compile, and flash OTA. |
| `make deploy-compose` | Copy `hub/docker-compose.yml` to RPi (no rebuild). |
| `make deploy-model` | Push trained weights (MLP + DSP + audits) to RPi and restart. |
| `make restart` | Restart the detector container (no rebuild). |
| `make logs` | Tail detector container logs. |
| `make logs-esp` | Stream ESP32 logs via ESPHome on RPi. |

### ML and retraining

| Target | Description |
|---|---|
| `make retrain` | Full cycle: merge recorder -> sync -> augment -> train DSP + MLP -> audit -> deploy -> clean. |
| `make sync-dataset` | Sync labeled data from RPi to local Mac. |
| `make augment` | Augment siren + explosion + uav (cleans old `aug_*` first). |
| `make augment-siren` | Augment siren class only. |
| `make augment-explosion` | Augment explosion class only. |
| `make augment-uav` | Augment UAV class (normal + quiet). |
| `make train` | Train MLP classifier on local dataset. |
| `make train-dsp` | Train DSP pre-filter (Conv1D on DSP features). |
| `make audit` | Run dataset and model audit, save snapshot. |
| `make audit-compare` | Compare latest audit with previous (or specify `OLD=file NEW=file`). |
| `make package-retrain-release` | Package retrain outputs for a GitHub Release. |

### Cleanup

| Target | Description |
|---|---|
| `make clean-clips` | Delete all saved clips on RPi. |
| `make clean-dataset` | Delete all labeled WAVs on RPi (sync first!). |
| `make clean-aug` | Remove all `aug_*` files locally before regenerating. |
| `make clean-docker` | Prune Docker images and build cache on RPi (preserves volumes). |

### Utilities

| Target | Description |
|---|---|
| `make settings` | Dump current runtime settings from detector as JSON. |
| `make open` | Open web UI in browser. |
| `make unlock` | Add SSH key to agent for 10 minutes. |
| `make fetch-esc50` | Download and extract the ESC-50 dataset. |
| `make help` | Show available commands. |

## Troubleshooting

- **No source appears:** check ESPHome logs, target IP/port, and that UDP `9000` is reachable on the Hub host.
- **Hub sees audio but detector does not:** check `DETECTOR_PORT` and the Hub forwarding configuration in `hub/docker-compose.yml`.
- **No events:** confirm the correct source is enabled, review thresholds and smoothing, and inspect live scores before changing thresholds.
- **First startup is slow:** YAMNet is fetched on first use and cached in a Docker volume.
- **High CPU:** use `round_robin`, increase the inference interval, or disable optional inference work where appropriate.
