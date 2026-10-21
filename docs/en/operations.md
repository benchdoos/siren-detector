# Operations

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
docker compose up -d --build detector
docker compose logs -f detector
```

## Web UI and settings

The web UI is served at `http://<host>:8080`. It displays live scores, sources, event history, and saved clips. The settings page controls class thresholds, smoothing, cooldowns, audio sources, and inference mode. Persistent detector settings and clips are stored in the Docker `clips` volume. Defaults, valid ranges, and parameter descriptions are listed in the [settings reference](settings.md).

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

## Troubleshooting

- **No source appears:** check ESPHome logs, target IP/port, and that UDP `9000` is reachable on the Hub host.
- **Hub sees audio but detector does not:** check `DETECTOR_PORT` and the Hub forwarding configuration in `hub/docker-compose.yml`.
- **No events:** confirm the correct source is enabled, review thresholds and smoothing, and inspect live scores before changing thresholds.
- **First startup is slow:** YAMNet is fetched on first use and cached in a Docker volume.
- **High CPU:** use `round_robin`, increase the inference interval, or disable optional inference work where appropriate.
