# Siren Detector API

This page documents the application's HTTP API and WebSocket interface. It is
the API for managing and consuming application data, not the contract for a
pluggable source, processor, or output adapter. Target component boundaries are
described in the [component API](component-api.md); current and target system
diagrams are in [Architecture](architecture.md).

The REST API is served by the same server as the web UI. For a running instance, FastAPI's interactive docs are at `http://<host>:8080/docs` and OpenAPI JSON at `http://<host>:8080/openapi.json`. OpenAPI is generated automatically from the application's REST routes. WebSocket routes are not included and are documented below.

When authentication is enabled (`AUTH_ENABLED` in `hub/.env`), protected requests must include a token checked for viewer or admin access. Reads generally require viewer access; settings updates, feedback, and calibration controls require admin access. Public exceptions include `/api/auth/status`, `/api/auth/check`, clip downloads, and calibration audio downloads; check the deployed configuration. With authentication disabled, requests are not restricted by tokens.

## REST routes

| Method and path | Access when auth is enabled | Purpose |
|---|---|---|
| `GET /api/auth/status` | Public | Reports whether authentication is enabled. |
| `POST /api/auth/check` | Public | Checks a token from JSON body `{"token":"…"}`. |
| `GET /api/settings` | Viewer | Current settings, defaults, and DSP model availability. |
| `GET /api/runtime/status` | Viewer | Audio subscriber states, queue depth, dropped frames, and callback errors. |
| `POST /api/settings` | Admin | Updates provided settings; invalid types, ranges, or modes return `400`. |
| `POST /api/settings/reset` | Admin | Resets settings to defaults. |
| `GET /api/sources` | Viewer | Known source status and custom names. |
| `GET /api/metrics` | Viewer | Validation metrics when available. |
| `GET /api/audits` | Viewer | Recent model and dataset audits. |
| `GET /api/calibrate` | Viewer | Background calibration status. |
| `POST /api/calibrate` | Admin | Starts background calibration; see Swagger UI for the exact schema. |
| `POST /api/calibrate/stop` | Admin | Stops and saves current calibration. |
| `POST /api/calibrate/cancel` | Admin | Cancels calibration. |
| `GET /api/calibrate/recordings` | Viewer | Lists calibration recordings. |
| `DELETE /api/calibrate/recordings` | Admin | Deletes calibration recordings. |
| `GET /calibrate-audio/{source_name}/{filename}` | Viewer token in query or Authorization | Downloads calibration audio. |
| `GET /calibrate-audio/{source_name}/{filename}/spectrogram` | No route-level check | Calibration recording spectrogram. |
| `POST /feedback` | Admin | Labels a saved clip; body includes `filename` and `label`. |
| `POST /feedback/segments` | Admin | Saves clip segment labels. |
| `POST /api/manual-event` | Admin | Creates a manual event and records a clip. |
| `GET /clips/{filename}` | Viewer token in query or Authorization | Downloads a WAV clip. |
| `GET /clips/{filename}/spectrogram` | No route-level check | Returns a clip spectrogram. |

Protected routes return `401` or `403` for invalid or missing tokens. See the [settings reference](settings.md) for setting names and ranges.

`GET /api/runtime/status` returns `available`, a `components` array, and a
`subscribers` array. Each component includes `component_id`, `name`, `version`,
`state`, `started_at`, `updated_at`, `error_count`, and `last_error`. Each audio
subscription includes `subscriber_id`, `state`, `queue_depth`,
`queue_capacity`, `overflow`, counters for `frames_received`,
`frames_processed`, `frames_dropped`, `sequence_gaps`, `callback_errors`, and
`last_error`. YAMNet also reports errors from its parallel inference workers;
third-party plugins with their own background threads should report failures
through `runtime.set_component_state(component_id, "degraded", error)`. Hard
timeouts and forced interruption of Python callbacks are not supported.

## WebSocket

### `/ws`

Connect to `ws://<host>:8080/ws` (or `wss://` behind a TLS proxy). The server first sends `{"type":"session","started_at":...}`, then the latest score snapshot and available event history, followed by updates from the event bus. This route does not check a token.

### `/ws/audio`

Connect to `ws://<host>:8080/ws/audio` to receive binary float32 PCM audio. The optional `?source_id=<ID>` parameter filters to one source. When authentication is enabled, this route requires a viewer token; rejected connections close with code `4401`.

## Specification source

FastAPI publishes OpenAPI at `/openapi.json` and Swagger UI at `/docs`. The schema is generated from REST contracts. Use the target server's OpenAPI for exact request bodies, responses, and errors.
