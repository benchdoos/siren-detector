# Siren Detector API

This page documents the application's HTTP API and WebSocket interface. It is
the API for managing and consuming application data, not the contract for a
pluggable source, processor, or output adapter. Target component boundaries are
described in the [component API](component-api.md); current and target system
diagrams are in [Architecture](architecture.md).

The REST API is served by the same server as the web UI. For a running instance, FastAPI's interactive docs are at `http://<host>:8080/docs` and OpenAPI JSON at `http://<host>:8080/openapi.json`. OpenAPI is generated automatically from the application's REST routes. WebSocket routes are not included and are documented below.

When authentication is enabled, protected requests accept the admin bearer token or an OIDC session cookie. Roles are `viewer` (read), `editor` (read and label/segment events), and `admin` (settings and calibration). Cookie-authenticated editor/admin writes also require the `X-CSRF-Token` returned by `/api/auth/status`. `ADMIN_TOKEN` remains available for CLI tools. With authentication disabled, access checks are off.

## REST routes

| Method and path | Access when auth is enabled | Purpose |
|---|---|---|
| `GET /api/auth/status` | Public | Reports auth mode, current role, and session CSRF token. |
| `POST /api/auth/check` | Public | Checks a token from JSON body `{"token":"…"}`. |
| `/auth/login`, `/auth/callback`, `/auth/logout` | Public/session | OIDC sign-in and sign-out routes when OIDC is configured. |
| `GET /api/settings` | Admin | Current settings, defaults, and DSP model availability. |
| `GET /api/runtime/status` | Viewer | Audio subscriber states, queue depth, dropped frames, and callback errors. |
| `GET /metrics` | Viewer | Prometheus 0.0.4 text metrics for engine, audio queue, and output health. |
| `GET /events` | Viewer | One-way SSE stream of the current snapshot and score/event updates. |
| `GET /api/engines/manifests` | Viewer | Locally registered engine manifests. |
| `GET /api/engine-graph` | Admin | Stored graph and configured engine availability. |
| `PUT /api/engine-graph` | Admin | Validates and stores `engine-graph.v1`. |
| `POST /api/settings` | Admin | Updates provided settings; invalid types, ranges, or modes return `400`. |
| `POST /api/settings/reset` | Admin | Resets settings to defaults. |
| `GET /api/sources` | Viewer | Known source status and custom names. |
| `GET /api/metrics` | Viewer | Validation metrics when available. |
| `GET /api/audits` | Viewer | Recent model and dataset audits. |
| `GET /api/calibrate` | Viewer | Background calibration status. |
| `POST /api/calibrate` | Admin | Starts synchronized background recording for one or more sources. Body: `{"source_ids":[0,1],"duration":300}`; duration is 10–900 seconds. A single `source_id` is accepted for backwards compatibility. |
| `POST /api/calibrate/stop` | Admin | Stops and saves each source separately; the response contains a `recordings` array with each `source_id`, source name, and file path. |
| `POST /api/calibrate/cancel` | Admin | Cancels calibration. |
| `GET /api/calibrate/recordings` | Viewer | Lists calibration recordings. |
| `DELETE /api/calibrate/recordings` | Admin | Deletes calibration recordings. |
| `GET /calibrate-audio/{source_name}/{filename}` | Viewer session or token | Downloads calibration audio. |
| `GET /calibrate-audio/{source_name}/{filename}/spectrogram` | Viewer | Calibration recording spectrogram. |

| `POST /feedback` | Editor | Labels a saved clip; body includes `filename` and `label`. |
| `POST /feedback/segments` | Editor | Saves clip segment labels. |
| `POST /api/manual-event` | Admin | Creates a manual event and records a clip. |
| `GET /clips/{filename}` | Viewer session or token | Downloads a WAV clip. |
| `GET /clips/{filename}/spectrogram` | Viewer | Returns a clip spectrogram. |

`GET /api/calibrate` returns overall progress and a `sources` object containing per-source progress and `done` state. Overall progress follows the least advanced source; `done` is true when every source has recorded the requested duration.

Protected routes return `401` or `403` for invalid or missing tokens. See the [settings reference](settings.md) for setting names and ranges.

`GET /api/runtime/status` returns `available`, and `components`, `subscribers`,
and `outputs` arrays. Each component includes `component_id`, `name`, `version`,
`state`, `started_at`, `updated_at`, `error_count`, and `last_error`. Each audio
subscription includes `subscriber_id`, `state`, `queue_depth`,
`queue_capacity`, `overflow`, counters for `frames_received`,
`frames_processed`, `frames_dropped`, `sequence_gaps`, `callback_errors`, and
`last_error`. YAMNet also reports errors from its parallel inference workers;
third-party plugins with their own background threads should report failures
through `runtime.set_component_state(component_id, "degraded", error)`. Hard
timeouts and forced interruption of Python callbacks are not supported.

`GET /api/engines/manifests` lists locally registered engines.
`GET/PUT /api/engine-graph` read and store graph configuration while preserving
unknown fields. A stored graph is activated after the detector restarts and
selects engines and source routes. When a graph is active, an explicit
`--engine` is rejected so there is one source of engine selection.

## WebSocket

### `/events` (Server-Sent Events)

Connect with `GET http://<host>:8080/events` and `Accept: text/event-stream`.
The stream starts with `session` and `score` events, then forwards event-bus
updates; keep-alive comments are sent while idle. A viewer token is required
when authentication is enabled. Slow clients receive recent messages: the
oldest message is dropped when their bounded queue fills.

### `/ws`

Connect to `ws://<host>:8080/ws` (or `wss://` behind a TLS proxy). The server first sends `{"type":"session","started_at":...}`, then the latest score snapshot and available event history, followed by updates from the event bus. Viewer authentication is required when enabled.

### `/ws/audio`

Connect to `ws://<host>:8080/ws/audio` to receive binary float32 PCM audio. The optional `?source_id=<ID>` parameter filters to one source. When authentication is enabled, this route requires viewer access; rejected connections close with code `4401`.

## Specification source

FastAPI publishes OpenAPI at `/openapi.json` and Swagger UI at `/docs`. The schema is generated from REST contracts. Use the target server's OpenAPI for exact request bodies, responses, and errors.
