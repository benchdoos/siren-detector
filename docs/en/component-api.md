# Component API

This document describes the target architecture for replaceable components.
It is a proposed contract: the current code does not yet implement every
interface and format below. The web application's HTTP API is documented
separately in [API](api.md).

## Goal and terminology

The system is composed of replaceable blocks:

```text
sources → input adapters → Runtime Hub ⇄ processing chains
                              │       DSP → YAMNet → MLP
                              └→ export layer → UI / WS / REST / MQTT
```

Each block can be replaced by an implementation that follows its component
contract. A user can replace one stage (such as MLP), the full recognition
chain, an audio source, routing, or an output without coupling the new
component directly to the others.

**Runtime Hub** means the logical router inside Detector. The existing
**UDP Hub** (`hub/hub.py`) is a separate network relay that forwards datagrams.
They have different roles.

### Component types

| Type | Input | Output | Examples |
|---|---|---|---|
| Source | Microphone, file, or network stream | Audio data | ESP32/SDR1, USB/ALSA, file |
| Input adapter | Source protocol or driver | Common `AudioFrame` | SDR1/UDP to PCM frames |
| Runtime Hub | Frames and route configuration | Routed frames/requests | Fan-out, chain selection |
| Processing stage | Frame or previous stage result | Transform or score | DSP, YAMNet, MLP, custom model |
| Event policy | Scores from one or more stages | Normalized events | Thresholds, temporal smoothing, fusion |
| Output adapter | Normalized scores/events | External representation | WebSocket, MQTT, webhook, clips |

A microphone driver belongs to a source/input adapter, not to a model that
receives normalized frames. An LLM or remote model may need a processing
adapter, such as audio windowing or transcript/summary handoff, without
changing the microphone driver.

## Boundaries and responsibilities

### Audio source and input adapter

The adapter handles device and protocol specifics: capture, decoding, format
conversion, and stable `source_id` assignment. It does not classify sirens or
publish MQTT directly.

The internal audio contract normalizes data independently of transport. The
initial project profile is mono, 16 kHz, float32 PCM inside Detector. An
adapter may accept another format and convert it to this profile while keeping
the original source parameters in metadata. Current SDR1/UDP remains one
adapter; the SDR1 wire format is not the common internal contract.

Proposed frame shape:

```json
{
  "contract": "audio-frame.v1",
  "source_id": "0",
  "sequence": 1842,
  "timestamp": "2026-10-21T10:15:30.125Z",
  "sample_rate_hz": 16000,
  "channels": 1,
  "sample_format": "float32",
  "samples": "binary payload",
  "metadata": {"transport": "sdr1/udp"}
}
```

In a real API, `samples` should be a binary block or an internal object, not
JSON-encoded. The frame carries enough timing and sequence information to
detect gaps and preserve order.

The current implementation accepts only `audio-frame.v1`, mono float32 PCM,
and rejects an unknown version or format when constructing a frame. SDR1/UDP
reconstructs missing input packets as silence. If a bounded processor queue
still drops frames, `audio-subscription.v1` counts dropped frames and sequence
gaps. After five seconds without packets, the input resets its expected
sequence and accepts a reconnecting source at its new sequence number.

A callback failure is isolated to that subscriber's queue: it increments the
error count, marks the subscriber `degraded`, and the next successful callback
returns it to `ready`. An engine startup failure is reported as `error` without
preventing other engines from starting. A contract version change requires
explicit support: current dataclasses reject unknown versions with `ValueError`
so an incompatible frame/result cannot silently enter processing.

### Runtime Hub / router

Runtime Hub works in both directions:

1. It receives audio frames from input adapters and routes them to assigned
   processing chains.
2. It receives scores, events, and execution status from chains and associates
   them with the originating `source_id`, frame, and `engine_id`.
3. Based on route configuration, it forwards results to event policy and/or
   the export layer.

So yes: a model returns information to Runtime Hub, and the Hub routes it to
the output layer. Intermediate scores can go directly to monitoring, while
final events can pass through event policy for thresholds, temporal smoothing,
deduplication, and fusion. Runtime Hub itself contains no YAMNet, DSP, MQTT, or
class-specific rules; its job is routing, correlation, and lifecycle.

The first version can use static routing: a source is assigned to one or more
chains. Dynamic rules, conditional stage execution, and a UI graph editor are
later capabilities.

#### Audio subscriber isolation (`audio-subscription.v1`)

Each processing component receives frames through its own bounded queue and
worker. The input adapter only fans frames out to queues; it does not wait for
model inference. Subscribe with
`runtime.subscribe_audio(callback, subscriber_id=..., max_queue_size=8,
overflow="drop_oldest", source_ids={0, 2})`; omit `source_ids` to receive all sources.

| Field | Contract |
|---|---|
| `subscriber_id` | Unique within Runtime Hub; use a stable component/engine ID. |
| `source_ids` | Optional set of source IDs; when set, the subscriber receives only those sources. |
| `max_queue_size` | Positive number of frames; defaults to 8. Bounded to prevent unbounded latency and memory growth. |
| `overflow` | `drop_oldest` (default, favor fresh audio) or `drop_newest`. Enqueue never blocks the input adapter. |
| `frames_received` / `frames_processed` | Frames accepted by the subscription and frames successfully handled by its callback. |
| `frames_dropped` | Frames lost to overflow or discarded from a pending queue during shutdown. |
| `sequence_gaps` | Sequence gaps observed between consecutively processed frames from one source. |
| `callback_errors` / `last_error` | Callback failures; its worker continues and does not stop other subscriptions. |
| `state` | `starting`, `ready`, `degraded`, `stopping`, or `stopped`; a later successful frame returns `degraded` to `ready`. |

`runtime.audio_status()` returns per-subscriber diagnostic snapshots. On Hub
shutdown, inputs stop first, then pending queues are cleared and workers are
stopped. If a callback hangs, the Hub waits up to two seconds; Python threads
cannot be forcibly interrupted, so that daemon worker remains in `stopping`.
Frames are shared by reference: subscribers must not mutate `frame.samples` in
place and should copy it before transforming.

Overflow loses audio only for the affected subscriber: other subscribers and
the live-audio output continue receiving frames. Components must account for
sequence gaps and tolerate incomplete streams. Choose queue size based on
latency, frame rate, and inference speed.

### Processing chain

Processing is configured as an ordered list of components. The default profile
preserves current behavior:

```text
DSP pre-filter → YAMNet embeddings/scores → MLP classifier → event policy
```

DSP, YAMNet, and MLP are replaceable stages/models. An implementation may
combine them in one process for efficiency if it preserves the same external
contract. A chain may emit intermediate scores and a final result. Shared
event policy converts scores into states and events; later it can also fuse
results from multiple chains.

A processor receives an audio frame and invocation context, then returns a
result with the same source/frame identifiers. It should not call the UI,
MQTT, or clip filesystem directly. Stage-specific resources and private state
remain inside that component.

### Export layer and output adapters

The export layer receives normalized results forwarded by Runtime Hub and
converts them for recipients. Its adapters can subscribe to scores, events,
and component status. WebSocket/UI provides live visualization, REST reads
state and manages the system, MQTT/webhook delivers externally, and clip
storage persists audio. Failure of an optional output must not stop audio
input or inference; each output defines buffering, retry, and drop behavior.

## Shared inference result

Processing components publish scores in the current internal
`inference-result.v1` shape with source, engine and stage IDs, timestamp,
namespaced scores and metadata:

```json
{
  "contract": "inference-result.v1",
  "source_id": "0",
  "engine_id": "yamnet",
  "timestamp": "2026-10-21T10:15:30.125Z",
  "stage_id": "mlp",
  "scores": {"siren": 0.87, "background": 0.08},
  "metadata": {"model_version": "local"}
}
```

`engine_id` is required and refers to a stable ID in the manifest; `stage_id`
identifies a stage within that engine. These IDs provide attribution without
embedding model-specific names in the common protocol. A score is model output; by itself it is not a
confirmed event.

An event is created by event policy and carries at least:

```json
{
  "contract": "detector-event.v1",
  "event_id": "unique-id",
  "class": "siren",
  "state": "confirmed",
  "confidence": 0.87,
  "timestamp": "2026-10-21T10:15:30.125Z",
  "source_id": "0",
  "engine_ids": ["default-audio-chain"],
  "evidence": [{"stage_id": "mlp", "score": 0.87}],
  "clip_url": "/clips/example.wav"
}
```

For multiple solutions, an event preserves provenance and evidence. Confirmation,
deduplication, and fusion rules belong to event policy, not hidden UI or MQTT
adapter logic.

### `event-policy.v1` contract

Runtime Hub selects one active policy. It receives each `InferenceResult` via
`process(result)` and publishes score/event messages through
`context.runtime.publish(RuntimeMessage)`. An implementation declares
`policy_id` and `api_version = "event-policy.v1"`. Select the current policy as
`default`; a locally installed Python class can be selected with
`--event-policy package.module.ClassName`. The class is constructed with
`EventPolicyContext`, which provides Runtime Hub, the per-source state registry,
and an optional DSP component.

```python
from detector.runtime.contracts import RuntimeMessage


class MyPolicy:
    policy_id = "my-policy"
    api_version = "event-policy.v1"

    def __init__(self, context):
        self.runtime = context.runtime

    def process(self, result):
        self.runtime.publish(RuntimeMessage.from_legacy({
            "type": "score",
            "top": "custom model",
            "source_id": result.source_id,
            "siren_score": result.scores.get("my-model:siren", 0.0),
        }))
```

Python classes are intended for developer-managed local configuration; the UI
does not install or load user code. `EventPolicyContext` currently exposes
Detector internals and is not yet a portable public SDK.

## Configuration and lifecycle

### `engine-manifest.v1`

Each registered engine declares an `ENGINE_MANIFEST` with a stable `engine_id`, name,
`MAJOR.MINOR.PATCH` version, classes, input/output contracts, a JSON Schema for
settings, resource requirements, and Python entrypoint. Inspect the built-in
manifest with `python -m detector.main --engine yamnet --show-engine-manifest`.
A third-party engine can declare the same `EngineManifest` on its class. The
manifest is available from `GET /api/engines/manifests`; registered manifest
settings also drive the Settings UI when the engine is selected in the saved
graph. Array and object settings are edited as JSON.

Configuration scope is explicit: legacy `/api/settings` values remain global
defaults; `engine_configs[engine_id].settings` overrides those defaults for one
engine; source-specific DSP/click-removal overrides stay under the source
settings. In graph mode, engine settings are copied and do not mutate global
defaults. The built-in YAMNet manifest exposes its thresholds, inference cadence,
and DSP controls; shared clip and smoothing policy values continue to use the
global settings.

Settings validation uses the `jsonschema` Draft 2020-12 validator by default, or
the draft declared by the manifest's `$schema`. This includes composition,
conditional schemas, local references, tuple validation, and the standard
assertions. Schemas are checked before settings are accepted. Remote references
are not fetched. `format` remains an annotation unless a format checker is
explicitly configured. Unknown graph-level and
engine-entry fields are preserved when the graph is loaded and saved; a manifest
with `additionalProperties: false` can still reject unknown keys inside its
settings object.

An embedding application can register a short selector before startup:

```python
from detector.engine.registry import engine_registry
from detector.main import main

engine_registry.register("my-engine", "mypackage.engine.MyEngine")
main()  # start with --engine my-engine
```

A registered engine must declare `ENGINE_MANIFEST`, and its `engine_id` must
match the selector. List built-in and registered names with
`python -m detector.main --list-engines`.

The current CLI also accepts a class dotted path directly. This preserves the
older `start(runtime, settings, args)` interface; a class without
`ENGINE_MANIFEST` loads with a warning. The registry imports Python code in the
same process, so the package must be installed locally.

A component declares a stable ID, type, version, supported contract versions,
its settings schema, and capabilities. Settings are scoped globally, per
source, per component/stage, and optionally per source-to-chain assignment.
Secrets (such as external service tokens) are stored separately from ordinary
settings and never returned in clear text by the API.

Minimum lifecycle: `configured → starting → ready → stopping → stopped`, with
an `error` state and diagnostics on failure. Components accept resource limits
and timeouts; one component's stop or failure must not silently block the full
graph. Process restart may be required for some component changes, and dynamic
settings application must be declared explicitly.

The current implementation registers the selected engine with its manifest
`engine_id`, `name`, and `version`; state is available from
`GET /api/runtime/status`. Audio subscriptions can be restricted to explicit
`source_ids`, and Runtime Hub selects event policy by `engine_id`. If a third-party engine starts additional background
workers, it can report their state with
`runtime.set_component_state(engine_id, "degraded", error)` and restore `ready`
after recovery. If a child worker has failed, do not report the engine as fully
healthy.

The component registry and graph configuration belong to a management API.
Arbitrary Python imports or uploading user code through the web UI should not
be treated as a safe extension installation mechanism. The MVP can use
developer-managed file/CLI configuration; UI installation and activation can
follow once contracts are stable.

## Versions and compatibility

- Audio, inference-result, and event contracts have independent versions.
- Breaking changes increment the major contract version.
- Unknown optional fields are ignored and preserved by pass-through relays.
- Each adapter declares supported formats and versions.
- Until public `v1` is finalized, these schemas are proposals, not a stable
  wire protocol.

## Current implementation and migration

If there is no engine graph, `detector/main.py` keeps the legacy single-engine
`--engine` mode. When `engine_configs` is set, main starts engines assigned to
routes with separate settings; an explicit `--engine` is rejected. Runtime Hub
applies `default`/`by_source` routes to audio subscriptions. The
`UDPSourceAdapter` publishes `AudioFrame`; the engine subscribes to frames, and
the YAMNet + MLP stage publishes an `InferenceResult` with `source_id`,
`engine_id`, `stage_id`, timestamp, namespaced scores, and metadata back through
the Hub. The current policy receives it through the
Hub, applies smoothing, and publishes legacy score/event messages. The Hub
routes those to `EventBusOutput`, which preserves the old format for
`EventBus` (`detector/bus.py`), WebSocket, and integrations. Buffers and DSP
preprocessing remain in `yamnet_engine.py`; policy lives in
`detector/engine/event_policy.py`, and WAV/sidecar persistence goes through
`WavClipStorage` via Runtime Hub. Policy follows `event-policy.v1`. Engines
load through `detector/engine/registry.py`: `yamnet` is registered by name,
while third-party classes can be registered in code or passed as a dotted path
to `--engine`. A class without a manifest remains available through the legacy
interface with a warning. UDP Hub is a separate network relay. `RuntimeMessage`
wraps the legacy payload and does not yet define the final public `v1` schema.

A runnable minimal third-party engine, launch instructions, and the expected
UI behavior are covered in [Connect a custom engine](custom-engine-example.md).
It receives `AudioFrame` through Runtime Hub and publishes score/event messages
back through the Hub; it does not need direct UDP, WebSocket, or MQTT access.

Recommended migration:

1. Document current score/event and audio wire formats as a legacy profile.
2. [x] Add an internal `audio-frame.v1` and SDR1 input adapter without changing
   the SDR1 network format.
3. [~] Move YAMNet + MLP inference into a stage with `inference-result.v1`;
   extract DSP/buffers and register the full chain as a pipeline provider.
4. [~] Extract the current event policy and clip storage; define a shared policy
   contract and output adapters for events.
5. Add configurable routing and a second independent component, then update
   the management UI and APIs.

This allows each contract to be validated against the existing YAMNet + MLP
before changing transport, model, and interface together.

## Related documents

- [Architecture](architecture.md) — current and target diagrams.
- [HTTP API and WebSocket](api.md) — application interface, not the plugin API.
- [Settings](settings.md) — current runtime parameters.
- `detector/engine/interface.py` — actual legacy engine contract.
