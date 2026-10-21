# Architecture

## Target replaceable-component architecture

```text
Audio sources → input adapters → Runtime Hub ──audio──→ processing chain
                                       ↑                 DSP → YAMNet → MLP
                                       └──── scores / events ────┘
                                       │
                                       ├→ score export / monitoring
                                       └→ event policy → export layer
                                                            ├→ Web UI / WS
                                                            ├→ REST API
                                                            ├→ MQTT and other outputs
                                                            └→ clips / storage
```

This is the target architecture, not a claim that every component is already
replaceable. The principle is that each block has a contract, so a source,
router, processing stage, or output adapter can be changed independently. For
example, a user should be able to replace the MLP, the full recognition chain,
or event delivery without rewriting microphone support or the UI.

“Hub” currently means the UDP fan-out relay in `hub/hub.py`, which forwards
packets to the detector and recorder. In the target design, Runtime Hub is the
logical router inside Detector: it routes audio through chains, receives
processing results, and forwards them to event policy and the export layer.
These are different roles and should not be conflated in component contracts
or documentation.

## Current audio and event flow

```text
INMP441 → ESP32-S3 → UDP / SDR1 → UDP Hub → Detector → Web UI / MQTT
                                             └──────→ event clips
```

The current UDP Hub is a small network relay. `UDPSourceAdapter` in Detector
decodes SDR1 and emits `AudioFrame`; Runtime Hub routes the frame to
subscribers. The YAMNet engine subscribes to frames but still owns buffers,
inference, and event smoothing. Its scores and events flow back through Runtime
Hub to the legacy EventBus, which feeds WebSocket and MQTT.

## Components

| Component | Responsibility | Code/configuration |
|---|---|---|
| Audio node | Capture 16 kHz audio and send SDR1 datagrams | `esp32/siren-detector.yaml` |
| UDP Hub | Forward each datagram to configured destinations | `hub/hub.py`, `hub/docker-compose.yml` |
| Runtime Hub | Route normalized audio frames and processing results | `detector/runtime/` |
| Detector | Own the UDP input adapter, buffers, processing, UI/API, and integrations | `detector/` |
| Recorder | Record incoming audio to labeled WAV files on demand | `recorder/` |
| ML tools | Prepare data, train classifier heads, and evaluate models | `dataset/`, `ml/` |

## Detector pipeline

The current engine uses a separate YAMNet + MLP inference stage; DSP/TFLite remains a pre-filter. The stage returns `inference-result.v1` through Runtime Hub. A separate `EventPolicy` applies thresholds and smoothing, manages clips, and publishes legacy score/event messages back through Hub to `EventBusOutput`. The policy is separated from the engine but is still a project implementation, not a pluggable API.

The Python package is split into transport, engine, server, and integration
layers connected by an in-process `EventBus`. This does not yet mean every layer
can be replaced through a stable public API: `EngineInterface.start()` receives
Runtime Hub, settings, and CLI args. `main.py` registers the UDP adapter, the
engine subscribes to frames, and results pass back through Hub and
`EventBusOutput` into `EventBus`. Buffers and DSP preprocessing remain in the
engine; inference lives in `detector/engine/stages/yamnet_mlp.py` and event
policy lives in `detector/engine/event_policy.py`.
Legacy message formats are described in code and partly in
[`detector/engine/interface.py`](../../detector/engine/interface.py), but are
not yet a versioned public contract.

The target boundaries and formats are described in the [component API](component-api.md).
The application HTTP and WebSocket interfaces are covered separately in the
[API reference](api.md); current runtime options are in the [settings reference](settings.md).

## Audio protocol

SDR1 has a 12-byte network-order header followed by PCM payload:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 bytes | Magic `0x53445231` (`SDR1`) |
| 4 | 4 bytes | Packet sequence number |
| 8 | 2 bytes | Payload length in bytes |
| 10 | 2 bytes | Source ID |
| 12 | variable | PCM payload |

The payload uses 16 kHz mono I2S audio in 32-bit slots. The ESP32 transport keeps each microphone's source ID in the header; the receiver separates sources into independent buffers and smoothers.

## Multiple microphones: current behavior

The ESP32 firmware currently captures two optional INMP441 microphones on separate I2S buses, then averages the samples and sends one SDR1 stream. They therefore appear as **one source** to the detector. A USB microphone running `audio_source/audio_source.py` is an independent source. See [Multi-microphone setup](multi-microphone.md) for details.

## Inference modes

The detector supports `round_robin` inference (one inference loop cycles over active sources) and `parallel` inference (a loop per source). Parallel mode can improve per-source latency at the cost of CPU and memory as source count grows. It can be selected in detector settings. Neither mode runs inference on the ESP32. See [inference modes and settings](settings.md#processing-modes) for details.
