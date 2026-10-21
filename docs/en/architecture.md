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
subscribers through independent bounded queues. The YAMNet engine still owns
buffers, inference, and event smoothing. Its scores and events flow back
through Runtime Hub to the legacy EventBus, which feeds WebSocket and MQTT.

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

The current engine uses a separate YAMNet + MLP inference stage; DSP/TFLite remains a pre-filter. The stage returns `inference-result.v1` through Runtime Hub. `EventPolicy` applies thresholds and smoothing, then sends clip storage requests through Hub to `WavClipStorage`. Legacy score/event messages flow through Hub to `EventBusOutput`. The policy can be selected as `default` or as a local Python class through the CLI. `event-policy.v1` exists, but its context still exposes Detector internals.

The Python package is split into transport, engine, server, and integration
layers connected by an in-process `EventBus`. This does not yet mean every layer
can be replaced through a stable public API: `EngineInterface.start()` receives
Runtime Hub, settings, and CLI args. `main.py` registers the UDP adapter, the
engine subscribes to frames, and results pass back through Hub and
`EventBusOutput` into `EventBus`. Buffers and DSP preprocessing remain in the
engine; inference lives in `detector/engine/stages/yamnet_mlp.py` and event
policy lives in `detector/engine/event_policy.py`.
Legacy message formats are described in code and partly in
[`detector/engine/interface.py`](https://github.com/benchdoos/siren-detector/blob/main/detector/engine/interface.py), but are
not yet a versioned public contract.

When `engine_configs` is non-empty, startup uses the stored `engine-graph.v1`:
each engine assigned to a route starts once with its own settings, and Runtime
Hub applies `default`/`by_source` to its audio subscription. The graph takes
effect after restart. An explicit `--engine` is rejected while a graph is
active. The same Python class cannot be assigned multiple `engine_id` values
because its manifest identity, runtime subscription, and policy keys are unique.
In graph mode YAMNet settings and smoothers are scoped to its entry, and its
event policy receives that entry's settings. Settings UI manages the graph and
shows lifecycle/errors on each engine card; its Runtime panel shows audio
subscriber health. The REST API exposes the same configuration.

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

## DSP pre-filter cascade

The detector can run an optional lightweight DSP model before YAMNet to skip expensive inference when the audio is clearly background.

### How it works

```text
Audio (0.5 s) → DSP feature extraction → Conv1D classifier → Decision
                                                                 │
                                          background (high conf) → skip YAMNet
                                          anything else          → run YAMNet + MLP
```

**Step 1: Feature extraction** (`detector/dsp_features.py`). Eight microphone-invariant features are extracted per 40 ms frame (20 ms hop, 50 fps):

| Feature | What it captures |
|---|---|
| Pitch (YIN) | Fundamental frequency — siren sweeps have characteristic pitch patterns |
| Log RMS energy | Loudness — explosions produce sharp energy spikes |
| Spectral centroid | Brightness of the sound |
| HNR (harmonic-to-noise ratio) | Tonal vs noisy — sirens are tonal, background is noisy |
| ZCR (zero crossing rate) | Frequency content proxy |
| Spectral bandwidth | How spread the energy is across frequencies |
| Spectral flatness | Noise-like vs tone-like spectrum |
| Spectral rolloff | Frequency below which 85% of energy falls |

Delta features (rate of change) are appended, giving 16 features per frame.

**Step 2: Conv1D classifier** (`detector/dsp_infer.py`). A TFLite model processes 250 frames (5 seconds) of normalized features. Architecture: Conv1D(24, k=7) -> Pool -> Conv1D(48, k=5) -> Pool -> Conv1D(48, k=3) -> GlobalAvgPool -> Dense(48) -> Dense(4). Total: ~18K parameters, ~30 KB as INT8 TFLite.

**Step 3: Gate decision**. If the DSP model predicts "background" with confidence above `dsp_bg_threshold` (default 0.92), YAMNet inference is skipped for that window. A hysteresis mechanism prevents rapid toggling:

- `dsp_bg_confirm_windows` (default 2): consecutive background predictions needed before skipping starts.
- `dsp_bg_release_threshold` (default 0.90): confidence must drop below this to resume YAMNet.
- `dsp_event_gate` (default 0.30): if any event class probability exceeds this, YAMNet runs regardless.
- `dsp_max_consecutive_skips` (default 20): safety valve — after N consecutive skips, one forced YAMNet pass prevents the DSP model from permanently blocking detection.

### Why DSP features are microphone-invariant

Unlike spectrogram-based approaches that depend on the microphone's frequency response, DSP features like pitch, HNR, and ZCR describe structural properties of the sound. Changing the microphone or location shifts absolute levels but preserves the patterns that distinguish sirens (periodic sweep), explosions (sharp energy transient), and UAVs (sustained tonal energy) from background noise.

### Configuration

DSP settings are in [Settings](settings.md#dsp-pre-filter). Per-source overrides allow different thresholds per microphone. The DSP model must be present in the model directory; when missing, the pre-filter is automatically disabled.

## Inference modes

The detector supports `round_robin` inference (one inference loop cycles over active sources) and `parallel` inference (a loop per source). Parallel mode can improve per-source latency at the cost of CPU and memory as source count grows. It can be selected in detector settings. Neither mode runs inference on the ESP32. See [inference modes and settings](settings.md#processing-modes) for details.
