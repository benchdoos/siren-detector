# Architecture

## Audio and event flow

```text
INMP441 → ESP32-S3 → UDP / SDR1 → Hub → Detector → Web UI / MQTT
                                          └──────→ event clips
```

The Hub is a small UDP fan-out relay. It forwards incoming packets to the detector and to the recorder when the recorder is running. The detector owns the inference, event decision, clip storage, web server, and integrations.

## Components

| Component | Responsibility | Code/configuration |
|---|---|---|
| Audio node | Capture 16 kHz audio and send SDR1 datagrams | `esp32/siren-detector.yaml` |
| Hub | Forward each datagram to configured destinations | `hub/hub.py`, `hub/docker-compose.yml` |
| Detector | Receive audio, maintain per-source state, run inference, publish events | `detector/` |
| Recorder | Record incoming audio to labeled WAV files on demand | `recorder/` |
| ML tools | Prepare data, train classifier heads, and evaluate models | `dataset/`, `ml/` |

## Detector pipeline

The current engine loads YAMNet once and uses its audio embeddings with an optional project-trained MLP classifier. An optional DSP/TFLite model can pre-filter background windows to reduce YAMNet work. The event smoother combines repeated window scores before confirming an event. Thresholds and timing settings can be changed from the web UI.

The Python detector is split into transport, engine, server, and integration layers, connected by an in-process `EventBus`. Project modules can be replaced with custom implementations as long as they preserve the relevant API/contracts: an inference engine implements `detector/engine/interface.py`, while an audio source sends SDR1 packets expected by the transport. See the [settings reference](settings.md) for runtime options and the [API reference](api.md) for HTTP and WebSocket interfaces.

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
