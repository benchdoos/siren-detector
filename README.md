<h1 align="center">
  <a href="https://github.com/benchdoos/siren-detector"><img src=".github/images/siren-detector-logo.png" alt="Siren Detector" width="200"></a>
  <br>
  Siren Detector
</h1>

<p align="right">
  <b>English</b> •
  <a href="README.ru.md">Русский</a>
</p>

Open-source detector of dangerous sound events based on ESP32 + Raspberry Pi. It recognizes civil alert sirens, explosions, and approaching UAVs — sounds that it is important to react to in time.

---

> [!WARNING]
> **This application and all its components are intended exclusively for peaceful purposes. Any use of this software for military, harmful, or destructive purposes is strictly prohibited.**

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to copy, share, and adapt it for **non-commercial** purposes, provided you give appropriate credit and link to the [original repository](https://github.com/benchdoos/siren-detector). The software is provided **"as is"**, without warranty of any kind.

---

## 1. Project Concept

### Idea

An autonomous detector of sounds that represent danger or require attention. Not a universal classifier of every sound in the world, but a system with a specific task — to warn when something happens nearby that is worth paying attention to.

Target classes:

- 🚨 `siren` — civil alert sirens (air raid);
- 💥 `explosion` — explosions and sharp impulse sounds;
- 🛵 `motorcycle` — UAVs;
- 🔇 `background` / `unknown` — normal background and low-confidence classifications.

Key principle:

> **ESP32 is a universal Audio Node, not a mandatory ML engine.**

The same Audio Node should be able to work:

1. fully autonomously with a local TinyML model;
2. with Raspberry Pi;
3. with a regular Linux/PC;
4. with another server or edge computer;
5. in the future — with a cloud or specialized inference engine.

### Target Architecture

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

## Quick Start

Minimum requirements: ESP32-S3 + INMP441 microphone + any Linux/Mac host.

1. **Set up the environment**:
   ```bash
   cp .env.example .env                         # root .env for Makefile
   cp hub/.env.example hub/.env                  # .env for Docker Compose
   cp esp32/secrets.yaml.example esp32/secrets.yaml  # ESP32 secrets
   ```
   Fill in IP addresses, ports, and passwords in each file.

2. **Flash ESP32**: copy `esp32/siren-detector.yaml` and `secrets.yaml`
   to the ESPHome host and flash. The microphone connects via I2S (GPIO4/5/6,
   see section 5).

3. **Start Hub** on the host (receives UDP stream from ESP32):
   ```bash
   cd hub
   docker compose up -d hub
   ```

4. **Start Detector** (YAMNet + MLP):
   ```bash
   docker compose up -d detector
   ```

5. **Open Web UI**: `http://<your_host_ip>:8080` — real-time confidence
   bar, event history, clip playback.

6. **Retrain the model** on your data:
   ```bash
   make sync-dataset   # sync labeled WAVs
   make train          # train MLP classifier
   make deploy-model   # deploy weights + restart
   ```

For standalone ESP32 operation without a server — see Phase 9 (in development).

---

## 2. Architectural Principles

### Separation of Concerns

**Audio Node**
- captures sound;
- buffers PCM;
- transmits audio;
- optionally performs local inference;
- can provide local voice/sound alerts.

**Detection Engine**
- extracts features;
- runs the ML model;
- evaluates confidence;
- combines multiple time windows;
- makes event decisions.

**Integration**
- Home Assistant;
- MQTT;
- REST;
- WebSocket;
- webhook;
- other event consumers.

None of these components should be tightly coupled to a specific hardware vendor.

### Modular Detector Architecture

The Detection Engine is implemented as a Python package `detector/` with atomic layers
connected via an in-process **EventBus** (`detector/bus.py`):

```
detector/
├── main.py               # entry point: wires bus, engine, server, integrations
├── config.py             # constants, detector definitions, default settings
├── bus.py                # EventBus — the single shared object between layers
├── transport/udp.py      # UDP receiver (SDR1 protocol)
├── engine/
│   ├── interface.py      # EngineInterface Protocol — contract for engine replacement
│   ├── smoother.py       # TemporalSmoother (idle → possible → confirmed)
│   └── yamnet_engine.py  # YAMNet + MLP (current implementation)
├── server/               # FastAPI: WebSocket, clips, feedback, settings
└── integrations/mqtt.py  # MQTT, Home Assistant discovery
```

To plug in your own engine, implement a class with a
`start(bus, settings, args)` method per the `detector/engine/interface.py` contract
and wire it in `main.py`.

### Interchangeable Engine

ESP32 should not know which specific neural network runs on the server.

Possible engines:

- YAMNet + MLP (current, `detector/engine/yamnet_engine.py`);
- custom TensorFlow/TFLite model;
- LiteRT/TensorFlow Lite Micro;
- ONNX Runtime;
- PyTorch-based inference;
- Edge Impulse;
- specialized server inference engine.

### ESP32 Operating Modes

ESP32 supports three operating modes (configurable via Home Assistant):

**Standalone** — ESP32 detects sounds on its own and alerts via speaker:

```text
INMP441 → ESP32-S3 → mel spectrogram → compact CNN (TFLite Micro)
   → event → speaker / local alert
```

**Combined** — ESP32 performs initial screening, sends audio to server on suspicion:

```text
INMP441 → ESP32-S3 → local model → suspicion?
   → yes: send audio to RPi → YAMNet + MLP → confirmation
   → no: silence, no traffic
```

**Remote** (current mode) — entire stream goes to server:

```text
INMP441 → ESP32-S3 → UDP → RPi → YAMNet + MLP → event
```

The external engine always remains optional.

---

# 3. Technology Stack

## Hardware

### Audio Node

- ESP32-S3 DevKitC-1 N16R8
- 8 MB PSRAM
- Wi-Fi
- INMP441 I2S MEMS microphone
- in the future — a small speaker/amplifier for local voice alerts

### Server / Edge

- Raspberry Pi 5
- Linux
- Docker / Docker Compose

## Firmware / Device Layer

- ESPHome 2026.9.1 — current development stage
- ESP-IDF
- I2S audio
- mDNS
- ESPHome API
- OTA

ESPHome is currently used for quick hardware bring-up and device management. For a universal Audio Node, the audio transport may later be extracted into a separate firmware/application layer.

## ML / Audio

First baseline:

- YAMNet
- AudioSet

Future directions:

- custom compact CNN;
- MFCC / log-mel spectrogram — feature extraction from raw PCM;
- INT8 quantization — model compression for microcontrollers;
- TensorFlow Lite Micro (aka LiteRT Micro) — framework for running models on ESP32;
- ESP-NN — Espressif library that accelerates neural network operations on ESP32-S3 via SIMD instructions (automatically integrated via TFLite Micro);
- Edge Impulse — alternative TinyML platform.

## Transport

First planned transport:

- WebSocket
- PCM audio stream

Additional options:

- UDP;
- TCP;
- MQTT for events/commands;
- Serial/USB;
- other protocols.

## Integrations

Primary target consumer:

- Home Assistant

Additionally:

- MQTT;
- REST API;
- WebSocket;
- webhook;
- Node-RED;
- custom applications.

---

# 4. Current Setup

## Network

Addresses depend on your specific installation. Example:

| Device | Address |
|---|---|
| Router | `<your_router_ip>` |
| Raspberry Pi 5 | `<your_rpi_ip>` |
| ESP32-S3 | `<your_esp32_ip>` |
| ESP32 hostname | `siren-detector.local` |
| Wi-Fi SSID | `<your_ssid>` |

Actual addresses, passwords, and SSID are set via `.env` files and `secrets.yaml`
(not committed to Git). See `.env.example` in each component.

## Raspberry Pi

ESPHome project:

```text
~/Projects/esphome/
├── docker-compose.yml
├── .env
└── esphome/
    ├── secrets.yaml
    └── siren-detector.yaml
```

Current Docker Compose:

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

## Current ESPHome Configuration

Basic capabilities:

- ESP32-S3;
- ESP-IDF;
- Wi-Fi;
- ESPHome API;
- OTA;
- captive portal;
- Wi-Fi signal sensor;
- uptime sensor;
- status/online sensor.

### First Audio Test: INMP441

I2S capture without ML has been added to `esp32/siren-detector.yaml`:

| INMP441 | ESP32-S3 |
|---|---|
| `VDD` | `3V3` |
| `GND` | `GND` |
| `SCK` / `BCLK` | `GPIO4` |
| `WS` / `LRCLK` | `GPIO5` |
| `SD` / `DOUT` | `GPIO6` |
| `L/R` | `GND` (left channel) |

Capture parameters: mono, `16 kHz`, microphone data in `32-bit I2S` slots. The recorder converts the stream to WAV `16-bit PCM`.

For verification, a `Microphone capture` toggle appeared in Home Assistant:

- enable — start reading I2S;
- disable — stop reading;
- 5 seconds after ESP32 boot, capture starts automatically.

During active capture, ESP32 logs block size and signal levels (`rms` and `peak`) roughly every half second. This is a check for the presence of a real audio signal, not an event detector.

Before the first flash, validate the config on Raspberry Pi:

```bash
cd ~/Projects/esphome
docker compose run --rm esphome config /config/siren-detector.yaml
```

Then flash via OTA:

```bash
docker compose run --rm esphome upload /config/siren-detector.yaml --device siren-detector.local
docker compose run --rm esphome logs /config/siren-detector.yaml --device siren-detector.local
```

After boot, `Microphone capture` enables automatically after 5 seconds. In Home Assistant you can disable it manually; after the next reboot, capture will start again.

Expected log line format:

```text
INMP441 data: 1024 bytes, chunks=32, samples=256, rms=0.00..., peak=0.00...
```

## UDP Recorder: First Working Prototype

ESP32 sends I2S packets over UDP to Raspberry Pi 5 (`<your_rpi_ip>:9000`). Capture starts on device boot. Each packet gets a sequence number: if UDP loses a packet, the recorder inserts silence in its place rather than shortening the WAV. The recorder saves the stream as WAV `16 kHz / mono / 16 bit` and requires no third-party Python libraries.

The recorder is in a separate `recorder/` directory:

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

Copy the directory to Raspberry Pi:

```bash
scp -r recorder <user>@<your_rpi_ip>:~/Projects/esphome/
ssh <user>@<your_rpi_ip>
cd ~/Projects/esphome/recorder
mkdir -p dataset/{siren,explosion,motorcycle,background,unknown,inbox}
docker compose build
LABEL=background DURATION=10 docker compose run --rm recorder
```

After changes to the UDP packet format, you need to rebuild and reflash ESP32, then rebuild the recorder:

```bash
cd ~/Projects/esphome/recorder
docker compose build --no-cache
```

`LABEL` selects the subdirectory, and `DURATION` sets the recording length in seconds. For example, the command above creates a 10-second WAV in `dataset/background/` and finishes automatically. Allowed labels: `siren`, `explosion`, `motorcycle`, `background`, `unknown`, `inbox`. If `DURATION` is not set or equals `0`, recording continues until `Ctrl-C`.

Examples:

```bash
LABEL=siren DURATION=10 docker compose run --rm recorder
LABEL=background DURATION=30 docker compose run --rm recorder
docker compose run --rm recorder
```

The last command saves a continuous recording to `dataset/inbox/`.

### Annotating a Long Recording After an Event

To avoid guessing the class in advance, the recorder can be started in `inbox` without
a duration limit. After stopping, it saves a WAV and a JSON file with
the start time. Transfer both files to Mac, then annotate intervals by clock time:

```bash
python3 dataset/annotate_recording.py dataset/inbox/<recording>.wav \
  --timezone Europe/Moscow \
  --segment siren 12:03 12:08 \
  --segment background 12:08 12:15
```

Intervals will be copied to the corresponding dataset directories, the original WAV
remains untouched in `inbox`, and the segment-to-source mapping is saved in
`dataset/annotations.csv`. Full instructions are in [`dataset/README.md`](dataset/README.md).

To specify time relative to the file start, add `-f` after `--segment`:
`--segment -f siren 12:03 12:08` — these are minutes and seconds (`MM:SS`); for recordings
longer than an hour, use `HH:MM:SS`, e.g. `1:12:03`.

## Pre-made Sounds for Quick Start

To quickly test the classifier, you can use the external dataset
[ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 five-second recordings,
including `siren`, `rain`, `wind`, `washing_machine`, `keyboard_typing`,
`engine`, and `fireworks`. It has no separate classes for "dishwasher" or "3D printer".

Important: ESC-50 is distributed under **CC BY-NC 3.0** (non-commercial
use); the ESC-10 subset is CC BY 3.0. The archive includes
attribution of original clip authors. Before commercial use, you need to
check the rights for each item. Therefore, audio is not stored in Git and is not
mixed with our own recordings.

Data structure:

```text
dataset/
├── siren/                 # own recordings from recorder
├── explosion/
├── motorcycle/
├── background/
├── unknown/
├── inbox/
└── external/
    └── esc50/             # downloaded separately; do not commit audio
        ├── audio/
        ├── meta/esc50.csv
        ├── LICENSE
        └── SOURCE.md
```

From the repository root, run:

```bash
python3 dataset/fetch_esc50.py
```

The script downloads the official archive and extracts only selected categories
(approximately 280 five-second clips), preserving original filenames, the full CSV
of metadata, and the LICENSE. The archive is temporary; the WAV files remain locally in
`dataset/external/esc50/audio/` and won't enter Git. Do not rename files:
the CSV links them to class, fold, and source. Mappings for sanity check:

| ESC-50 class | Use in validation | Our target label |
|---|---|---|
| `siren` | siren | `siren` |
| `rain`, `wind`, `washing_machine`, `keyboard_typing` | diverse background | `background` |
| `engine` | engine → moped/motorcycle | `motorcycle` |
| `fireworks`, `thunderstorm`, `helicopter`, etc. | background sounds | `background` |

Do not copy external clips into `dataset/siren/` or `dataset/background/`:
this would hide provenance and could cause data leakage of identical source recordings
between training and validation. When comparing, split data by `fold` from CSV;
the ESC-50 result is only a quick sanity check, not a quality assessment on
the INMP441 microphone. For real quality metrics, you need your own recordings
from the device. ESC-50 may have processing artifacts that affect
training and metrics.

### Adding Data from Freesound and Other Sources

For classes not present in ESC-50 (e.g. `explosion`), you can download clips
from [Freesound.org](https://freesound.org/) or other open sources.

Downloaded files go to `dataset/inbox/downloads/`, then are converted
by the `convert_downloads.py` script:

```bash
python3 dataset/convert_downloads.py explosion
python3 dataset/convert_downloads.py motorcycle
```

The script automatically converts to 16 kHz / mono / 16-bit WAV, splits into
5-second chunks, and places them in `dataset/<label>/`. Processed originals
are moved to `dataset/inbox/downloads/done/`.

License and dataset card:
[ESC-50 README and license](https://github.com/karolpiczak/ESC-50).

### Data Augmentation

The `dataset/augment.py` script creates realistic training data from clean
external sources (ESC-50, Freesound) by applying:

- volume variation (-20…+3 dB — simulating different distances);
- mixing with real INMP441 background recordings (street, room);
- optionally: microphone noise profile (from a silence recording with INMP441).

```bash
# Augment siren and explosion
make augment

# Or individually
make augment-siren
make augment-explosion

# With noise profile (after recording silence with INMP441)
ml/.venv/bin/python dataset/augment.py siren --count 200 \
  --noise-profile dataset/background/noise-profile-inmp441.wav

# Preview what will be created
ml/.venv/bin/python dataset/augment.py siren --count 10 --dry-run
```

Augmented files are saved as `dataset/<label>/aug_*.wav` and participate
in training alongside other recordings. After augmentation — `make retrain`.

Detailed recommendations on aug/real data ratios, class imbalance, domain gap,
and a pre-retrain checklist are in [`ml/TRAINING_GUIDE.md`](ml/TRAINING_GUIDE.md).

### First Local Baseline: YAMNet

An isolated environment and script [`ml/evaluate_yamnet.py`](ml/evaluate_yamnet.py) are prepared on Mac.
Launch instructions are in [`ml/README.md`](ml/README.md). The script converts WAVs to
mono/16 kHz on the fly and tests YAMNet on selected clips by original fold;
source audio files are not modified, results are saved locally in
`ml/results/` and don't enter Git.

The first run on 280 clips gave ROC AUC `0.996` for ranking siren against
other selected classes; exact top-1 match for class `siren` — `60%`.
This is only a sanity check of the pre-trained model on an external dataset, not an assessment of
INMP441 performance in a room or outdoors. YAMNet does not have a separate
`washing_machine` class.

Before flashing the UDP transport version to Raspberry Pi:

```bash
cd ~/Projects/esphome
docker compose run --rm esphome config /config/siren-detector.yaml
docker compose run --rm esphome compile /config/siren-detector.yaml
docker compose run --rm esphome upload /config/siren-detector.yaml --device siren-detector.local
```

The device successfully connects to ESPHome API:

```text
siren-detector.local:6053
```

OTA:

```text
siren-detector.local:3232
```

The Wi-Fi signal level at the time of testing was approximately `-35 dB`.

## Home Assistant

The ESPHome device integration is already connected.

Current entities include:

- Wi-Fi Signal;
- Uptime;
- Online.

---

# 5. INMP441

Wiring:

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| SCK / BCLK | GPIO4 |
| WS / LRCLK | GPIO5 |
| SD / DOUT | GPIO6 |
| L/R | GND |

Target audio format for the first prototype:

```text
PCM
16 kHz
mono
16 bit
```

The first stage is not recognition, but verifying capture quality.

---

# 6. Implementation Plan

## Phase 0 — Bring-up

**Status: completed**

- [x] ESP32-S3 identified;
- [x] flashed;
- [x] connected to Wi-Fi;
- [x] accessible via mDNS;
- [x] ESPHome API works;
- [x] OTA prepared;
- [x] Home Assistant connected;
- [x] INMP441 connected;
- [x] stable I2S stream obtained;
- [x] audio transmitted via UDP to Raspberry Pi;
- [x] WAV `16 kHz / mono / 16 bit` saved and listened to.

## Phase 1 — Audio Capture

**Status: basic capture and playback confirmed; quality testing continues.**

Goal:

```text
INMP441
   ↓
ESP32-S3
   ↓
I2S
   ↓
PCM 16 kHz / mono / 16 bit
```

Verified on the bench: recording `20261008-130016.wav`, duration `16.864 s`, `dropped=0`. Next: check noise floor and collect labeled recordings under various conditions.

Additional checks:

- signal level;
- self-noise;
- clipping;
- I2S stability;
- recording quality;
- effect of device position;
- operation through window/balcony.

## Phase 2 — Audio Node

**Status: ahead.**

Make ESP32 independent of a specific backend.

Functions:

- audio capture;
- ring buffer;
- chunking;
- timestamp;
- transport;
- heartbeat;
- configuration;
- optional local inference.

## Phase 3 — Audio Transport

**Status: UDP prototype works; Hub implemented; WebSocket — next step.**

The current prototype uses UDP with sequence numbers. A **Hub** has been implemented — a UDP fan-out relay on Raspberry Pi that receives the stream from ESP32 and forwards it to all subscribers:

```text
ESP32 ──UDP:9000──> Hub (RPi)
                       ├──> Recorder  (:9001)  → WAV files
                       ├──> Detector  (:9002)  → YAMNet events
                       └──> Future consumer    → any engine
```

This allows multiple consumers to work with a single stream without changes to the ESP32 firmware. Each consumer is independent and can be any implementation.

The next transport version — WebSocket with a full protocol:

```text
ESP32 ── WebSocket ──> Raspberry Pi / any host
```

Preliminary message types:

```text
HELLO
CONFIG
AUDIO
PING
PONG
EVENT
MODEL_INFO
```

The protocol should remain independent of a specific ML model. In the future, a consumer will be able to connect to ESP32 directly and indicate that it takes over audio processing — ESP32 then continues or stops local analysis by agreement.

## Phase 4 — Recording / Dataset

**Status: recorder can save own WAVs by category; external ESC-50 can be added separately for sanity check.**

Raspberry Pi should be able to save real recordings:

```text
dataset/
├── siren/
├── explosion/
├── motorcycle/
├── background/
├── unknown/
├── inbox/
└── external/esc50/  # third-party dataset with separate license and metadata
```

Special attention:

- real household noises;
- wind;
- rain;
- speech;
- TV;
- cars;
- regular engines;
- music;
- birds/animals;
- random claps.

A large `background` dataset is essential for reducing false positives.

## Phase 5 — Baseline Engine

**Status: YAMNet + MLP classifier deployed on Raspberry Pi, running on live stream.**

The first baseline runs on Mac; see [`ml/README.md`](ml/README.md). First run result:
siren-vs-selected-others ROC AUC `0.996`, top-1 `siren` `60%`.
Test on own WAVs before connecting to the live stream.

First backend:

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

Goal — obtain an objective baseline before training a custom model.

Checking:

- siren;
- explosion;
- motorcycle;
- false positives;
- false negatives;
- latency;
- noise robustness.

## Phase 6 — Event Engine

**Status: implemented and running.**

Raw predictions should not immediately become alerts.

For example:

```text
window 1 → siren 0.71
window 2 → siren 0.83
window 3 → siren 0.91
window 4 → siren 0.94
             ↓
        CONFIRMED EVENT
```

Required:

- confidence threshold;
- temporal smoothing;
- debounce;
- minimum event duration;
- cooldown;
- `unknown/background`.

## Phase 7 — Integrations

First variant:

```text
Detector Engine
      ↓
MQTT
      ↓
Home Assistant
```

The event must be standardized.

Example:

```json
{
  "event": "siren",
  "confidence": 0.94,
  "timestamp": "2026-10-07T20:00:00Z",
  "source": "siren-detector-01",
  "engine": "yamnet"
}
```

## Phase 8 — Custom Model

Architecture: YAMNet as a frozen feature extractor + a small classifier on top of embeddings.

```text
audio (16 kHz)
   ↓
YAMNet (frozen) → embeddings 1024-dim / window
   ↓
small classifier (2–3 dense layers, ~50–100 KB)
   ↓
siren / explosion / motorcycle / background / unknown
```

YAMNet is already trained on AudioSet and knows how to extract features. We only need to teach the head to recognize **our** classes on **our** recordings. This trains on Mac in minutes and runs on RPi without issues.

After obtaining a sufficient dataset:

```text
dataset (labeled WAVs)
   ↓
extract YAMNet embeddings
   ↓
train classifier head
   ↓
evaluate on test set
   ↓
export to TFLite
   ↓
deploy to RPi
```

Target classes:

```text
siren
explosion
motorcycle
background
unknown
```

## Phase 9 — Local Inference on ESP32

Goal: run ML directly on ESP32-S3 so the device can operate autonomously.

**TinyML** is the general name for the concept (ML on microcontrollers). For implementation we use a specific stack:

```text
INMP441 → PCM 16 kHz
   ↓
log-mel spectrogram (feature extraction, pure math)
   ↓
compact CNN (~50-200 KB, INT8 quantization)
   ↓
TensorFlow Lite Micro (framework, runs the model)
   + ESP-NN (computation acceleration via ESP32-S3 SIMD instructions)
   ↓
siren / explosion / motorcycle / background → event
   ↓
speaker (local alert) + optionally → RPi for confirmation
```

YAMNet (~14 MB) won't fit on ESP32. Instead, we train a compact CNN on mel spectrograms, optionally using knowledge distillation from the large model.

Target characteristics:
- 60-80% of server model accuracy — sufficient for initial screening;
- inference ~50-200 ms per window — real-time;
- three operating modes: standalone, combined, remote (see section 2).

The server Detection Engine always remains available as an alternative or for confirmation.

---

# 7. Recognition Safety and Quality

This project deals with potentially dangerous events, so the main metric is not just accuracy.

Especially important:

- false negative rate;
- false positive rate;
- latency;
- confidence;
- noise robustness;
- quality on real recordings;
- stability with multiple audio sources.

### Don't

Don't turn a single weak prediction into a categorical message:

```text
"EXPLOSION!!!"
```

Better to have states:

```text
possible
probable
confirmed
```

and consider multiple consecutive windows.

### Unknown — Mandatory Class

The model should not be forced to choose one of the three dangerous classes for any sound.

```text
siren
explosion
motorcycle
background
unknown
```

---

# 8. Why We Don't Tie to Raspberry Pi

Raspberry Pi is a convenient and cheap inference host for home installations, but not an architectural requirement.

Possible setups:

```text
ESP32 → Raspberry Pi
ESP32 → PC
ESP32 → NAS
ESP32 → mini-PC
ESP32 → server
ESP32 → cloud
ESP32 → local TinyML
```

This way the user can choose the complexity level:

### Minimum (standalone mode)

```text
ESP32 + microphone + speaker → local CNN → alert
```

### Home (combined mode)

```text
ESP32 + Raspberry Pi → local screening + server confirmation
```

### Advanced

```text
ESP32 + powerful PC/server + custom model
```

### Remote (current mode)

```text
ESP32 → entire stream to RPi/server → YAMNet + MLP
```

---

# 9. Non-goals

The project should not:

- be tied to Home Assistant;
- require Raspberry Pi;
- require cloud;
- require a specific ML model;
- require ESPHome as the final firmware;
- be limited to sirens only;
- attempt to classify absolutely any sound.

---

# 10. Current Status

**Project:** `Sound Detector`

**Phase:** Live Detection + Feedback Collection.

Already working:

```text
INMP441
   ↓
ESP32-S3 (I2S, UDP transport)
   ↓
Hub (fan-out relay, RPi :9000)
   ├──> Recorder (:9001) → WAV dataset (on demand)
   └──> Detector (:9002) → YAMNet + MLP → events → Web UI
```

**Detector** (`http://<your_rpi_ip>:8080`):
- YAMNet (frozen, TF Hub) extracts 1024-dim embeddings every 0.5 s
- MLP classifier (`siren` / `background` / `explosion` / `motorcycle`) on top of embeddings
- Temporal smoother: 4 triggers → possible, 8 → confirmed (window 30 s, cooldown 60 s)
- On confirmed: WAV clip saved to `/clips/` (two recording modes, see below)
- WebSocket broadcast: score every 0.5 s + events
- **Parallel mode**: one inference thread per audio source (configurable in Settings, requires container restart)

**Web UI**:
- Real-time confidence bar with pulse animation on possible/confirmed
- Per-source score display for multiple microphones
- Event list with audio player (confirmed events with clips only)
- Labeling buttons: clicking copies the clip to `/dataset/{label}/` for retraining
- Segment labeling: drag on spectrogram → select range + label → `POST /feedback/segments` slices WAV and copies to dataset
- Filters by type, history of 100 events
- Dedicated **Settings page**: thresholds, detection mode, microphone management (enable/disable individual sources, custom display names)
- **Background calibration**: record ambient audio from a specific source → saves to `/dataset/background/{source_name}/` → `make retrain`. Recordings list with spectrogram player and delete button in Settings → Calibration
- Admin auth: viewing/listening — no token required; labeling, settings — require `ADMIN_TOKEN`
- Mobile-friendly: audio listening and layout work on mobile devices

**Clip recording modes** (`Clip from possible` setting in Settings):

- **Off** (default) — clip contains the last N seconds before the `confirmed` moment. Suitable for general monitoring.
- **On** — clip starts from the `possible` moment (+ 2 s pre-buffer), capturing the sound wave that triggered detection. Per-window scores are saved in the clip's JSON metadata for precise labeling during retraining. Recommended for collecting quality training data.

**MQTT / Home Assistant**:
- Mosquitto broker — separate Docker Compose (`mosquitto/`)
- Detector publishes HA MQTT Discovery on broker connection
- In HA, a **Sound Detector** device automatically appears with sensors:
  - `binary_sensor.sound_siren` / `sound_explosion` / `sound_motorcycle` — ON on confirmed, OFF after cooldown
  - `sensor.sound_siren_confidence` / `sound_explosion_confidence` / `sound_motorcycle_confidence`
- MQTT is optional: without `MQTT_HOST` the detector works as before

Next step: accumulate labeled examples via UI → retrain model → reduce false positives.

---

# 11. Makefile

Common operations have a `Makefile` — an alternative to the manual commands in the sections above.

```bash
make help            # Show all available commands
```

### Deploy

```bash
make deploy          # Full rebuild: copy detector/ + rebuild on RPi
make deploy-ui       # Frontend only (no rebuild)
make deploy-compose  # Send docker-compose.yml to RPi
make restart         # Restart container
make logs            # Container logs
```

### ML / Retraining

```bash
make sync-dataset    # Sync labeled WAVs from RPi to Mac
make train           # Train MLP classifier
make deploy-model    # Send weights to RPi + restart
make retrain         # Full cycle: sync → train → deploy → clean
# If calibration is recording, retrain waits for it to finish and saves the file
# Background calibration: Web UI → Calibrate → select source → Start Recording
# File is saved to dataset/background/{source_name}/ → make retrain
make clean-clips     # Delete saved clips on RPi
make clean-dataset   # Delete labeled WAVs on RPi (sync first!)
make clean-docker    # Clean Docker cache on RPi (volumes untouched)
```

### Other

```bash
make fetch-esc50     # Download ESC-50 dataset
make open            # Open Web UI in browser
make unlock          # Unlock SSH key for 10 minutes
```

## MQTT / Home Assistant

MQTT Discovery is used for Home Assistant integration. Mosquitto runs as a separate Docker Compose.

### First Mosquitto Launch on RPi

```bash
cd ~/Projects/esphome/mosquitto
cp .env.example .env
# edit .env: set MQTT_USER and MQTT_PASSWORD
sh create_passwd.sh
docker compose up -d
```

### Detector Configuration

Set in `hub/.env` (or environment variables):

```bash
MQTT_HOST=127.0.0.1
MQTT_PORT=1883
MQTT_USER=siren
MQTT_PASSWORD=<your_password>
ADMIN_TOKEN=<your_admin_token>
```

### Web UI Admin Auth

The Web UI is split into public and admin sections:

- **Without token**: viewing the score bar, listening to clips, viewing metrics — everything works.
- **With token**: labeling clips (label buttons), changing detector settings.

If `ADMIN_TOKEN` is not set in environment variables, a token is auto-generated at startup and printed to the log:

```bash
docker compose logs detector | grep ADMIN_TOKEN
```

In the Web UI, click the **Admin** button in the header and enter the token. The token is stored in the browser's `sessionStorage` (reset when the tab is closed).

After restarting the detector, a **Sound Detector** device automatically appears in Home Assistant
with binary_sensor and sensor for each class.

Without `MQTT_HOST`, the detector works as before — MQTT is optional.

---

# 12. UDP Audio Protocol

ESP32 sends audio over UDP. Each packet has a header:

```text
Offset  Size   Type         Description
──────  ─────  ───────────  ──────────────────────────────────
0       4      uint32 BE    Magic: 0x53445231 ("SDR1")
4       4      uint32 BE    Sequence number (monotonically increasing)
8       2      uint16 BE    Payload size (bytes)
10      2      uint16 BE    Source ID (0 = default; see docs/MULTI_MIC.md)
12      N      bytes        Audio payload: signed 32-bit I2S samples (little-endian)
```

- `HEADER = struct.Struct("!IIHH")` — 12 bytes, network byte order (big-endian)
- Payload: signed 32-bit I2S slots, little-endian. To get 16-bit PCM,
  do `sample >> 16` and clip to `int16`.
- Sample rate: 16 kHz, mono.
- Typical payload size: 4096 bytes (1024 samples = ~64 ms of audio).
- On packet loss (gap in sequence numbers), the receiver should insert silence
  of the corresponding length.

To connect your own inference engine, just listen on the Hub's UDP port
and parse packets per this specification. Implementation examples:
`recorder/recorder-files/udp_recorder.py` and `detector/transport/udp.py`.

For connecting multiple microphones (ESP32 stereo, USB mic on RPi, etc.)
use the Source ID field — see **[Multi-Microphone Setup](docs/MULTI_MIC.md)**.

---

# 13. Web & WebSocket API

The Detection Engine provides HTTP and WebSocket APIs on port 8080.

## HTTP endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | — | Web UI (single-page) |
| `GET` | `/api/auth/status` | — | `{"auth_enabled": true}` |
| `POST` | `/api/auth/check` | — | Token check: `{"token": "..."}` → `{"ok": true}` |
| `GET` | `/api/settings` | viewer | Current settings + defaults |
| `POST` | `/api/settings` | admin | Update settings (partial update) |
| `POST` | `/api/settings/reset` | admin | Reset to defaults |
| `GET` | `/api/metrics` | viewer | Metrics from last model validation |
| `GET` | `/api/audits` | viewer | Last 5 model audits |
| `POST` | `/feedback` | admin | Assign label to clip: `{"filename": "...", "label": "siren"}` |
| `POST` | `/api/manual-event` | admin | Save current buffer as manual event |
| `GET` | `/clips/{name}` | viewer | Download WAV clip |
| `GET` | `/clips/{name}/spectrogram` | — | Clip spectrogram (JSON: width, height, data) |
| `GET` | `/api/calibrate/recordings` | viewer | List background calibration recordings |
| `DELETE` | `/api/calibrate/recordings` | admin | Delete recording: `{"path": "source/file.wav"}` |
| `GET` | `/calibrate-audio/{source}/{file}` | viewer | Download calibration WAV |
| `GET` | `/calibrate-audio/{source}/{file}/spectrogram` | — | Calibration recording spectrogram |

Auth: `Authorization: Bearer <ADMIN_TOKEN>` header. Viewer and admin use the same token.

## WebSocket endpoints

### `/ws` — events and scores

Connection: `ws://<host>:8080/ws?token=<ADMIN_TOKEN>`

On connection, the server sends:
1. Last score message
2. Event history (up to 100)

Then in real time:

**Score** (~every 0.5s):
```json
{
  "type": "score",
  "top": "Speech",
  "siren_score": 0.42, "siren_state": "possible", "siren_triggers": 5,
  "explosion_score": 0.0, "explosion_state": "idle",
  "audio_alive": true, "audio_pps": 31.2
}
```

**Event** (on state change):
```json
{
  "type": "event",
  "event": "siren", "state": "confirmed",
  "confidence": 0.87, "timestamp": "2026-10-16T12:00:00+00:00",
  "clip_url": "/clips/20261016-120000.wav",
  "engine": "yamnet+mlp"
}
```

### `/ws/audio` — live PCM stream

Connection: `ws://<host>:8080/ws/audio?token=<ADMIN_TOKEN>`

Binary messages: raw float32 PCM, 16 kHz, mono. For browser playback —
decode in an AudioWorklet.

---

# 14. Long-term Goal

Build an affordable and extensible platform:

> **cheap microphone Audio Node + replaceable inference engine + standardized events + local integrations.**

The first practical scenario — detecting potentially dangerous sounds:

```text
🚨 Siren
💥 Explosion
🛵 Motorcycle
```

But the architecture should allow adding new models and classes without changing the base device.

---

## Project Philosophy

**Cheap hardware.  
Local-first.  
Open architecture.  
Replaceable intelligence.  
Real-world data.  
Safety over hype.**

---
