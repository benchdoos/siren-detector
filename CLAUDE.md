# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dangerous sound detector** (sirens, explosions, UAVs). ESP32-S3 with INMP441 microphone acts as an Audio Node (capture + transport), while a separate Detection Engine (currently YAMNet + MLP on Raspberry Pi) handles ML inference. Target classes: `siren`, `explosion`, `uav`, `background`, `unknown`.

## Repository Structure

```
esp32/                    # ESPHome YAML config for ESP32-S3 firmware
recorder/                 # Docker-based UDP recorder (runs on Raspberry Pi)
audio_source/             # USB microphone capture on RPi (Python + Docker, SDR1/UDP)
ml/                       # ML training, evaluation, audit scripts (run on Mac)
hub/                      # UDP fan-out relay + detector (Docker Compose, runs on RPi)
dataset/                  # Labeled WAV recordings (not committed to Git)
label-studio/             # Label Studio annotation stack (Docker Compose, runs on Mac)
docs/                     # Bilingual project documentation (en/ru, MkDocs)

detector/                 # Modular detection engine (Python package)
├── main.py               #   Entry point: wires bus, runtime, engine, server, integrations
├── config.py             #   Constants, detector definitions, default settings
├── audio.py              #   I2S conversion, AudioRingBuffer, ClipBuffer
├── settings.py           #   Dynamic settings: load, save, apply
├── auth.py               #   Admin token auth for FastAPI
├── bus.py                #   EventBus: in-process pub/sub (replaces globals)
├── classifier_infer.py   #   MLP classifier (numpy-only inference)
├── dsp_features.py       #   DSP feature extractor (runtime, numpy-only)
├── dsp_infer.py          #   TFLite DSP inference + rolling feature buffer
├── transport/
│   ├── udp.py            #   UDP receiver (SDR1 protocol)
│   └── file.py           #   WAV file transport (for local tests)
├── engine/
│   ├── interface.py      #   EngineInterface Protocol (contract for custom engines)
│   ├── smoother.py       #   TemporalSmoother (idle → possible → confirmed)
│   ├── yamnet_engine.py  #   YAMNet + MLP inference loop
│   ├── event_policy.py   #   Event policy: thresholds, cooldown, clip triggers
│   ├── source_state.py   #   Per-source state tracking
│   ├── manifest.py       #   Engine manifest (ID, version, classes, schema)
│   ├── registry.py       #   Engine registry (built-in + local Python entrypoints)
│   └── stages/
│       └── yamnet_mlp.py #   YAMNet + MLP as a composable inference stage
├── runtime/              #   Runtime orchestration layer
│   ├── hub.py            #   RuntimeHub: audio fan-out, result routing, subscriptions
│   ├── contracts.py      #   Data contracts (audio-frame, inference-result, events)
│   ├── engine_graph.py   #   Multi-engine graph: configs, routes, lifecycle
│   ├── engine_settings.py#   Per-engine settings storage and validation
│   ├── lifecycle.py      #   Engine lifecycle management (start/stop/error)
│   ├── manifest.py       #   Runtime manifest handling
│   ├── metrics.py        #   Prometheus metrics for engines, subscriptions, outputs
│   ├── policies.py       #   Event merge/filter policies
│   ├── clip_storage.py   #   WAV clip storage adapter
│   └── webhook.py        #   Webhook output adapter (HTTP POST events)
├── examples/
│   └── energy_threshold_engine.py  # Minimal third-party engine example
├── server/
│   ├── app.py            #   FastAPI app factory, static mount, lifespan
│   ├── routes_clips.py   #   /clips, spectrogram, clip restore
│   ├── routes_api.py     #   /api/settings, /feedback, /api/metrics, etc.
│   └── routes_ws.py      #   /ws, /ws/audio WebSocket endpoints
├── integrations/
│   └── mqtt.py           #   MQTT client, HA discovery
└── static/
    ├── index.html          #   Web UI (single-page)
    ├── settings.html       #   Settings page
    └── js/audio-player.js  #   Shared spectrogram + player module
```

## Commands

### ML Baseline (run on Mac, from repo root)

```bash
# First-time setup
uv venv --python 3.11 ml/.venv
uv pip install --python ml/.venv/bin/python -r ml/requirements.txt

# Run YAMNet evaluation on ESC-50
ml/.venv/bin/python ml/evaluate_yamnet.py

# Download and extract ESC-50 dataset
python3 dataset/fetch_esc50.py
```

### Dataset Annotation

```bash
# Annotate a long recording into labeled segments
python3 dataset/annotate_recording.py dataset/inbox/<recording>.wav \
  --timezone Europe/Moscow \
  --segment siren 12:03 12:08 \
  --segment background 12:08 12:15

# Use -f for relative timestamps (MM:SS from file start)
python3 dataset/annotate_recording.py dataset/inbox/<recording>.wav \
  --segment -f siren 0:03 5:08
```

### Recorder (runs on Raspberry Pi via Docker)

```bash
cd ~/Projects/esphome/recorder
docker compose build

# Record labeled audio (labels: siren, explosion, uav, background, unknown, inbox)
LABEL=background DURATION=30 docker compose run --rm recorder
LABEL=siren DURATION=10 docker compose run --rm recorder
# Continuous recording to inbox until Ctrl-C:
docker compose run --rm recorder
```

### ESP32 Firmware (runs on Raspberry Pi with ESPHome Docker)

```bash
cd ~/Projects/esphome
# Validate config
docker compose run --rm esphome config /config/siren-detector.yaml
# Compile
docker compose run --rm esphome compile /config/siren-detector.yaml
# Flash via OTA
docker compose run --rm esphome upload /config/siren-detector.yaml --device siren-detector.local
# Monitor logs
docker compose run --rm esphome logs /config/siren-detector.yaml --device siren-detector.local
```

## Architecture

**Audio Node** (ESP32-S3): I2S capture from INMP441 → PCM 16 kHz / mono / 16-bit → UDP transport (SDR1 protocol) with sequence numbers. Supports `source_id` in SDR1 header for multi-microphone setups.

**Audio Source** (`audio_source/`): USB microphone capture on RPi via `sounddevice`/PortAudio → SDR1/UDP. Alternative audio input alongside ESP32.

**Hub** (`hub/`): UDP fan-out relay — one stream from ESP32/USB mic → multiple consumers (recorder on port 9001, detector on port 9002).

**Recorder** (Raspberry Pi, Docker): Receives UDP packets, inserts silence for dropped packets, saves WAV files to labeled subdirectories.

**Detection Engine** (`detector/`): Modular Python package on RPi. Architecture:
- **Runtime Hub** (`detector/runtime/hub.py`): audio fan-out to engine subscribers, result routing, bounded queues with overflow protection.
- **Engine Graph** (`detector/runtime/engine_graph.py`): multi-engine configs, per-source routes, lifecycle. Engines register via manifest (`detector/engine/manifest.py`) and registry (`detector/engine/registry.py`).
- **YAMNet + MLP** (`detector/engine/yamnet_engine.py`): main inference engine. Optional DSP pre-filter (`detector/dsp_infer.py`) skips YAMNet when background confidence is high.
- **Event Policy** (`detector/engine/event_policy.py`): thresholds, temporal smoothing, cooldown, clip triggers — decoupled from inference.
- **Outputs**: WebSocket, SSE `/events`, REST API, MQTT (HA Discovery), Prometheus metrics, webhooks.
- Custom engines plug in via `EngineInterface` (`detector/engine/interface.py`); see `detector/examples/energy_threshold_engine.py`.

**Dataset pipeline**: Long recordings go to `dataset/inbox/`, then `annotate_recording.py` slices them into labeled WAVs and logs provenance in `dataset/annotations.csv`.

**Augmentation**: `dataset/augment.py` creates realistic training data from clean sources (ESC-50, Freesound) by varying volume, mixing with real INMP441 backgrounds, and optionally applying mic noise profile. Run `make augment` before `make retrain`.

**Docs** (`docs/`): Bilingual (en/ru) MkDocs documentation covering architecture, API, settings, multi-microphone setup, custom engines, data/models, operations.

## Key Constraints

- ESPHome config lives in `esp32/siren-detector.yaml` locally but must be deployed to `~/Projects/esphome/esphome/siren-detector.yaml` on the RPi host (IP configured in `.env` / `Makefile`).
- `dataset/external/esc50/audio/` is git-ignored (CC BY-NC 3.0 license, large files).
- `ml/results/` and `ml/.venv/` are git-ignored.
- INMP441 sends 24-bit audio in 32-bit I2S slots — the firmware shifts right by 8 bits before normalizing.
- ESP32 hostname: `siren-detector.local` (IP depends on local network).


## Detector / Web UI

The detector stack runs on RPi from `hub/docker-compose.yml` (not `detector/docker-compose.yml`):

```bash
# Deploy (from Mac) — use `make deploy` instead, or manually:
rsync -av --delete --exclude='__pycache__' detector/ <user>@<rpi_ip>:~/Projects/esphome/detector/
scp hub/docker-compose.yml <user>@<rpi_ip>:~/Projects/esphome/hub/

# Rebuild & restart (on RPi)
cd ~/Projects/esphome/hub
docker compose up --build -d detector

# Static-only update (no rebuild needed) — or use `make deploy-ui`
scp detector/static/index.html <user>@<rpi_ip>:~/Projects/esphome/detector/static/
ssh <user>@<rpi_ip> "docker cp ~/Projects/esphome/detector/static/index.html siren-detector-engine:/app/detector/static/index.html"
```

Web UI: `http://<rpi_ip>:8080`

Key architectural notes:
- **Modular design**: detector is a Python package with replaceable layers (engine, transport, integrations, runtime). See `detector/engine/interface.py` for the engine contract.
- **Runtime Hub** (`detector/runtime/hub.py`) manages audio fan-out and result routing between engines, event policies, and output adapters.
- **EventBus** (`detector/bus.py`) delivers legacy scores/events to UI and integrations. Thread-safe via `call_soon_threadsafe`.
- **WebSocket** uses `await ws.receive()` (not `receive_text()`) — handles ping frames correctly; `receive_text()` causes silent client drop
- **Feedback flow**: confirmed clip saved to `/clips/` → user clicks label button → `POST /feedback` copies WAV to `/dataset/{label}/` on RPi → ready for retraining
- `feedback` dict is in-memory only (on `EventBus`) — labels are lost on container restart; history replay includes current labels so UI restores correctly after browser reload
- **Volumes**: `clips` (named, for WAV clips), `../dataset` (bind mount for feedback copies), `../ml/results` (read-only, classifier weights)
- Thresholds are dynamic settings, defaults: `possible_triggers=4`, `confirmed_triggers=8`, `siren_threshold=0.10`, `cooldown_seconds=60`
- **Background calibration**: Web UI Calibrate button records ambient audio from a specific source → saves to `/dataset/background/{source_name}/` → `make retrain` picks it up via `rglob`. Solves domain gap when adding new microphones with different audio profiles.

## Retraining workflow

Using Makefile (preferred):

```bash
make retrain         # Full cycle: merge-recorder → sync → augment → train-dsp → train → audit → deploy → clean
# Or step by step:
make sync-dataset    # Sync labeled data from RPi
make augment         # Augment siren + explosion + uav data
make train-dsp       # Train DSP pre-filter (Conv1D on DSP features)
make train           # Train MLP classifier
make audit           # Run dataset & model audit snapshot
make deploy-model    # Push weights + DSP model to RPi + restart
```

Other useful Makefile targets: `make deploy`, `make deploy-audio`, `make deploy-ui`, `make deploy-firmware`, `make logs`, `make logs-esp`, `make audit-compare`, `make settings`, `make help`.

## After edits
- claude or codex should update `README.md` with a summary of the project, setup instructions, and usage examples.
- claude or codex should also update `TODO.md` with a clear list of tasks, marking completed items and indicating next steps for development and testing.
- show list of affected files and directories, including any new files created, and any files that were deleted or renamed.