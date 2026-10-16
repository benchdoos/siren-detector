# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal **Audio Event Detection** platform. ESP32-S3 with INMP441 microphone acts as an Audio Node (capture + transport), while a separate Detection Engine (currently YAMNet on Mac/Raspberry Pi) handles ML inference. Target classes: `siren`, `explosion`, `motorcycle`, `background`, `unknown`.

## Repository Structure

```
esp32/           # ESPHome YAML config for ESP32-S3 firmware
recorder/        # Docker-based UDP recorder (runs on Raspberry Pi)
ml/              # ML baseline scripts (run on Mac)
hub/             # UDP fan-out relay + detector (Docker Compose, runs on RPi)
detector/        # FastAPI detection engine: YAMNet + MLP + WebSocket UI + MQTT
dataset/         # Labeled WAV recordings (not committed to Git)
mosquitto/       # Standalone MQTT broker (Docker Compose, runs on RPi)
label-studio/    # Label Studio annotation stack (Docker Compose, runs on Mac)
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

# Record labeled audio (labels: siren, explosion, motorcycle, background, unknown, inbox)
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

**Audio Node** (ESP32-S3): I2S capture from INMP441 → PCM 16 kHz / mono / 16-bit → UDP transport with sequence numbers to Raspberry Pi port 9000.

**Recorder** (Raspberry Pi, Docker): Receives UDP packets, inserts silence for dropped packets, saves WAV files to labeled subdirectories.

**Detection Engine** (Mac/Raspberry Pi): YAMNet baseline in `ml/evaluate_yamnet.py`; evaluates ESC-50 clips, writes results to `ml/results/` (git-ignored).

**Dataset pipeline**: Long recordings go to `dataset/inbox/`, then `annotate_recording.py` slices them into labeled WAVs and logs provenance in `dataset/annotations.csv`.

**Augmentation**: `dataset/augment.py` creates realistic training data from clean sources (ESC-50, Freesound) by varying volume, mixing with real INMP441 backgrounds, and optionally applying mic noise profile. Run `make augment` before `make retrain`.

## Key Constraints

- ESPHome config lives in `esp32/siren-detector.yaml` locally but must be deployed to `~/Projects/esphome/esphome/siren-detector.yaml` on the RPi host (IP configured in `config.mk` / `Makefile`).
- `dataset/external/esc50/audio/` is git-ignored (CC BY-NC 3.0 license, large files).
- `ml/results/` and `ml/.venv/` are git-ignored.
- INMP441 sends 24-bit audio in 32-bit I2S slots — the firmware shifts right by 8 bits before normalizing.
- ESP32 hostname: `siren-detector.local` (IP depends on local network).


## Detector / Web UI

The detector stack runs on RPi from `hub/docker-compose.yml` (not `detector/docker-compose.yml`):

```bash
# Deploy (from Mac) — use `make deploy` instead, or manually:
scp -r detector/ <user>@<rpi_ip>:~/Projects/esphome/
scp hub/docker-compose.yml <user>@<rpi_ip>:~/Projects/esphome/hub/

# Rebuild & restart (on RPi)
cd ~/Projects/esphome/hub
docker compose up --build -d detector

# Static-only update (no rebuild needed) — or use `make deploy-ui`
scp detector/static/index.html <user>@<rpi_ip>:~/Projects/esphome/detector/static/
ssh <user>@<rpi_ip> "docker cp ~/Projects/esphome/detector/static/index.html siren-detector-engine:/app/static/index.html"
```

Web UI: `http://<rpi_ip>:8080`

Key architectural notes:
- **WebSocket** uses `await ws.receive()` (not `receive_text()`) — handles ping frames correctly; `receive_text()` causes silent client drop
- **Feedback flow**: confirmed clip saved to `/clips/` → user clicks label button → `POST /feedback` copies WAV to `/dataset/{label}/` on RPi → ready for retraining
- `_feedback` dict is in-memory only — labels are lost on container restart; history replay includes current labels so UI restores correctly after browser reload
- **Volumes**: `clips` (named, for WAV clips), `../dataset` (bind mount for feedback copies), `../ml/results` (read-only, classifier weights)
- Thresholds: `POSSIBLE_TRIGGERS=4`, `CONFIRMED_TRIGGERS=8`, `SIREN_THRESHOLD=0.10`, `COOLDOWN_SECONDS=60`

## Retraining workflow

Using Makefile (preferred):

```bash
make retrain         # Full cycle: sync → train → deploy → clean
# Or step by step:
make sync-dataset    # Sync labeled data from RPi
make train           # Train MLP classifier
make deploy-model    # Push weights to RPi + restart
```

Manual commands (equivalent):

```bash
rsync -av <user>@<rpi_ip>:~/Projects/esphome/dataset/ dataset/
ml/.venv/bin/python ml/train_classifier.py
scp ml/results/classifier_weights.npz ml/results/classifier_labels.json \
    <user>@<rpi_ip>:~/Projects/esphome/ml/results/
ssh <user>@<rpi_ip> "docker restart siren-detector-engine"
```

Other useful Makefile targets: `make deploy`, `make deploy-ui`, `make logs`, `make help`.

## After edits
- claude or codex should update `README.md` with a summary of the project, setup instructions, and usage examples.
- claude or codex should also update `TODO.md` with a clear list of tasks, marking completed items and indicating next steps for development and testing.
- show list of affected files and directories, including any new files created, and any files that were deleted or renamed.