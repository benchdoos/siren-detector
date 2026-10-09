# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal **Audio Event Detection** platform. ESP32-S3 with INMP441 microphone acts as an Audio Node (capture + transport), while a separate Detection Engine (currently YAMNet on Mac/Raspberry Pi) handles ML inference. Target classes: `siren`, `explosion`, `motorcycle`, `background`, `unknown`.

## Repository Structure

```
esp32/           # ESPHome YAML config for ESP32-S3 firmware
recorder/        # Docker-based UDP recorder (runs on Raspberry Pi)
ml/              # ML baseline scripts (run on Mac)
dataset/         # Labeled WAV recordings (not committed to Git)
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

## Key Constraints

- ESPHome config lives in `esp32/siren-detector.yaml` locally but must be deployed to `~/Projects/esphome/esphome/siren-detector.yaml` on Raspberry Pi (`192.168.1.100`).
- `dataset/external/esc50/audio/` is git-ignored (CC BY-NC 3.0 license, large files).
- `ml/results/` and `ml/.venv/` are git-ignored.
- INMP441 sends 24-bit audio in 32-bit I2S slots — the firmware shifts right by 8 bits before normalizing.
- Current ESP32 IP: `192.168.88.49` / hostname: `siren-detector.local`.
