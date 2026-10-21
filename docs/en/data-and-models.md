# Data and models

## Dataset layout

Keep first-party recordings separate from external reference audio. The main labels are `siren`, `explosion`, `uav`, `background`, and `unknown`; long unreviewed recordings belong in `dataset/inbox/`.

```text
dataset/
├── siren/  explosion/  uav/  background/  unknown/
├── inbox/
└── external/esc50/
```

Audio files and local background recordings are ignored by Git. Preserve source attribution and license files for downloaded datasets. See `dataset/README.md` for ESC-50 and recording annotation details.

## Record audio with the recorder

The `recorder/` service captures raw UDP audio from the Hub and saves WAV files on the Linux host. It runs as a Docker container alongside the Hub.

```bash
cd recorder

# Record 30 seconds of background audio
LABEL=background DURATION=30 docker compose run --rm recorder

# Record 10 seconds of siren audio
LABEL=siren DURATION=10 docker compose run --rm recorder

# Continuous recording to inbox (stop with Ctrl-C)
docker compose run --rm recorder
```

Supported labels: `siren`, `explosion`, `uav`, `background`, `unknown`, `inbox`. Files are saved to `dataset/recorder/<label>/` on the host (bind-mounted volume). Each recording also gets a JSON sidecar with metadata (start/stop time, sample rate, packet stats).

The recorder listens on port `9001` by default (`RECORDER_PORT` in `.env`). The Hub must be configured to forward packets to this port.

After recording, `make retrain` automatically merges files from `dataset/recorder/` into the main `dataset/` directories before syncing.

## Label audio with annotate_recording.py

For long inbox recordings, use `dataset/annotate_recording.py` to cut time ranges into labeled segments:

```bash
# Absolute timestamps (wall-clock time from the recording metadata)
python3 dataset/annotate_recording.py dataset/inbox/20261009-070807.wav \
  --timezone Europe/Moscow \
  --segment siren 12:03 12:08 \
  --segment background 12:08 12:15

# Relative timestamps (MM:SS from file start)
python3 dataset/annotate_recording.py dataset/inbox/20261009-070807.wav \
  --segment -f siren 0:03 5:08
```

Segments are saved as 5-second WAV clips in `dataset/<label>/` and logged in `dataset/annotations.csv`.

## Label audio with Label Studio

The `label-studio/` directory contains a Docker Compose stack for browser-based audio annotation with waveform visualization.

### Setup

```bash
cd label-studio
docker compose up -d
```

This starts two services:

- **Label Studio** at `http://localhost:8080` — the annotation interface.
- **Audio server** (nginx) at `http://localhost:8081` — serves WAV files from `dataset/` so Label Studio can load them.

### Usage

1. Open `http://localhost:8080` and create a project.
2. In project settings, paste the labeling config from `label-studio/labeling_config.xml`. It defines audio segmentation labels: `siren` (hotkey 1), `background` (2), `explosion` (3), `uav` (4), `unknown` (5).
3. Import tasks — either manually or using `label-studio/import.json` as a template. Each task references an audio URL from the nginx server (`http://localhost:8081/inbox/<filename>.wav`).
4. Annotate by selecting regions on the waveform and assigning labels.
5. Export annotations and use `ml/ls_export.py` to convert them into labeled WAV segments for training.

Pre-annotations from YAMNet can be imported via `ml/ls_import.py` to seed the annotation with model predictions.

## Label audio from the web UI

The detector web UI provides in-place labeling. When an event is confirmed, a clip is saved. Click the label buttons (siren / background / uav / unknown) to copy the clip to `dataset/<label>/` on the host. Segment labeling is also available: drag on the spectrogram to select a range, choose a label, and the segment is cut and saved.

Background calibration records ambient sound from a selected source into `dataset/background/<source>/`. Calibration examples are useful for adapting the background model to a microphone's acoustic profile; collect representative quiet, typical, and loud conditions.

## Train and deploy

The usual retraining command is `make retrain`. It merges recorder files on the host, waits for an active calibration to finish, syncs data, augments selected classes, trains DSP and MLP classifiers, audits the results, and deploys available model files. This workflow requires the SSH and web UI values in the root `.env`.

For individual steps:

```bash
make sync-dataset
make augment
make train-dsp
make train
make audit
make deploy-model
```

The MLP is trained on frozen YAMNet embeddings; this project does not fine-tune YAMNet. Validation is split by source recording to reduce leakage between related segments. Review per-class metrics and confusion matrices, not only overall accuracy. Model scores are not calibrated probabilities.

## Augmentation

`dataset/augment.py` creates realistic training samples from clean audio sources (ESC-50, Freesound downloads) by simulating real INMP441 recording conditions:

1. **Volume variation** — random gain between -20 dB and +3 dB to simulate different distances from the sound source.
2. **Background mixing** — overlays the clean signal onto real INMP441 background recordings (from `dataset/background/`) at random SNR between -3 dB and +20 dB.
3. **Noise profile** — optionally adds the INMP441 microphone noise pattern recorded from silence (`--noise-profile dataset/background/noise-profile-inmp441.wav`).

Each augmented sample is a 5-second chunk saved as `dataset/<label>/aug_<source>_<hash>.wav`.

### Commands

```bash
# Augment all classes (cleans old augmented files first)
make augment

# Individual classes
make augment-siren       # 172 samples from ESC-50 + Freesound sirens
make augment-explosion   # 150 samples from Freesound + converted downloads
make augment-uav         # 120 normal + 40 quiet samples

# Manual usage with options
ml/.venv/bin/python dataset/augment.py siren --count 200 --noise-profile dataset/background/noise-profile-inmp441.wav
ml/.venv/bin/python dataset/augment.py uav --count 40 --quiet   # distant/muffled UAV sounds
ml/.venv/bin/python dataset/augment.py siren --count 10 --dry-run  # preview without writing
```

### Quiet mode

The `--quiet` flag generates distant/muffled samples with lower volume (-30 to -10 dB) and worse SNR (-6 to +5 dB). Useful for UAV where distant sounds are common.

### Source directories

| Label | Sources |
|---|---|
| `siren` | `dataset/external/esc50_curated/siren/`, `dataset/external/freesound/audio/` |
| `explosion` | `dataset/external/esc50_curated/explosion/`, `dataset/explosion/dl_*.wav` |
| `uav` | `dataset/uav/dl_*.wav`, `dataset/uav/ls_*.wav` |

Backgrounds for mixing come from `dataset/background/*.wav` (date-stamped INMP441 recordings only).

### Guidelines

- Keep augmented samples at or below roughly 2x the number of real samples per class.
- Run `make clean-aug` to remove all `aug_*` files before regenerating.
- Always run `make augment` before `make train` (or use `make retrain` which does both).

## Comparing multiple engines

Do not rank engines from a single aggregate accuracy score. Evaluate every
candidate on the same versioned, labelled recordings, split by original source
recording so windows from one source cannot leak between train and validation.
Report per-class precision, recall and F1, the confusion matrix, false events
per hour, and event detection latency. Record the engine and model versions,
settings, dataset revision, and hardware with each result. Scores are not
calibrated probabilities and must not be averaged across engines as if they
were comparable probabilities. Keep engine selection manual until the project
has a representative labelled evaluation set and an agreed acceptance criterion.

See `ml/TRAINING_GUIDE.md` for data balance and validation guidance, `ml/README.md` for standalone YAMNet evaluation, and `ml/AUDIT.md` for recorded audit history.

## Data quality notes

- Prefer real recordings from the deployment microphones and environments.
- Keep augmented samples at or below roughly twice the number of real samples for a class, then inspect validation results.
- Do not treat fireworks as ground-truth explosions or short ESC-50 clips as a substitute for deployment recordings.
- Remove no data from the host until it has been synced and the training result has been checked.
