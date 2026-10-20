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

## Record and label audio

Run the recorder on the Linux host when you want labeled examples. For a continuous recording, set `LABEL=inbox` and `DURATION=0`; stop with Ctrl-C. Copy the WAV and its matching JSON sidecar into `dataset/inbox/` without renaming them. Use `dataset/annotate_recording.py` to cut time ranges into labeled samples and update `dataset/annotations.csv`.

The web UI can also save feedback clips into the dataset. Background calibration records ambient sound from a selected source into `dataset/background/<source>/`. Calibration examples are useful for adapting the background model to a microphone's acoustic profile; collect representative quiet, typical, and loud conditions.

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

See `ml/TRAINING_GUIDE.md` for data balance and validation guidance, `ml/README.md` for standalone YAMNet evaluation, and `ml/AUDIT.md` for recorded audit history.

## Data quality notes

- Prefer real recordings from the deployment microphones and environments.
- Keep augmented samples at or below roughly twice the number of real samples for a class, then inspect validation results.
- Do not treat fireworks as ground-truth explosions or short ESC-50 clips as a substitute for deployment recordings.
- Remove no data from the host until it has been synced and the training result has been checked.
