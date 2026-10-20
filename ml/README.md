# YAMNet baseline

Run this on the Mac from the repository root. It does not alter audio files and
does not move data to Raspberry Pi.

```bash
uv venv --python 3.11 ml/.venv
uv pip install --python ml/.venv/bin/python -r ml/requirements.txt
ml/.venv/bin/python ml/evaluate_yamnet.py
```

The first run downloads TensorFlow dependencies and the pretrained YAMNet model.
The script converts the curated ESC-50 clips to mono float audio at 16 kHz in
memory, evaluates all five official folds, and writes per-file predictions to
`ml/results/esc50-yamnet.csv` (ignored by Git).

The report includes top-1 matches where YAMNet has an equivalent label group
(there is no washing-machine class) and fold-wise ROC AUC for the model's
`Siren` score against the selected non-siren examples. These are reference-set sanity checks only; ESC-50 is
CC BY-NC 3.0 and is acoustically different from our microphone recordings.
YAMNet is pretrained on AudioSet and is not trained/fine-tuned here.

## Project recordings

From the repository root, run YAMNet over the WAV files directly under
`dataset/siren/` and `dataset/background/`:

```bash
ml/.venv/bin/python ml/evaluate_recordings_yamnet.py
```

To test only a specific file or folder, pass it as an argument:

```bash
ml/.venv/bin/python ml/evaluate_recordings_yamnet.py \
  dataset/siren/20261009-074711-497701__siren__0-173s.wav
```

The script writes `ml/results/recordings-yamnet/summary.csv` with mean, peak,
and 95th-percentile scores for relevant YAMNet classes, plus
`frame_scores.csv` with scores every 0.48 seconds. Long WAVs are read and
scored in bounded chunks, so they are not loaded fully into RAM. The frame
timestamps are offsets from the beginning of each input WAV. YAMNet scores are
model outputs, not calibrated probabilities or final alarm thresholds. The
default scan includes every WAV in `background/`, including long captures, so
the first complete run may take a while.

## DSP pre-filter training

Run `make train-dsp` from the repository root. Calibration diversity is enabled
by default: the trainer reads full calibration recordings, divides each into
non-overlapping five-second chunks, and keeps up to 16 quiet, typical, loud, and
acoustically distinct chunks per recording. Calibration chunks are prioritized
within the background-window limit; other recordings fill the remaining space.
To compare with the 30-second-per-file behavior, run
`make train-dsp DSP_TRAIN_ARGS=--no-calibration-diversity`.

The server uses the float32 TFLite export. The current INT8 export sharply
misclassifies the calibration recordings, so it is retained for analysis but is
not deployed. `make deploy` updates the detector code; `make deploy-model`
copies the trained float32 model and refreshes the detector Compose command so
the running container loads that export.
