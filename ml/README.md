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
