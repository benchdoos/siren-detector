# Dataset layout

Keep recordings made with the INMP441 separate from downloaded reference clips.
The recorder writes only to the first-party folders (`siren/`, `background/`,
etc.); do not point it at `external/`.

```text
dataset/
├── siren/                 # recordings from our device
├── explosion/
├── motorcycle/
├── background/
├── unknown/
├── inbox/
└── external/
    └── esc50/             # optional third-party reference set
        ├── audio/         # ignored by Git; download locally
        ├── meta/esc50.csv # keep provenance and folds
        ├── LICENSE        # keep original license/attribution
        └── SOURCE.md      # import date and selected categories
```

## ESC-50

Run `python3 dataset/fetch_esc50.py` from the repository root to download the
official archive and extract only the selected categories below. The script
keeps original filenames, the upstream CSV and LICENSE together. ESC-50 is
CC BY-NC 3.0 as a whole; its ESC-10 subset is CC BY 3.0. The audio is
intentionally excluded from Git. Check individual clip attributions in the
upstream LICENSE, especially before reuse beyond non-commercial experiments.

Useful initial classes:

| ESC-50 category | Project use | Notes |
|---|---|---|
| `siren` | positive siren examples | Not a substitute for local recordings |
| `rain`, `wind`, `washing_machine`, `keyboard_typing`, `engine` | background variety | These are short individual clips, not long natural recordings |
| `fireworks` | `unknown` / hard negative | Do not relabel as explosion ground truth |

The dataset has five folds. When evaluating a model, split by the supplied fold
and never randomly split excerpts from the same source across train and test.
Treat results as a quick baseline only: microphone, room, distance, and recording
conditions differ from our ESP32 setup. Add local recordings for real evaluation.

Do not mix these files into `dataset/siren/` or `dataset/background/`; that
would erase their provenance and make future evaluation harder to trust.

## Pretrained baseline

The isolated YAMNet sanity-check on the Mac is documented in [`../ml/README.md`](../ml/README.md).
Its dependencies, model cache, and results stay outside the recorder and ESP32
firmware workflow.
