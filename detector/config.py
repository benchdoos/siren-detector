"""Constants, detector definitions, and default settings."""

from __future__ import annotations

import struct
from pathlib import Path

SAMPLE_RATE = 16_000
PACKET_MAX = 4096 + 12
HEADER = struct.Struct("!IIHH")
MAGIC = 0x53445231  # "SDR1"

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
ANALYSIS_SECONDS = 2.0
INFERENCE_HOP_SAMPLES = SAMPLE_RATE // 2

# Each detector: event name → (YAMNet class names, threshold, clip prefix)
DETECTORS = {
    "siren":      (("Siren",), 0.10, ""),
    "explosion":  (("Explosion", "Burst, pop", "Gunshot, gunfire"), 0.15, "exp-"),
    "motorcycle": (("Motorcycle",), 0.15, "moto-"),
}

CLIP_SECONDS = 10
CLIP_DIR = Path("/clips")
WEB_PORT = 8080

DATASET_DIR = Path("/dataset")
VALID_LABELS = {"siren", "background", "motorcycle", "explosion", "unknown"}

SETTINGS_PATH = CLIP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "siren_threshold": 0.10,
    "explosion_threshold": 0.15,
    "motorcycle_threshold": 0.15,
    "possible_triggers": 4,
    "confirmed_triggers": 8,
    "cooldown_seconds": 60,
    "refractory_seconds": 15,
    "smoothing_window_seconds": 30,
    "yamnet_min_gate": 0.01,
    "clip_seconds": 10,
    "clip_from_possible": 0,
    "disabled_sources": [],
    "source_custom_names": {},
    "detection_mode": "round_robin",
}
