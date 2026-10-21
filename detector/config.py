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

# Each detector: event name → (YAMNet class names, threshold, clip prefix)
DETECTORS = {
    "siren":      (("Siren",), 0.10, ""),
    "explosion":  (("Explosion", "Burst, pop", "Gunshot, gunfire"), 0.15, "exp-"),
    "uav": (("UAV",), 0.15, "uav-"),
}

CLIP_SECONDS = 10
CLIP_DIR = Path("/clips")
WEB_PORT = 8080

DATASET_DIR = Path("/dataset")
VALID_LABELS = {"siren", "background", "uav", "explosion", "unknown"}

SETTINGS_PATH = CLIP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "siren_threshold": 0.10,
    "explosion_threshold": 0.15,
    "uav_threshold": 0.15,
    "possible_triggers": 4,
    "confirmed_triggers": 8,
    "cooldown_seconds": 60,
    "refractory_seconds": 15,
    "smoothing_window_seconds": 30,
    "yamnet_min_gate": 0.01,
    "clip_seconds": 10,
    "user_event_clip_seconds": 15,
    "save_background_segments": False,
    "clip_from_possible": 0,
    "disabled_sources": [],
    "source_custom_names": {},
    "detection_mode": "round_robin",
    "inference_interval_seconds": 0.5,
    "dsp_enabled": 1,
    "dsp_bg_threshold": 0.92,
    "dsp_bg_release_threshold": 0.90,
    "dsp_bg_confirm_windows": 2,
    "dsp_max_consecutive_skips": 20,
    "dsp_event_gate": 0.30,
    "dsp_source_overrides": {},
    "click_removal_enabled": 1,
    "click_removal_threshold": 10.0,
    "click_removal_max_width": 5,
    "click_removal_source_overrides": {},
}

# DSP pre-filter model paths (defaults for Docker volume mount)
DSP_MODEL_PATH = "/models/dsp_classifier_f32.tflite"
DSP_NORM_STATS_PATH = "/models/dsp_norm_stats.npz"
DSP_LABELS_PATH = "/models/dsp_classifier_labels.json"
