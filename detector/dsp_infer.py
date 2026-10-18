"""TFLite-based DSP classifier for pre-filtering before YAMNet.

Loads a quantised Conv1D model (INT8, ~30 KB) trained on DSP features.
Used as a lightweight gate: if DSP says "background" with high confidence,
skip the heavy YAMNet inference.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

import numpy as np

from .dsp_features import (
    N_RAW_FEATURES,
    add_deltas,
    extract_features,
)

WINDOW_FRAMES = 250   # 5 seconds at 50 fps (must match training)
N_FEATURES = N_RAW_FEATURES * 2  # 8 raw + 8 delta = 16


class DspPreFilter:
    """Lightweight DSP classifier that gates YAMNet inference.

    Accumulates DSP features in a rolling buffer (per source).
    When buffer is full (250 frames = 5s), runs Conv1D and decides
    whether to skip YAMNet.
    """

    def __init__(
        self,
        model_path: str | Path,
        norm_stats_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        import tensorflow as tf

        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        # A single interpreter is shared by all per-source inference threads.
        # TFLite's set_tensor/invoke/get_tensor sequence is not thread-safe.
        self._inference_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._checks = 0
        self._yamnet_skips = 0
        self._prediction_counts: dict[str, int] = {}
        self._background_confidence_sum = 0.0
        self._background_confidence_max = 0.0
        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_detail = self._interpreter.get_output_details()[0]

        stats = np.load(norm_stats_path)
        self._mean = stats["mean"].astype(np.float32)  # (16,)
        self._std = stats["std"].astype(np.float32)     # (16,)

        self.labels: list[str] = json.loads(Path(labels_path).read_text())
        self._bg_idx: int | None = (
            self.labels.index("background") if "background" in self.labels else None
        )
        print(f"[dsp] Loaded: {Path(model_path).name}, labels={self.labels}", flush=True)

    def create_feature_buf(self) -> collections.deque:
        """Create a per-source rolling feature buffer."""
        return collections.deque(maxlen=WINDOW_FRAMES)

    def process(
        self,
        new_audio: np.ndarray,
        feature_buf: collections.deque,
    ) -> dict:
        """Extract features from new audio, accumulate, and optionally infer.

        Args:
            new_audio: float32 mono 16kHz audio (typically last 0.5s = 8000 samples).
            feature_buf: per-source rolling deque of feature frames.

        Returns:
            dict with keys:
              - ready: bool (enough frames for inference)
              - skip_yamnet: bool (DSP says confident background)
              - label: str or None
              - confidence: float
              - probs: dict[str, float] or None
        """
        feats = extract_features(new_audio)
        for row in feats:
            feature_buf.append(row)

        if len(feature_buf) < WINDOW_FRAMES:
            return {"ready": False, "skip_yamnet": False,
                    "label": None, "confidence": 0.0, "probs": None}

        feat_array = np.array(feature_buf, dtype=np.float32)[-WINDOW_FRAMES:]
        feat_16 = add_deltas(feat_array)

        feat_norm = (feat_16 - self._mean) / (self._std + 1e-8)

        probs = self._infer(feat_norm)
        top_idx = int(probs.argmax())
        label = self.labels[top_idx]
        confidence = float(probs[top_idx])

        probs_dict = {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))}

        return {
            "ready": True,
            "skip_yamnet": False,  # caller decides based on threshold
            "label": label,
            "confidence": confidence,
            "probs": probs_dict,
        }

    def record_decision(self, label: str, confidence: float, skipped_yamnet: bool) -> None:
        """Count completed DSP decisions across all audio sources."""
        with self._stats_lock:
            self._checks += 1
            self._prediction_counts[label] = self._prediction_counts.get(label, 0) + 1
            if label == "background":
                self._background_confidence_sum += confidence
                self._background_confidence_max = max(self._background_confidence_max, confidence)
            if skipped_yamnet:
                self._yamnet_skips += 1

    def stats_snapshot(self) -> dict:
        with self._stats_lock:
            bg_count = self._prediction_counts.get("background", 0)
            return {
                "checks": self._checks,
                "yamnet_skips": self._yamnet_skips,
                "prediction_counts": dict(self._prediction_counts),
                "background_confidence_avg": (
                    self._background_confidence_sum / bg_count if bg_count else None
                ),
                "background_confidence_max": self._background_confidence_max if bg_count else None,
            }

    def _infer(self, features: np.ndarray) -> np.ndarray:
        """Run TFLite inference on normalised (250, 16) features.

        Returns softmax probabilities (n_classes,).
        """
        inp = features[np.newaxis].astype(np.float32)  # (1, 250, 16)

        input_dtype = self._input_detail["dtype"]
        if input_dtype == np.int8:
            scale, zero_point = self._input_detail["quantization"]
            inp = (inp / scale + zero_point).clip(-128, 127).astype(np.int8)

        with self._inference_lock:
            self._interpreter.set_tensor(self._input_detail["index"], inp)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_detail["index"])

        output_dtype = self._output_detail["dtype"]
        if output_dtype == np.int8:
            scale, zero_point = self._output_detail["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        # The trained model ends with a softmax layer, so TFLite already
        # returns probabilities. Applying softmax again compresses the maximum
        # possible confidence to about 47.5% for four classes, making the
        # configured 0.85 background skip threshold unreachable.
        probs = np.maximum(output[0].astype(np.float32), 0.0)
        total = probs.sum()
        if total <= 0:
            return np.full(len(self.labels), 1.0 / len(self.labels), dtype=np.float32)
        return probs / total
