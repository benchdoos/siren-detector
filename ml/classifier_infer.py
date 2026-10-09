"""
Minimal numpy classifier for use in detector.py on Raspberry Pi.

Usage:
    clf = SirenClassifier("ml/results/classifier_weights.npz",
                          "ml/results/classifier_labels.json")
    probs = clf.predict(embeddings)   # embeddings: (N, 1024) float32
    label, score = clf.top1(embeddings[0])
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


class SirenClassifier:
    def __init__(self, weights_path: str | Path, labels_path: str | Path) -> None:
        w = np.load(weights_path)
        self._W1, self._b1 = w["W1"], w["b1"]
        self._W2, self._b2 = w["W2"], w["b2"]
        self._W3, self._b3 = w["W3"], w["b3"]
        self.labels: list[str] = json.loads(Path(labels_path).read_text())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (N, num_classes)."""
        x = np.maximum(0.0, x @ self._W1 + self._b1)   # Dense 256 + ReLU
        x = np.maximum(0.0, x @ self._W2 + self._b2)   # Dense 128 + ReLU
        x = x @ self._W3 + self._b3                     # Dense num_classes
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    def top1(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return (label, score) for a single 1024-dim embedding."""
        probs = self.predict(embedding[np.newaxis])[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])
