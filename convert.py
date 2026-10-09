#!/usr/bin/env python3
"""
Export classifier weights as numpy .npz for portable inference.

TFLite conversion is broken in TF 2.16 (MLIR bug).
The classifier is just 3 dense layers — numpy inference is simpler and
runs anywhere without TF dependency.

Output: ml/results/classifier_weights.npz
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

RESULTS = Path("ml/results")

model = tf.keras.models.load_model(str(RESULTS / "classifier.keras"))
w = model.get_weights()  # [W1, b1, W2, b2, W3, b3]

out = RESULTS / "classifier_weights.npz"
np.savez(out, W1=w[0], b1=w[1], W2=w[2], b2=w[3], W3=w[4], b3=w[5])
print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")

labels = json.loads((RESULTS / "classifier_labels.json").read_text())
print(f"Labels: {labels}")
print("Inference: see ml/classifier_infer.py")
