"""YAMNet + project MLP inference stage."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow_hub as hub

from ...classifier_infer import SirenClassifier
from ...config import DETECTORS, MODEL_URL
from ...runtime.contracts import InferenceResult


class YAMNetMlpStage:
    """Loads YAMNet and an optional MLP head and returns raw model scores."""

    def __init__(
        self,
        yamnet,
        class_names: list[str],
        detector_indices: dict[str, list[int]],
        classifier: SirenClassifier | None,
    ) -> None:
        self._yamnet = yamnet
        self._class_names = class_names
        self._detector_indices = detector_indices
        self._classifier = classifier

    @classmethod
    def load(
        cls,
        classifier_weights: str | Path,
        classifier_labels: str | Path,
    ) -> "YAMNetMlpStage":
        print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
        yamnet = hub.load(MODEL_URL)

        with open(yamnet.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
            class_names = [row["display_name"] for row in csv.DictReader(f)]

        detector_indices: dict[str, list[int]] = {}
        for name, (yamnet_classes, _, _) in DETECTORS.items():
            indices = [i for i, label in enumerate(class_names) if label in yamnet_classes]
            detector_indices[name] = indices
            print(f"Detector '{name}' YAMNet classes: {[class_names[i] for i in indices]}", flush=True)

        weights_path = Path(classifier_weights)
        labels_path = Path(classifier_labels)
        classifier = None
        if weights_path.exists() and labels_path.exists():
            classifier = SirenClassifier(weights_path, labels_path)
            print(f"Classifier loaded: {classifier.labels}", flush=True)
        else:
            print("No classifier weights — using raw YAMNet siren score.", flush=True)

        return cls(yamnet, class_names, detector_indices, classifier)

    @property
    def has_classifier(self) -> bool:
        return self._classifier is not None

    @property
    def engine_name(self) -> str:
        return "yamnet+mlp" if self.has_classifier else "yamnet"

    def predict(self, source_id: int, waveform: np.ndarray) -> InferenceResult:
        """Run inference and return a model-agnostic runtime contract."""
        scores, embeddings, _ = self._yamnet(waveform)
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)
        top_idx = int(mean_scores.argmax())
        yamnet_top = self._class_names[top_idx]

        yamnet_scores = {
            name: float(scores_np[:, indices].max()) if indices else 0.0
            for name, indices in self._detector_indices.items()
        }
        yamnet_top3_idx = mean_scores.argsort()[::-1][:3]
        yamnet_top3 = [
            {"name": self._class_names[i], "score": round(float(mean_scores[i]), 3)}
            for i in yamnet_top3_idx
        ]

        mlp_score = None
        mlp_top1 = None
        mlp_top3: list[dict] = []
        if self._classifier is not None:
            probabilities = self._classifier.predict(embeddings.numpy())
            mean_probabilities = probabilities.mean(axis=0)
            mlp_top1 = self._classifier.labels[int(mean_probabilities.argmax())]
            mlp_top3_idx = mean_probabilities.argsort()[::-1][:3]
            mlp_top3 = [
                {"name": self._classifier.labels[i], "score": round(float(mean_probabilities[i]), 3)}
                for i in mlp_top3_idx
            ]
            if "siren" in self._classifier.labels:
                siren_idx = self._classifier.labels.index("siren")
                mlp_score = float(mean_probabilities[siren_idx])

        namespaced_scores = {f"yamnet:{name}": value for name, value in yamnet_scores.items()}
        if self._classifier is not None:
            mean_probabilities = probabilities.mean(axis=0)
            namespaced_scores.update({
                f"mlp:{label}": float(mean_probabilities[index])
                for index, label in enumerate(self._classifier.labels)
            })
        return InferenceResult(
            source_id=source_id,
            stage_id="yamnet-mlp",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            scores=namespaced_scores,
            metadata={
                "yamnet_top": yamnet_top,
                "yamnet_top3": yamnet_top3,
                "mlp_top1": mlp_top1,
                "mlp_top3": mlp_top3,
                "has_classifier": self.has_classifier,
            },
        )
