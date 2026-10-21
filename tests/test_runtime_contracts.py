"""Runtime contract compatibility checks."""

from __future__ import annotations

import unittest

import numpy as np

from detector.runtime.contracts import AudioFrame, InferenceResult, normalize_detector_event


class RuntimeContractTests(unittest.TestCase):
    def test_audio_frame_rejects_unknown_contract_and_format(self) -> None:
        base = {
            "source_id": 1,
            "sequence": 0,
            "timestamp": "2026-10-21T00:00:00Z",
            "sample_rate_hz": 16_000,
            "samples": np.zeros(8, dtype=np.float32),
            "metadata": {},
        }
        with self.assertRaisesRegex(ValueError, "unsupported audio frame contract"):
            AudioFrame(**base, contract="audio-frame.v2")
        with self.assertRaisesRegex(ValueError, "unsupported audio sample format"):
            AudioFrame(**base, sample_format="int16")

    def test_inference_result_rejects_unknown_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported inference result contract"):
            InferenceResult(
                source_id=1,
                engine_id="yamnet",
                stage_id="yamnet-mlp",
                timestamp="2026-10-21T00:00:00Z",
                scores={},
                metadata={},
                contract="inference-result.v2",
            )

    def test_legacy_event_is_upgraded_and_versioned_event_is_validated(self) -> None:
        event = normalize_detector_event({
            "type": "event", "event": "siren", "confidence": 0.7,
            "state": "confirmed", "engine": "yamnet", "source_id": 2,
            "source": "north-mic", "timestamp": "2026-10-21T10:00:00Z",
        })
        self.assertEqual(event["contract"], "detector-event.v1")
        self.assertEqual(event["engine_id"], "yamnet")
        with self.assertRaisesRegex(ValueError, "unsupported event contract"):
            normalize_detector_event({"type": "event", "contract": "detector-event.v2"})
        with self.assertRaisesRegex(ValueError, "invalid detector-event.v1"):
            normalize_detector_event({
                "type": "event", "event": "siren", "state": "confirmed",
                "confidence": 1.5, "engine_id": "yamnet", "source_id": 1,
            })


if __name__ == "__main__":
    unittest.main()
