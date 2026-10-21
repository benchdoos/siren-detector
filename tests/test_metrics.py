"""Prometheus exposition format tests."""

from __future__ import annotations

import unittest

from detector.runtime.metrics import render_runtime_metrics


class RuntimeMetricsTests(unittest.TestCase):
    def test_runtime_counters_and_labels_are_rendered_and_escaped(self) -> None:
        rendered = render_runtime_metrics({
            "components": [{"component_id": 'yamnet"main', "state": "degraded", "error_count": 2}],
            "subscribers": [{
                "subscriber_id": "mic\\1", "engine_id": "yamnet",
                "frames_received": 12, "frames_processed": 10, "frames_dropped": 2,
                "sequence_gaps": 1, "callback_errors": 3,
                "queue_depth": 4,
            }],
            "outputs": [{
                "adapter_id": "webhook:2", "messages_delivered": 9,
                "failures": 1, "sent": 8, "dropped": 2, "queue_depth": 1,
            }],
        })
        self.assertIn('# TYPE detector_audio_frames_received_total counter', rendered)
        self.assertIn('detector_engine_state{engine_id="yamnet\\"main",state="degraded"} 1', rendered)
        self.assertIn('detector_audio_frames_dropped_total{engine_id="yamnet",subscriber_id="mic\\\\1"} 2', rendered)
        self.assertIn('detector_webhook_deliveries_total{adapter_id="webhook:2"} 8', rendered)
        self.assertTrue(rendered.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
