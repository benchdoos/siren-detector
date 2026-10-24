"""Regression coverage for persisted background calibrations."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from detector.bus import EventBus
from detector.config import SAMPLE_RATE
from detector.server import routes_api


class CalibrationPersistenceTests(unittest.TestCase):
    def test_finish_calibration_publishes_a_complete_wav(self) -> None:
        bus = EventBus()
        bus.source_names[7] = "usb-mic"

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch("detector.bus.DATASET_DIR", Path(temp_dir)):
            self.assertTrue(bus.start_calibration([7], 10))
            bus.broadcast_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), source_id=7)

            paths = bus.finish_calibration()

            saved = Path(paths[7])
            self.assertTrue(saved.is_file())
            self.assertEqual(list(saved.parent.iterdir()), [saved])
            info = sf.info(saved)
            self.assertEqual(info.samplerate, SAMPLE_RATE)
            self.assertEqual(info.frames, SAMPLE_RATE)

    def test_failed_spec_cleanup_keeps_calibration_wav(self) -> None:
        """Deleting a cache must not leave its recording as an orphan."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            wav = dataset_dir / "background" / "usb-mic" / "calibration.wav"
            wav.parent.mkdir(parents=True)
            sf.write(wav, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            spec = wav.with_suffix(".spec")
            spec.write_text("cached spectrogram", encoding="utf-8")

            original_unlink = Path.unlink

            def fail_only_for_spec(path: Path, *args, **kwargs):
                if path == spec:
                    raise OSError("simulated cache cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(routes_api, "DATASET_DIR", dataset_dir), \
                    patch.object(Path, "unlink", fail_only_for_spec):
                with self.assertRaises(OSError):
                    asyncio.run(routes_api.delete_calibration_recording({
                        "path": "usb-mic/calibration.wav",
                    }))

            self.assertTrue(wav.exists())

    def test_failed_wav_write_keeps_calibration_for_retry(self) -> None:
        bus = EventBus()
        bus.source_names[7] = "usb-mic"

        with tempfile.TemporaryDirectory() as temp_dir, \
                patch("detector.bus.DATASET_DIR", Path(temp_dir)):
            self.assertTrue(bus.start_calibration([7], 10))
            bus.broadcast_audio(np.zeros(SAMPLE_RATE, dtype=np.float32), source_id=7)

            with patch("detector.bus.sf.write", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    bus.finish_calibration()

            self.assertIsNotNone(bus.calibration_status())
            paths = bus.finish_calibration()
            self.assertTrue(Path(paths[7]).is_file())
