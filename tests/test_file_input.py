from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from detector.transport.file import FileAudioAdapter


class FileAudioAdapterTests(unittest.TestCase):
    def test_reads_mono_wav_as_sequenced_audio_frames(self) -> None:
        samples = np.linspace(-0.5, 0.5, 2500, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, samples, 16_000, subtype="FLOAT")
            frames = []
            done = threading.Event()

            def receive(frame):
                frames.append(frame)
                if len(frames) == 3:
                    done.set()

            adapter = FileAudioAdapter(path, source_id=9, chunk_samples=1000)
            adapter.start(receive)
            self.assertTrue(done.wait(2))
            adapter.stop()

        self.assertEqual([frame.sequence for frame in frames], [0, 1, 2])
        self.assertEqual([len(frame.samples) for frame in frames], [1000, 1000, 500])
        self.assertTrue(all(frame.source_id == 9 for frame in frames))
        np.testing.assert_allclose(np.concatenate([f.samples for f in frames]), samples)

    def test_rejects_unsupported_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-rate.wav"
            sf.write(path, np.zeros(100, dtype=np.float32), 8_000)
            with self.assertRaisesRegex(ValueError, "16000 Hz"):
                FileAudioAdapter(path).start(lambda _frame: None)


if __name__ == "__main__":
    unittest.main()
