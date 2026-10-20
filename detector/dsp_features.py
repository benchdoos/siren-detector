"""DSP feature extraction for microphone-invariant sound detection.

Stripped-down runtime version of ml/dsp_features.py — numpy only, no CLI.

Extracts 8 features per frame (40ms window, 20ms hop = 50 fps):
  pitch_hz, log_rms, spectral_centroid_hz, hnr_db, zcr,
  spectral_bandwidth_hz, spectral_flatness, spectral_rolloff_hz
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SIZE = 640      # 40ms window
HOP_SIZE = 320        # 20ms hop -> 50 fps
N_RAW_FEATURES = 8

# YIN pitch tracker
YIN_THRESHOLD = 0.15
MIN_F0 = 50.0
MAX_F0 = 2000.0
MIN_LAG = int(SAMPLE_RATE / MAX_F0)   # 8
MAX_LAG = int(SAMPLE_RATE / MIN_F0)   # 320


def _yin_pitch(frame: np.ndarray) -> float:
    n = len(frame)
    if n < MAX_LAG + 1:
        return 0.0

    d = np.zeros(MAX_LAG + 1)
    for tau in range(1, MAX_LAG + 1):
        diff = frame[:n - tau] - frame[tau:n]
        d[tau] = np.sum(diff * diff)

    d_prime = np.ones(MAX_LAG + 1)
    running_sum = 0.0
    for tau in range(1, MAX_LAG + 1):
        running_sum += d[tau]
        if running_sum == 0:
            d_prime[tau] = 1.0
        else:
            d_prime[tau] = d[tau] * tau / running_sum

    for tau in range(MIN_LAG, MAX_LAG + 1):
        if d_prime[tau] < YIN_THRESHOLD:
            if 0 < tau < MAX_LAG:
                s0, s1, s2 = d_prime[tau - 1], d_prime[tau], d_prime[tau + 1]
                adjustment = 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2 + 1e-10)
                tau_interp = tau + adjustment
            else:
                tau_interp = float(tau)
            return SAMPLE_RATE / tau_interp if tau_interp > 0 else 0.0

    return 0.0


def _spectral_features(frame: np.ndarray) -> tuple[float, float, float, float]:
    n_fft = len(frame)
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    total = spectrum.sum()

    if total < 1e-10:
        return 0.0, 0.0, 0.0, 0.0

    centroid = float(np.sum(freqs * spectrum) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))

    power = spectrum ** 2 + 1e-20
    log_mean = np.mean(np.log(power))
    flatness = float(np.exp(log_mean) / (np.mean(power) + 1e-10))

    cumsum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    return centroid, bandwidth, flatness, rolloff


def _hnr(frame: np.ndarray, f0: float) -> float:
    if f0 < MIN_F0:
        return 0.0
    n_fft = len(frame)
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft))) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)

    harmonic_energy = 0.0
    tolerance = 30.0
    for h in range(1, 11):
        hf = f0 * h
        if hf > SAMPLE_RATE / 2:
            break
        harmonic_energy += spectrum[np.abs(freqs - hf) < tolerance].sum()

    total_energy = spectrum.sum()
    if total_energy < 1e-10:
        return 0.0
    noise_energy = total_energy - harmonic_energy
    if noise_energy < 1e-10:
        return 30.0
    return float(10 * np.log10(harmonic_energy / noise_energy + 1e-10))


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract DSP features from float32 mono 16kHz audio.

    Returns (n_frames, 8) float32 array.
    """
    n_frames = max(0, (len(audio) - FRAME_SIZE) // HOP_SIZE + 1)
    features = np.zeros((n_frames, N_RAW_FEATURES), dtype=np.float32)

    for i in range(n_frames):
        start = i * HOP_SIZE
        frame = audio[start:start + FRAME_SIZE]

        f0 = _yin_pitch(frame)
        features[i, 0] = f0
        rms = np.sqrt(np.mean(frame ** 2))
        features[i, 1] = max(20 * np.log10(rms + 1e-10), -80.0)

        centroid, bandwidth, flatness, rolloff = _spectral_features(frame)
        features[i, 2] = centroid
        features[i, 5] = bandwidth
        features[i, 6] = flatness
        features[i, 7] = rolloff

        features[i, 3] = _hnr(frame, f0)
        features[i, 4] = np.mean(np.abs(np.diff(np.sign(frame))) > 0)

    return features


def add_deltas(features: np.ndarray) -> np.ndarray:
    """Append delta (rate-of-change) features: (N, 8) -> (N, 16)."""
    delta = np.diff(features, axis=0, prepend=features[:1])
    return np.concatenate([features, delta], axis=1)
