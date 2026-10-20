#!/usr/bin/env python3
"""
DSP Feature Extractor for microphone-invariant sound event detection.

Extracts 5 features every 20ms (hop=320 samples at 16kHz):
  1. pitch (f0) via YIN autocorrelation
  2. RMS energy (log-scaled)
  3. spectral centroid
  4. harmonic-to-noise ratio (HNR)
  5. zero crossing rate (ZCR)

These features track *relative* changes in the signal and are largely
invariant to microphone frequency response, unlike raw spectrograms.

Usage:
  ml/.venv/bin/python ml/dsp_features.py                    # extract + print stats
  ml/.venv/bin/python ml/dsp_features.py --plot              # visualize features
  ml/.venv/bin/python ml/dsp_features.py --eval              # evaluate rule-based detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"
SAMPLE_RATE = 16_000
FRAME_SIZE = 640      # 40ms window
HOP_SIZE = 320        # 20ms hop → 50 frames/sec
LABELS = ["siren", "background", "explosion", "uav"]

# YIN pitch tracker parameters
YIN_THRESHOLD = 0.15
MIN_F0 = 50.0         # Hz
MAX_F0 = 2000.0       # Hz
MIN_LAG = int(SAMPLE_RATE / MAX_F0)  # 8 samples
MAX_LAG = int(SAMPLE_RATE / MIN_F0)  # 320 samples


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_mono_16k(path: Path, max_seconds: float | None = None) -> np.ndarray:
    info = sf.info(path)
    frames = int(max_seconds * info.samplerate) if max_seconds else -1
    audio, rate = sf.read(path, dtype="float32", always_2d=True, frames=frames)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio, SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


# ── Feature extraction ────────────────────────────────────────────────────────

def _yin_pitch(frame: np.ndarray) -> float:
    """Estimate fundamental frequency using YIN algorithm.

    Returns f0 in Hz, or 0.0 if unvoiced.
    """
    n = len(frame)
    if n < MAX_LAG + 1:
        return 0.0

    # Step 1: difference function
    d = np.zeros(MAX_LAG + 1)
    for tau in range(1, MAX_LAG + 1):
        diff = frame[:n - tau] - frame[tau:n]
        d[tau] = np.sum(diff * diff)

    # Step 2: cumulative mean normalized difference
    d_prime = np.ones(MAX_LAG + 1)
    running_sum = 0.0
    for tau in range(1, MAX_LAG + 1):
        running_sum += d[tau]
        if running_sum == 0:
            d_prime[tau] = 1.0
        else:
            d_prime[tau] = d[tau] * tau / running_sum

    # Step 3: absolute threshold
    for tau in range(MIN_LAG, MAX_LAG + 1):
        if d_prime[tau] < YIN_THRESHOLD:
            # Step 4: parabolic interpolation
            if tau > 0 and tau < MAX_LAG:
                s0 = d_prime[tau - 1]
                s1 = d_prime[tau]
                s2 = d_prime[tau + 1]
                adjustment = 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2 + 1e-10)
                tau_interp = tau + adjustment
            else:
                tau_interp = float(tau)
            if tau_interp > 0:
                return SAMPLE_RATE / tau_interp
            return 0.0

    return 0.0


def _spectral_features(frame: np.ndarray) -> tuple[float, float, float, float]:
    """Compute spectral centroid, bandwidth, flatness, and rolloff.

    All based on spectrum shape — invariant to mic frequency response.
    Returns: (centroid_hz, bandwidth_hz, flatness, rolloff_hz)
    """
    n_fft = len(frame)
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    total = spectrum.sum()

    if total < 1e-10:
        return 0.0, 0.0, 0.0, 0.0

    # Centroid: weighted mean frequency
    centroid = float(np.sum(freqs * spectrum) / total)

    # Bandwidth: weighted std of frequency (spread around centroid)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))

    # Flatness: geometric mean / arithmetic mean of power spectrum
    # High = noise-like, low = tonal
    power = spectrum ** 2 + 1e-20
    log_mean = np.mean(np.log(power))
    flatness = float(np.exp(log_mean) / (np.mean(power) + 1e-10))

    # Rolloff: frequency below which 85% of energy is concentrated
    cumsum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    return centroid, bandwidth, flatness, rolloff


def _harmonic_to_noise_ratio(frame: np.ndarray, f0: float) -> float:
    """Simplified HNR: ratio of harmonic energy to total energy in dB."""
    if f0 < MIN_F0:
        return 0.0
    n_fft = len(frame)
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft))) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)

    # Sum energy near harmonics (within +/- 30 Hz)
    harmonic_energy = 0.0
    tolerance = 30.0  # Hz
    for h in range(1, 11):  # first 10 harmonics
        hf = f0 * h
        if hf > SAMPLE_RATE / 2:
            break
        mask = np.abs(freqs - hf) < tolerance
        harmonic_energy += spectrum[mask].sum()

    total_energy = spectrum.sum()
    if total_energy < 1e-10:
        return 0.0
    noise_energy = total_energy - harmonic_energy
    if noise_energy < 1e-10:
        return 30.0  # cap at 30 dB
    return float(10 * np.log10(harmonic_energy / noise_energy + 1e-10))


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract DSP features from audio.

    Returns array of shape (n_frames, 8):
      [pitch_hz, log_rms, spectral_centroid_hz, hnr_db, zcr,
       spectral_bandwidth_hz, spectral_flatness, spectral_rolloff_hz]
    """
    n_frames = max(0, (len(audio) - FRAME_SIZE) // HOP_SIZE + 1)
    features = np.zeros((n_frames, 8), dtype=np.float32)

    for i in range(n_frames):
        start = i * HOP_SIZE
        frame = audio[start:start + FRAME_SIZE]

        # 1. Pitch (f0)
        f0 = _yin_pitch(frame)
        features[i, 0] = f0

        # 2. RMS energy (log-scaled, floor at -80 dB)
        rms = np.sqrt(np.mean(frame ** 2))
        features[i, 1] = max(20 * np.log10(rms + 1e-10), -80.0)

        # 3-6. Spectral features (centroid, bandwidth, flatness, rolloff)
        centroid, bandwidth, flatness, rolloff = _spectral_features(frame)
        features[i, 2] = centroid
        features[i, 5] = bandwidth
        features[i, 6] = flatness
        features[i, 7] = rolloff

        # 4. HNR
        features[i, 3] = _harmonic_to_noise_ratio(frame, f0)

        # 5. ZCR (zero crossing rate)
        features[i, 4] = np.mean(np.abs(np.diff(np.sign(frame))) > 0)

    return features


# ── Rule-based detectors ──────────────────────────────────────────────────────

def detect_siren_rules(features: np.ndarray) -> dict:
    """Detect siren by periodic oscillation in spectral centroid.

    Uses autocorrelation to find periodic components in the 0.25-2 Hz range
    (siren sweep rate). This is more robust than counting direction changes
    because autocorrelation naturally averages over the whole signal and
    distinguishes periodic sweeps from random noise fluctuations.

    Returns dict with 'detected', 'confidence', 'details'.
    """
    energy = features[:, 1]
    centroid = features[:, 2]
    pitch = features[:, 0]

    # Only consider frames with reasonable energy (above noise floor)
    active = energy > -55
    if active.sum() < 50:  # need at least ~1s
        return {"detected": False, "confidence": 0.0, "details": "too few active frames"}

    # Use centroid values for active frames, fill gaps with mean
    c = centroid.copy()
    c_mean = centroid[active].mean()
    c[~active] = c_mean

    # Smooth to reduce frame-level noise
    if len(c) > 11:
        c = scipy.signal.medfilt(c, kernel_size=11)

    # Zero-mean for autocorrelation
    c = c - c.mean()
    c_std = c.std()
    if c_std < 50:  # centroid barely varies — no sweep
        return {"detected": False, "confidence": 0.0, "details": f"centroid std={c_std:.0f} Hz (too low)"}

    # Autocorrelation of centroid signal
    # Look for peaks in the 0.25-2 Hz range (period 0.5-4 seconds)
    # At 50 frames/sec: lag 25-200 frames
    fps = SAMPLE_RATE / HOP_SIZE  # 50 fps
    min_lag = int(0.5 * fps)   # 25 frames = 0.5s period
    max_lag = int(4.0 * fps)   # 200 frames = 4.0s period
    max_lag = min(max_lag, len(c) // 2)

    if max_lag <= min_lag:
        return {"detected": False, "confidence": 0.0, "details": "clip too short for autocorr"}

    # Normalized autocorrelation
    acf = np.correlate(c, c, mode='full')
    acf = acf[len(c) - 1:]  # take positive lags only
    acf = acf / (acf[0] + 1e-10)  # normalize

    # Find the strongest peak in the siren frequency range
    search_region = acf[min_lag:max_lag + 1]
    if len(search_region) == 0:
        return {"detected": False, "confidence": 0.0, "details": "no search region"}

    peak_idx = np.argmax(search_region)
    peak_val = search_region[peak_idx]
    peak_lag = peak_idx + min_lag
    peak_freq = fps / peak_lag  # Hz (oscillation frequency)

    # Also check for pitch content (bonus)
    voiced = (pitch > 100) & (pitch < 1800) & active
    voiced_ratio = voiced.sum() / max(active.sum(), 1)

    # Detection: strong periodic peak in centroid autocorrelation
    # peak_val > 0.3 means strong periodicity; real sirens typically > 0.4
    threshold = 0.25 if voiced_ratio > 0.15 else 0.35

    if peak_val > threshold:
        confidence = min(1.0, (peak_val - 0.2) / 0.5)
        if voiced_ratio > 0.15:
            confidence = min(1.0, confidence * 1.3)
        return {
            "detected": True,
            "confidence": round(confidence, 3),
            "details": f"acf_peak={peak_val:.2f} freq={peak_freq:.2f}Hz voiced={voiced_ratio:.2f} c_std={c_std:.0f}",
        }

    return {"detected": False, "confidence": 0.0,
            "details": f"acf_peak={peak_val:.2f} freq={peak_freq:.2f}Hz voiced={voiced_ratio:.2f} c_std={c_std:.0f}"}


def detect_explosion_rules(features: np.ndarray) -> dict:
    """Detect explosion by looking for sudden energy spike + slow decay."""
    energy = features[:, 1]
    centroid = features[:, 2]

    if len(energy) < 10:
        return {"detected": False, "confidence": 0.0, "details": "too short"}

    # Look for sudden energy increase
    energy_diff = np.diff(energy)
    max_rise_idx = np.argmax(energy_diff)
    max_rise = energy_diff[max_rise_idx]

    if max_rise < 10:  # need at least 10 dB jump
        return {"detected": False, "confidence": 0.0, "details": f"max rise {max_rise:.1f} dB"}

    # Check decay after spike
    peak_idx = max_rise_idx + 1
    if peak_idx + 10 >= len(energy):
        return {"detected": False, "confidence": 0.0, "details": "spike at end of clip"}

    peak_energy = energy[peak_idx]
    decay_window = energy[peak_idx:min(peak_idx + 50, len(energy))]  # ~1s window
    decay_amount = peak_energy - decay_window[-1]

    # High spectral centroid at spike = broadband
    spike_centroid = centroid[peak_idx] if peak_idx < len(centroid) else 0

    if max_rise > 10 and decay_amount > 5 and spike_centroid > 600:
        confidence = min(1.0, max_rise / 20) * min(1.0, decay_amount / 10)
        return {
            "detected": True,
            "confidence": round(confidence, 3),
            "details": f"rise {max_rise:.1f} dB, decay {decay_amount:.1f} dB, centroid {spike_centroid:.0f} Hz",
        }

    return {"detected": False, "confidence": 0.0, "details": f"rise={max_rise:.1f} decay={decay_amount:.1f} centroid={spike_centroid:.0f}"}


def detect_uav_rules(features: np.ndarray) -> dict:
    """Detect uav/UAV by sustained energy + stable spectral shape.

    Through INMP441 mic, motor sounds appear as sustained elevated energy
    with relatively stable spectral centroid — not necessarily tonal enough
    for YIN pitch detection. So we use energy + centroid stability instead.
    """
    energy = features[:, 1]
    centroid = features[:, 2]
    pitch = features[:, 0]

    if len(energy) < 20:
        return {"detected": False, "confidence": 0.0, "details": "too short"}

    # Sustained elevated energy: above background noise floor
    active = energy > -50  # frames with non-trivial energy
    active_ratio = active.sum() / len(energy)

    if active_ratio < 0.3:
        return {"detected": False, "confidence": 0.0, "details": f"active ratio {active_ratio:.2f}"}

    # Energy stability: low variance = steady motor hum (not speech, not explosion)
    active_energy = energy[active]
    energy_std = np.std(active_energy)
    energy_mean = np.mean(active_energy)

    # Spectral centroid stability: motor sounds have consistent spectral shape
    active_centroid = centroid[active]
    centroid_mean = np.mean(active_centroid)
    centroid_cv = np.std(active_centroid) / max(centroid_mean, 1)

    # Also check if there's any pitch detected (bonus, not required)
    voiced = (pitch > 50) & (pitch < 500) & active
    voiced_ratio = voiced.sum() / max(active.sum(), 1)

    # Motor: sustained LOUD energy, very stable spectrum
    # Energy must be elevated (motors are loud) and very steady
    is_loud = energy_mean > -42  # motors are louder than ambient
    is_steady = energy_std < 4
    is_spectrally_stable = centroid_cv < 0.35
    is_motor_range = 500 < centroid_mean < 2500
    is_sustained = active_ratio > 0.5

    if is_loud and is_steady and is_spectrally_stable and is_motor_range and is_sustained:
        confidence = min(1.0, active_ratio) * min(1.0, (6 - energy_std) / 4)
        if voiced_ratio > 0.1:  # bonus for pitch detection
            confidence = min(1.0, confidence * 1.3)
        return {
            "detected": True,
            "confidence": round(max(0, confidence), 3),
            "details": f"E_std={energy_std:.1f} C_cv={centroid_cv:.2f} C={centroid_mean:.0f}Hz active={active_ratio:.2f} voiced={voiced_ratio:.2f}",
        }

    return {
        "detected": False,
        "confidence": 0.0,
        "details": f"E_std={energy_std:.1f} C_cv={centroid_cv:.2f} C={centroid_mean:.0f}Hz active={active_ratio:.2f}",
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_rules(max_files: int = 50) -> None:
    """Run rule-based detectors on dataset and report accuracy."""
    detectors = {
        "siren": detect_siren_rules,
        "explosion": detect_explosion_rules,
        "uav": detect_uav_rules,
    }

    results: dict[str, dict[str, int]] = {}
    for label in LABELS:
        results[label] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}

    for label in LABELS:
        label_dir = DATASET / label
        if not label_dir.exists():
            continue
        files = sorted(label_dir.rglob("*.wav"))[:max_files]
        if not files:
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} ({len(files)} files)")
        print(f"{'='*60}")

        for f in files:
            audio = load_mono_16k(f, max_seconds=10)
            feats = extract_features(audio)

            for det_name, det_fn in detectors.items():
                result = det_fn(feats)
                is_positive = label == det_name
                detected = result["detected"]

                if is_positive and detected:
                    results[det_name]["tp"] += 1
                elif is_positive and not detected:
                    results[det_name]["fn"] += 1
                elif not is_positive and detected:
                    results[det_name]["fp"] += 1
                else:
                    results[det_name]["tn"] += 1

            results[label]["total"] += 1

            # Print per-file details for target class
            for det_name, det_fn in detectors.items():
                r = det_fn(feats)
                if label == det_name or r["detected"]:
                    tag = "HIT" if r["detected"] and label == det_name else \
                          "MISS" if not r["detected"] and label == det_name else \
                          "FP!" if r["detected"] and label != det_name else ""
                    print(f"  [{tag:4s}] {f.name:50s} {det_name}: {r['details']}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for det_name in detectors:
        r = results[det_name]
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        print(f"  {det_name:12s}  TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  "
              f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")


def print_stats(max_files: int = 10) -> None:
    """Print feature statistics per class."""
    for label in LABELS:
        label_dir = DATASET / label
        if not label_dir.exists():
            continue
        files = sorted(label_dir.rglob("*.wav"))[:max_files]
        if not files:
            continue

        all_feats = []
        for f in files:
            audio = load_mono_16k(f, max_seconds=10)
            feats = extract_features(audio)
            all_feats.append(feats)

        feats = np.vstack(all_feats)
        voiced = feats[:, 0] > MIN_F0
        names = ["pitch(Hz)", "logRMS(dB)", "centroid(Hz)", "HNR(dB)", "ZCR",
                 "bandwidth(Hz)", "flatness", "rolloff(Hz)"]

        print(f"\n{label.upper()} ({len(files)} files, {len(feats)} frames, {voiced.sum()} voiced)")
        for i, name in enumerate(names):
            vals = feats[:, i]
            if i == 0:  # pitch — only voiced frames
                vals = vals[voiced] if voiced.any() else vals
            print(f"  {name:15s}  mean={vals.mean():8.1f}  std={vals.std():7.1f}  "
                  f"min={vals.min():8.1f}  max={vals.max():8.1f}")


def plot_features(wav_path: str | None = None) -> None:
    """Plot features for one file per class (or a specific file)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    if wav_path:
        audio = load_mono_16k(Path(wav_path), max_seconds=10)
        feats = extract_features(audio)
        _plot_single(feats, Path(wav_path).name, plt)
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(len(LABELS), 5, figsize=(20, 3 * len(LABELS)))
    names = ["pitch (Hz)", "log RMS (dB)", "centroid (Hz)", "HNR (dB)", "ZCR"]

    for row, label in enumerate(LABELS):
        label_dir = DATASET / label
        files = sorted(label_dir.rglob("*.wav"))
        # Pick a representative file (not augmented, not too short)
        picked = None
        for f in files:
            if not f.name.startswith("aug_") and f.stat().st_size > 50_000:
                picked = f
                break
        if picked is None and files:
            picked = files[0]
        if picked is None:
            continue

        audio = load_mono_16k(picked, max_seconds=10)
        feats = extract_features(audio)
        t = np.arange(len(feats)) * HOP_SIZE / SAMPLE_RATE

        for col in range(5):
            ax = axes[row, col]
            ax.plot(t, feats[:, col], linewidth=0.5)
            if row == 0:
                ax.set_title(names[col])
            if col == 0:
                ax.set_ylabel(label.upper())
            ax.set_xlabel("time (s)")

    fig.suptitle("DSP Features per Class", fontsize=14)
    plt.tight_layout()
    out = REPO_ROOT / "ml" / "results" / "dsp_features.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


def _plot_single(feats, title, plt):
    names = ["pitch (Hz)", "log RMS (dB)", "centroid (Hz)", "HNR (dB)", "ZCR"]
    t = np.arange(len(feats)) * HOP_SIZE / SAMPLE_RATE
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(t, feats[:, i], linewidth=0.5)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)


def main():
    parser = argparse.ArgumentParser(description="DSP Feature Extractor")
    parser.add_argument("--plot", action="store_true", help="Plot features (requires matplotlib)")
    parser.add_argument("--eval", action="store_true", help="Evaluate rule-based detectors")
    parser.add_argument("--file", type=str, help="Specific WAV file to analyze")
    parser.add_argument("--max-files", type=int, default=50, help="Max files per class for eval")
    args = parser.parse_args()

    if args.plot:
        plot_features(args.file)
    elif args.eval:
        evaluate_rules(args.max_files)
    elif args.file:
        audio = load_mono_16k(Path(args.file), max_seconds=10)
        feats = extract_features(audio)
        print(f"File: {args.file}")
        print(f"Frames: {len(feats)} ({len(feats) * HOP_SIZE / SAMPLE_RATE:.1f}s)")
        print(f"\nSiren:      {detect_siren_rules(feats)}")
        print(f"Explosion:  {detect_explosion_rules(feats)}")
        print(f"UAVs: {detect_uav_rules(feats)}")
    else:
        print_stats()


if __name__ == "__main__":
    main()
