#!/usr/bin/env python3
"""Augment clean audio sources with realistic INMP441 conditions.

Takes clean WAV/MP3 files (ESC-50, Freesound, etc.) and creates augmented
training samples by:

1. Varying volume (simulating different distances to sound source)
2. Mixing with real INMP441 background recordings (street, room, etc.)
3. Optionally applying INMP441 noise profile (from silence recording)

Usage:
    # Augment all siren sources with random backgrounds
    python3 dataset/augment.py siren --count 200

    # Augment explosion sources
    python3 dataset/augment.py explosion --count 150

    # Use specific noise profile (recorded silence from INMP441)
    python3 dataset/augment.py siren --count 200 --noise-profile dataset/background/silence.wav

    # Dry run: show what would be created
    python3 dataset/augment.py siren --count 10 --dry-run

Sources are taken from:
    - dataset/external/esc50_curated/<label>/
    - dataset/external/freesound/audio/ (for siren)

Backgrounds are taken from:
    - dataset/background/*.wav (INMP441 recordings)

Output goes to:
    - dataset/<label>/aug_<source>_<variant>.wav

Requires: numpy, soundfile, scipy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal

DATASET = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# Volume range in dB relative to original (simulates distance)
VOLUME_RANGE_DB = (-20, 3)
# Quiet mode: simulates distant/muffled sounds
VOLUME_RANGE_QUIET_DB = (-30, -10)
# Signal-to-noise ratio range in dB
SNR_RANGE_DB = (-3, 20)
# Quiet mode: background is louder relative to signal
SNR_RANGE_QUIET_DB = (-6, 5)


def load_mono_16k(path: Path, max_seconds: float = 300.0) -> np.ndarray:
    """Load audio file as mono float32 at 16 kHz."""
    info = sf.info(path)
    max_frames = int(max_seconds * info.samplerate)
    audio, rate = sf.read(path, frames=max_frames, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio, SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def collect_sources(label: str) -> list[Path]:
    """Collect clean audio sources for a label."""
    sources: list[Path] = []

    # ESC-50 curated
    esc_dir = DATASET / "external" / "esc50_curated" / label
    if esc_dir.exists():
        sources.extend(sorted(esc_dir.glob("*.wav")))

    # Freesound (all files are siren by default per project convention)
    if label == "siren":
        fs_dir = DATASET / "external" / "freesound" / "audio"
        if fs_dir.exists():
            sources.extend(
                sorted(
                    p for p in fs_dir.iterdir()
                    if p.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac")
                )
            )

    # Freesound explosion downloads (converted)
    if label == "explosion":
        # Look for dl_* files in explosion dir (from convert_downloads.py)
        exp_dir = DATASET / "explosion"
        if exp_dir.exists():
            sources.extend(
                sorted(p for p in exp_dir.glob("dl_*.wav"))
            )

    return sources


def collect_backgrounds() -> list[Path]:
    """Collect INMP441 background recordings for mixing."""
    bg_dir = DATASET / "background"
    if not bg_dir.exists():
        return []
    # Only INMP441 recordings (date-stamped filenames)
    return sorted(
        p for p in bg_dir.glob("2026*.wav")
    )


def load_noise_profile(path: Path) -> np.ndarray | None:
    """Load a silence recording to extract noise characteristics."""
    if path is None or not path.exists():
        return None
    audio = load_mono_16k(path, max_seconds=60.0)
    return audio


def apply_volume(audio: np.ndarray, db: float) -> np.ndarray:
    """Apply volume change in dB."""
    gain = 10.0 ** (db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0)


def mix_with_background(
    signal: np.ndarray,
    background: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix signal with background at given SNR (dB)."""
    # Pick a random chunk from background
    if len(background) <= len(signal):
        bg_chunk = np.tile(background, (len(signal) // len(background) + 1))[:len(signal)]
    else:
        start = rng.integers(0, len(background) - len(signal))
        bg_chunk = background[start:start + len(signal)]

    # Compute power and scale
    sig_power = np.mean(signal ** 2) + 1e-10
    bg_power = np.mean(bg_chunk ** 2) + 1e-10
    target_bg_power = sig_power / (10.0 ** (snr_db / 10.0))
    bg_scale = np.sqrt(target_bg_power / bg_power)

    mixed = signal + bg_chunk * bg_scale
    return np.clip(mixed, -1.0, 1.0)


def add_noise_profile(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add microphone noise profile at given SNR."""
    if len(noise) <= len(audio):
        noise_chunk = np.tile(noise, (len(audio) // len(noise) + 1))[:len(audio)]
    else:
        start = rng.integers(0, len(noise) - len(audio))
        noise_chunk = noise[start:start + len(audio)]

    sig_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise_chunk ** 2) + 1e-10
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    result = audio + noise_chunk * noise_scale
    return np.clip(result, -1.0, 1.0)


def extract_chunk(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Extract a random CHUNK_SECONDS chunk from audio."""
    if len(audio) <= CHUNK_SAMPLES:
        # Pad with zeros
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded
    start = rng.integers(0, len(audio) - CHUNK_SAMPLES)
    return audio[start:start + CHUNK_SAMPLES]


def augment_one(
    source_audio: np.ndarray,
    backgrounds: list[np.ndarray],
    noise_profile: np.ndarray | None,
    rng: np.random.Generator,
    quiet: bool = False,
) -> np.ndarray:
    """Create one augmented sample from source audio."""
    # Extract random chunk
    chunk = extract_chunk(source_audio, rng)

    # Random volume change
    vol_range = VOLUME_RANGE_QUIET_DB if quiet else VOLUME_RANGE_DB
    vol_db = rng.uniform(*vol_range)
    chunk = apply_volume(chunk, vol_db)

    # Mix with random background
    if backgrounds:
        bg = backgrounds[rng.integers(len(backgrounds))]
        snr_range = SNR_RANGE_QUIET_DB if quiet else SNR_RANGE_DB
        snr_db = rng.uniform(*snr_range)
        chunk = mix_with_background(chunk, bg, snr_db, rng)

    # Add noise profile if available
    if noise_profile is not None:
        noise_snr = rng.uniform(20, 40)  # noise is subtle
        chunk = add_noise_profile(chunk, noise_profile, noise_snr, rng)

    return chunk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment clean audio with INMP441 conditions"
    )
    parser.add_argument("label", choices=["siren", "explosion", "motorcycle"])
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of augmented samples to generate (default: 100)"
    )
    parser.add_argument(
        "--noise-profile", type=Path, default=None,
        help="Path to INMP441 silence recording for noise profile"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Generate quiet/distant samples (lower volume, worse SNR)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing files"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Collect sources
    sources = collect_sources(args.label)
    if not sources:
        print(f"No source files found for '{args.label}'", file=sys.stderr)
        print("Expected sources in:", file=sys.stderr)
        print(f"  dataset/external/esc50_curated/{args.label}/", file=sys.stderr)
        if args.label == "siren":
            print("  dataset/external/freesound/audio/", file=sys.stderr)
        return 1

    print(f"Sources: {len(sources)} files for '{args.label}'")

    # Collect backgrounds
    bg_paths = collect_backgrounds()
    print(f"Backgrounds: {len(bg_paths)} INMP441 recordings")

    if not bg_paths:
        print("Warning: no background recordings found, augmenting without mixing",
              file=sys.stderr)

    if args.dry_run:
        vol_range = VOLUME_RANGE_QUIET_DB if args.quiet else VOLUME_RANGE_DB
        snr_range = SNR_RANGE_QUIET_DB if args.quiet else SNR_RANGE_DB
        mode = "quiet (distant)" if args.quiet else "normal"
        print(f"\nDry run ({mode}): would create {args.count} augmented files in "
              f"dataset/{args.label}/aug_*.wav")
        print(f"Volume range: {vol_range[0]} to {vol_range[1]} dB")
        print(f"SNR range: {snr_range[0]} to {snr_range[1]} dB")
        return 0

    # Load backgrounds into memory (only CHUNK_SECONDS * 3 per file to save RAM)
    print("\nLoading backgrounds...")
    backgrounds: list[np.ndarray] = []
    for i, bp in enumerate(bg_paths):
        try:
            bg = load_mono_16k(bp, max_seconds=CHUNK_SECONDS * 3)
            if len(bg) >= SAMPLE_RATE:  # at least 1 second
                backgrounds.append(bg)
        except Exception as e:
            print(f"  skip {bp.name}: {e}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(bg_paths)} backgrounds")

    print(f"  {len(backgrounds)} backgrounds loaded")

    # Load noise profile
    noise = load_noise_profile(args.noise_profile)
    if noise is not None:
        print(f"Noise profile: {len(noise) / SAMPLE_RATE:.1f}s from {args.noise_profile.name}")

    # Load all source audio
    print(f"\nLoading {len(sources)} source files...")
    source_audio: list[tuple[str, np.ndarray]] = []
    for sp in sources:
        try:
            audio = load_mono_16k(sp, max_seconds=300.0)
            if len(audio) >= SAMPLE_RATE // 2:  # at least 0.5 seconds
                source_audio.append((sp.stem, audio))
        except Exception as e:
            print(f"  skip {sp.name}: {e}", file=sys.stderr)

    if not source_audio:
        print("No valid source audio loaded", file=sys.stderr)
        return 1

    print(f"  {len(source_audio)} sources loaded")

    # Generate augmented samples
    out_dir = DATASET / args.label
    out_dir.mkdir(exist_ok=True)

    print(f"\nGenerating {args.count} augmented samples...")
    created = 0
    for i in range(args.count):
        # Pick random source
        src_idx = rng.integers(len(source_audio))
        src_name, src_audio = source_audio[src_idx]

        # Generate augmented chunk
        chunk = augment_one(src_audio, backgrounds, noise, rng, quiet=args.quiet)

        # Deterministic name from source + variant
        variant_hash = hashlib.md5(f"{src_name}_{i}_{args.seed}".encode()).hexdigest()[:8]
        prefix = "aug_q" if args.quiet else "aug"
        out_name = f"{prefix}_{src_name}_{variant_hash}.wav"
        out_path = out_dir / out_name

        sf.write(str(out_path), chunk, SAMPLE_RATE, subtype="PCM_16")
        created += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.count} created")

    print(f"\nDone: {created} augmented files saved to dataset/{args.label}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
