"""YAMNet + MLP detection engine.

Implements ``EngineInterface``: receives audio via UDP, runs YAMNet
embeddings + optional MLP classifier, pushes scores and events through
the RuntimeHub.

Supports multiple audio sources (multi-microphone).  Each source gets
its own buffers and smoothers; YAMNet + MLP are loaded once and shared.

Detection modes:
- **round_robin** — single thread cycles through sources sequentially.
- **parallel** — one inference thread per source, running simultaneously.
  TensorFlow releases the GIL during inference, so threads get real
  parallelism on the TF/numpy portions.  CPU load scales with source count.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow_hub as hub

from ..audio import AudioRingBuffer, ClipBuffer
from ..runtime.contracts import AudioFrame, RuntimeMessage
from ..runtime.hub import RuntimeHub
from ..config import (
    ANALYSIS_SECONDS,
    CLIP_DIR,
    CLIP_SECONDS,
    DETECTORS,
    DSP_LABELS_PATH,
    DSP_MODEL_PATH,
    DSP_NORM_STATS_PATH,
    MODEL_URL,
    SAMPLE_RATE,
)
from ..server.routes_clips import save_clip_meta
from ..transport.udp import udp_stats_snapshot
from .smoother import TemporalSmoother
from .source_state import SourceState
from .. import settings as settings_mod

_EMPTY_PEAK = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": "", "yamnet_top3": [], "mlp_top3": []}

# ── Click removal (impulse noise suppression) ─────────────────────────
# MAD-based outlier detection: finds samples that deviate > threshold * MAD
# from the median and replaces short groups (≤ max_width) with linear
# interpolation.  Real explosions span hundreds of samples and are untouched.

def _remove_clicks(waveform: np.ndarray, threshold: float = 10.0,
                   max_width: int = 5) -> tuple[np.ndarray, int]:
    """Remove impulse noise (clicks) from *waveform* in-place style.

    Returns (cleaned_waveform, clicks_removed).
    """
    if len(waveform) < 10:
        return waveform, 0

    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))
    if mad < 1e-10:
        # Nearly-silent signal — use absolute threshold instead
        mad = 1e-5

    # 1.4826 scales MAD to be consistent with std for normal distributions
    cutoff = threshold * mad * 1.4826
    outliers = np.abs(waveform - median) > cutoff

    if not np.any(outliers):
        return waveform, 0

    result = waveform.copy()
    indices = np.where(outliers)[0]

    # Split into groups of consecutive indices
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)

    clicks_removed = 0
    for group in groups:
        if len(group) == 0 or len(group) > max_width:
            continue
        start = int(group[0])
        end = int(group[-1])
        left = float(result[start - 1]) if start > 0 else 0.0
        right = float(result[end + 1]) if end + 1 < len(result) else 0.0
        n = end - start + 1
        for i, idx in enumerate(group):
            result[idx] = left + (right - left) * (i + 1) / (n + 1)
        clicks_removed += 1

    return result, clicks_removed


def _inference_hop_samples() -> int:
    interval = float(settings_mod.settings.get("inference_interval_seconds", 0.5))
    return int(SAMPLE_RATE * min(2.0, max(0.25, interval)))


def _dsp_stats_snapshot(dsp, sources: dict) -> dict | None:
    if dsp is None:
        return None
    stats = dsp.stats_snapshot()
    disabled = set(settings_mod.settings.get("disabled_sources", []))
    active_source_ids = set(sources) - disabled
    overrides = settings_mod.settings.get("dsp_source_overrides", {})
    dsp_source_ids = {
        sid for sid in active_source_ids
        if bool(overrides.get(str(sid), {}).get(
            "dsp_enabled", settings_mod.settings.get("dsp_enabled", 1),
        ))
    }
    skipping_ids = set(stats.pop("skipping_source_ids", []))
    stats["source_count"] = len(dsp_source_ids)
    stats["active_sources"] = len(dsp_source_ids & skipping_ids)
    stats["enabled"] = bool(dsp_source_ids)
    return stats


def _create_source(source_id: int, name: str) -> SourceState:
    """Create a fresh SourceState with buffers and smoothers from current settings."""
    s = settings_mod.settings
    inference_buf = AudioRingBuffer(capacity=int(ANALYSIS_SECONDS * SAMPLE_RATE))
    clip_buf = ClipBuffer(seconds=max(
        s.get("clip_seconds", CLIP_SECONDS), s.get("user_event_clip_seconds", 15),
    ))

    smoothers: dict[str, TemporalSmoother] = {}
    inference_interval = min(2.0, max(0.25, float(s.get("inference_interval_seconds", 0.5))))
    for det_name in DETECTORS:
        thr_key = f"{det_name}_threshold"
        smoothers[det_name] = TemporalSmoother(
            window_seconds=s["smoothing_window_seconds"],
            hop_seconds=inference_interval,
            threshold=s.get(thr_key, 0.15),
            possible=s["possible_triggers"],
            confirmed=s["confirmed_triggers"],
            cooldown_seconds=s["cooldown_seconds"],
            refractory_seconds=s["refractory_seconds"],
        )

    return SourceState(
        source_id=source_id,
        name=name,
        inference_buf=inference_buf,
        clip_buf=clip_buf,
        smoothers=smoothers,
        peaks={n: dict(_EMPTY_PEAK) for n in DETECTORS},
        possible_start={n: None for n in DETECTORS},
        window_scores={n: [] for n in DETECTORS},
    )


class YAMNetEngine:
    """YAMNet + optional MLP classifier engine."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Register YAMNet-specific CLI arguments."""
        parser.add_argument("--classifier-weights", default="/models/classifier_weights.npz",
                            help="Path to MLP classifier weights (.npz)")
        parser.add_argument("--classifier-labels", default="/models/classifier_labels.json",
                            help="Path to MLP classifier labels (.json)")
        parser.add_argument(
            "--source-name", action="append", default=None,
            help="Map source_id to name, e.g. '0=esp32-main' (repeatable)",
        )
        parser.add_argument("--dsp-model", default=DSP_MODEL_PATH,
                            help="Path to DSP TFLite model (.tflite)")
        parser.add_argument("--dsp-norm-stats", default=DSP_NORM_STATS_PATH,
                            help="Path to DSP normalisation stats (.npz)")
        parser.add_argument("--dsp-labels", default=DSP_LABELS_PATH,
                            help="Path to DSP classifier labels (.json)")
        parser.add_argument("--no-dsp", action="store_true",
                            help="Disable DSP pre-filter even if model files exist")

    def start(self, runtime: RuntimeHub, settings: dict, args: argparse.Namespace) -> None:
        # Keep the current bus for source/clip state and live audio; scores and
        # events go through RuntimeHub to the output layer.
        bus = runtime.event_bus
        settings_mod.settings.update(settings)
        print(f"Settings: {settings_mod.settings}", flush=True)

        # Parse source name mappings
        source_names: dict[int, str] = {}
        if args.source_name:
            for mapping in args.source_name:
                if "=" in mapping:
                    sid_str, name = mapping.split("=", 1)
                    source_names[int(sid_str)] = name
        # Default source 0 gets --source name
        source_names.setdefault(0, args.source)

        # ── Load YAMNet (shared across all sources) ──────────────────────
        print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
        yamnet = hub.load(MODEL_URL)

        with open(yamnet.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
            class_names = [row["display_name"] for row in csv.DictReader(f)]

        detector_indices: dict[str, list[int]] = {}
        for name, (yamnet_classes, _, _) in DETECTORS.items():
            indices = [i for i, n in enumerate(class_names) if n in yamnet_classes]
            detector_indices[name] = indices
            print(f"Detector '{name}' YAMNet classes: {[class_names[i] for i in indices]}", flush=True)

        # ── Load MLP classifier (shared) ─────────────────────────────────
        clf = None
        det_engine = "yamnet"
        weights_path = Path(args.classifier_weights)
        labels_path = Path(args.classifier_labels)
        if weights_path.exists() and labels_path.exists():
            from ..classifier_infer import SirenClassifier
            clf = SirenClassifier(weights_path, labels_path)
            det_engine = "yamnet+mlp"
            print(f"Classifier loaded: {clf.labels}", flush=True)
        else:
            print("No classifier weights — using raw YAMNet siren score.", flush=True)

        # ── Load DSP pre-filter (optional cascade) ──────────────────────
        dsp = None
        if not args.no_dsp:
            dsp_model = Path(args.dsp_model)
            dsp_stats = Path(args.dsp_norm_stats)
            dsp_labels = Path(args.dsp_labels)
            if dsp_model.exists() and dsp_stats.exists() and dsp_labels.exists():
                from ..dsp_infer import DspPreFilter
                dsp = DspPreFilter(dsp_model, dsp_stats, dsp_labels)
                det_engine = f"dsp+{det_engine}"
                print(f"[dsp] Pre-filter enabled (cascade mode)", flush=True)
            else:
                missing = [p.name for p in (dsp_model, dsp_stats, dsp_labels) if not p.exists()]
                print(f"[dsp] Pre-filter disabled — missing: {missing}", flush=True)

        # ── Source registry ───────────────────────────────────────────────
        sources: dict[int, SourceState] = {}
        sources_lock = threading.Lock()

        def _sync_smoothers() -> None:
            """Collect all smoothers from all sources into settings_mod."""
            all_smoothers: dict[str, TemporalSmoother] = {}
            for src in sources.values():
                for det_name, sm in src.smoothers.items():
                    key = f"{src.source_id}/{det_name}" if len(sources) > 1 else det_name
                    all_smoothers[key] = sm
            settings_mod.smoothers = all_smoothers

        def get_source(source_id: int) -> SourceState:
            """Get or register per-source processing state."""
            if source_id in sources:
                return sources[source_id]
            with sources_lock:
                if source_id in sources:
                    return sources[source_id]
                name = source_names.get(source_id, f"source-{source_id}")
                s = _create_source(source_id, name)
                sources[source_id] = s
                bus.clip_bufs[source_id] = s.clip_buf
                bus.source_names[source_id] = name
                _sync_smoothers()
                print(f"[multi-source] source {source_id} ({name}) registered", flush=True)
            return s

        # Pre-create default source (id=0)
        get_source(0)

        # The input adapter emits normalized frames; the runtime routes them
        # to this processing chain and the live-audio output.
        def on_audio_frame(frame: AudioFrame) -> None:
            source = get_source(frame.source_id)
            source.inference_buf.push(frame.samples)
            source.clip_buf.push(frame.samples)

        runtime.subscribe_audio(on_audio_frame)

        print("Ready.", flush=True)

        # ── Unified inference loop with hot-switchable mode ──────────────
        _last_heartbeat = 0.0
        _parallel_stop = threading.Event()
        _parallel_threads: dict[int, threading.Thread] = {}
        _current_mode = ""

        def _source_loop(source: SourceState, stop: threading.Event):
            """Inference loop for a single source (used in parallel mode)."""
            while not stop.is_set():
                disabled = settings_mod.settings.get("disabled_sources", [])
                waveform = source.inference_buf.consume(_inference_hop_samples())
                if waveform is None:
                    stop.wait(0.05)
                    continue
                if source.source_id in disabled:
                    continue
                self._process_source(
                    source, waveform, yamnet, class_names, detector_indices,
                    clf, det_engine, runtime, args, sources, dsp,
                )

        def _stop_parallel():
            nonlocal _parallel_threads
            if _parallel_threads:
                _parallel_stop.set()
                for t in _parallel_threads.values():
                    t.join(timeout=2.0)
                _parallel_threads.clear()
                _parallel_stop.clear()
                print("[mode] parallel threads stopped", flush=True)

        def _ensure_parallel():
            with sources_lock:
                for sid, source in sources.items():
                    if sid not in _parallel_threads:
                        t = threading.Thread(
                            target=_source_loop, args=(source, _parallel_stop),
                            daemon=True, name=f"infer-{source.name}",
                        )
                        t.start()
                        _parallel_threads[sid] = t
                        print(f"[parallel] thread started: {source.name} (id={sid})", flush=True)

        while True:
            mode = settings_mod.settings.get("detection_mode", "round_robin")

            if mode != _current_mode:
                if _current_mode == "parallel":
                    _stop_parallel()
                _current_mode = mode
                print(f"[mode] switched to {mode}", flush=True)

            if mode == "parallel":
                _ensure_parallel()
                # Main thread: heartbeat only
                now = time.monotonic()
                if now - _last_heartbeat >= 1.0:
                    heartbeat = self._build_heartbeat(sources)
                    heartbeat.update(udp_stats_snapshot(bus.source_names))
                    if dsp is not None:
                        heartbeat["dsp_stats"] = _dsp_stats_snapshot(dsp, sources)
                    runtime.publish(RuntimeMessage.from_legacy(heartbeat))
                    _last_heartbeat = now
                time.sleep(0.5)
            else:
                # Round-robin: process all sources on main thread
                any_data = False
                disabled = settings_mod.settings.get("disabled_sources", [])
                for source in list(sources.values()):
                    waveform = source.inference_buf.consume(_inference_hop_samples())
                    if waveform is None:
                        continue
                    any_data = True
                    if source.source_id in disabled:
                        continue
                    self._process_source(
                        source, waveform, yamnet, class_names, detector_indices,
                        clf, det_engine, runtime, args, sources, dsp,
                    )

                now = time.monotonic()
                if not any_data:
                    time.sleep(0.05)
                    if now - _last_heartbeat >= 1.0:
                        heartbeat = self._build_heartbeat(sources)
                        heartbeat.update(udp_stats_snapshot(bus.source_names))
                        if dsp is not None:
                            heartbeat["dsp_stats"] = _dsp_stats_snapshot(dsp, sources)
                        runtime.publish(RuntimeMessage.from_legacy(heartbeat))
                        _last_heartbeat = now
                else:
                    _last_heartbeat = now

    # ── Per-source inference ─────────────────────────────────────────────

    @staticmethod
    def _process_source(
        source: SourceState,
        waveform: np.ndarray,
        yamnet,
        class_names: list[str],
        detector_indices: dict[str, list[int]],
        clf,
        det_engine: str,
        runtime: RuntimeHub,
        args: argparse.Namespace,
        sources: dict | None = None,
        dsp=None,
    ) -> None:
        waveform = np.clip(waveform, -1.0, 1.0)

        # ── Click removal: suppress impulse noise before inference ──────
        cr_override = settings_mod.settings.get(
            "click_removal_source_overrides", {},
        ).get(str(source.source_id), {})
        cr_enabled = cr_override.get(
            "enabled", settings_mod.settings.get("click_removal_enabled", 1),
        )
        if cr_enabled:
            cr_threshold = cr_override.get(
                "threshold",
                settings_mod.settings.get("click_removal_threshold", 10.0),
            )
            cr_max_width = cr_override.get(
                "max_width",
                settings_mod.settings.get("click_removal_max_width", 5),
            )
            waveform, n_clicks = _remove_clicks(
                waveform, threshold=cr_threshold, max_width=cr_max_width,
            )
            if n_clicks:
                print(
                    f"[click-removal] source {source.source_id}: "
                    f"removed {n_clicks} click(s)",
                    flush=True,
                )

        # ── DSP pre-filter: skip YAMNet for confident background ────────
        dsp_result = None
        dsp_override = settings_mod.settings.get("dsp_source_overrides", {}).get(
            str(source.source_id), {},
        )
        source_dsp_enabled = dsp_override.get(
            "dsp_enabled", settings_mod.settings.get("dsp_enabled", 1),
        )
        if dsp is not None and source_dsp_enabled:
            new_audio = waveform[-_inference_hop_samples():]
            dsp_result = dsp.process(new_audio, source.dsp_feature_buf)

            if dsp_result["ready"]:
                bg_threshold = dsp_override.get(
                    "bg_threshold", settings_mod.settings.get("dsp_bg_threshold", 0.85),
                )
                bg_release_threshold = dsp_override.get(
                    "bg_release_threshold", settings_mod.settings.get("dsp_bg_release_threshold", 0.85),
                )
                bg_confirm_windows = dsp_override.get(
                    "bg_confirm_windows", settings_mod.settings.get("dsp_bg_confirm_windows", 2),
                )
                max_consecutive_skips = dsp_override.get(
                    "max_consecutive_skips",
                    settings_mod.settings.get("dsp_max_consecutive_skips", 20),
                )
                dsp_event_gate = dsp_override.get(
                    "event_gate",
                    settings_mod.settings.get("dsp_event_gate", 0.08),
                )
                dsp_label = dsp_result["label"]
                dsp_conf = dsp_result["confidence"]
                bg_conf = dsp_result["probs"].get("background", 0.0)

                # Event gate: if any event class has notable probability,
                # force YAMNet regardless of background confidence
                event_max = max(
                    (v for k, v in dsp_result["probs"].items() if k != "background"),
                    default=0.0,
                )

                # Keep YAMNet on every window while a detector is possible,
                # so DSP skips cannot decay the trigger count before confirm.
                # Once confirmed, return to the regular DSP cadence and its
                # periodic forced YAMNet passes.
                smoother_active = any(
                    source.smoothers[det].state == "possible"
                    for det in DETECTORS
                )

                # Update gate state (always), then override skip decision
                skip_yamnet = dsp.should_skip_yamnet(
                    source.source_id, dsp_label, bg_conf, bg_threshold,
                    bg_release_threshold, bg_confirm_windows,
                    max_consecutive_skips,
                )
                if smoother_active or (event_max >= dsp_event_gate > 0):
                    skip_yamnet = False
                dsp.record_decision(source.source_id, dsp_result["probs"], skip_yamnet)

                # Periodic DSP diagnostic log (every 10s per source)
                now_mono = time.monotonic()
                if now_mono - source.dsp_last_log >= 10.0:
                    source.dsp_last_log = now_mono
                    probs_str = " ".join(
                        f"{k}={v:.2f}" for k, v in sorted(dsp_result["probs"].items())
                    )
                    override = ""
                    if smoother_active:
                        active = [d for d in DETECTORS if source.smoothers[d].state != "idle"]
                        override = f" OVERRIDE:smoother({','.join(active)})"
                    elif event_max >= dsp_event_gate > 0:
                        override = f" OVERRIDE:event_gate"
                    print(
                        f"[dsp] src={source.source_id} label={dsp_label} "
                        f"conf={dsp_conf:.2f} skip={skip_yamnet} "
                        f"event_max={event_max:.3f}{override} | {probs_str}",
                        flush=True,
                    )

                if skip_yamnet:
                    # Confident background — skip YAMNet, broadcast minimal score
                    score_msg: dict = {
                        "type": "score", "top": "(dsp:background)",
                        "source": source.name, "source_id": source.source_id,
                        "dsp": dsp_result["probs"],
                        "dsp_label": dsp_label,
                        "dsp_confidence": round(dsp_conf, 3),
                        "dsp_skipped_yamnet": True,
                    }
                    for name in DETECTORS:
                        score_msg[f"{name}_score"] = 0.0
                        score_msg[f"{name}_state"] = source.smoothers[name].state
                        score_msg[f"{name}_triggers"] = source.smoothers[name].trigger_count
                        score_msg[f"{name}_possible_needed"] = source.smoothers[name].possible_needed
                        score_msg[f"{name}_confirmed_needed"] = source.smoothers[name].confirmed_needed
                        score_msg[f"{name}_cooldown"] = round(source.smoothers[name].cooldown_remaining, 1)
                        score_msg[f"{name}_refractory"] = round(source.smoothers[name].refractory_remaining, 1)
                        # Push zero score through smoothers (decays triggers)
                        source.smoothers[name].push(0.0)
                    score_msg.update(udp_stats_snapshot(runtime.event_bus.source_names))
                    score_msg["dsp_stats"] = _dsp_stats_snapshot(dsp, sources or {source.source_id: source})
                    if sources:
                        score_msg["source_names"] = {sid: src.name for sid, src in sources.items()}
                    runtime.publish(RuntimeMessage.from_legacy(score_msg))
                    return
        elif dsp is not None:
            source.dsp_feature_buf.clear()
            dsp.reset_source_gate(source.source_id)

        scores, embeddings, _ = yamnet(waveform)
        scores_np = scores.numpy()
        top_idx = int(scores_np.mean(axis=0).argmax())
        yamnet_top = class_names[top_idx]

        yamnet_scores: dict[str, float] = {}
        for name, indices in detector_indices.items():
            yamnet_scores[name] = float(scores_np[:, indices].max()) if indices else 0.0

        mlp_score = None
        mlp_top1 = None
        probs = None
        if clf is not None and "siren" in clf.labels:
            probs = clf.predict(embeddings.numpy())
            mean_probs = probs.mean(axis=0)
            siren_idx = clf.labels.index("siren")
            mlp_score = float(mean_probs[siren_idx])
            mlp_top1 = clf.labels[int(mean_probs.argmax())]

        effective: dict[str, float] = {}
        for name in DETECTORS:
            if name == "siren" and mlp_score is not None:
                if yamnet_scores["siren"] >= settings_mod.settings["yamnet_min_gate"]:
                    if mlp_top1 == name or mlp_score >= 0.5:
                        effective[name] = mlp_score
                    else:
                        effective[name] = 0.0
                else:
                    effective[name] = 0.0
            else:
                effective[name] = yamnet_scores[name]

        state_changes: dict[str, str | None] = {}
        for name, smoother in source.smoothers.items():
            state_changes[name] = smoother.push(effective[name])

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        mean_scores = scores_np.mean(axis=0)
        top3_idx = mean_scores.argsort()[::-1][:3]
        cur_yamnet_top3 = [
            {"name": class_names[i], "score": round(float(mean_scores[i]), 3)}
            for i in top3_idx
        ]

        cur_mlp_top3 = []
        if clf is not None:
            mp = probs.mean(axis=0) if mlp_score is not None else clf.predict(embeddings.numpy()).mean(axis=0)
            mlp_top3_idx = mp.argsort()[::-1][:3]
            cur_mlp_top3 = [
                {"name": clf.labels[i], "score": round(float(mp[i]), 3)}
                for i in mlp_top3_idx
            ]

        for name in DETECTORS:
            if source.smoothers[name].state == "idle":
                source.peaks[name] = dict(_EMPTY_PEAK)
            elif effective[name] > source.peaks[name]["score"]:
                source.peaks[name] = {
                    "score": effective[name],
                    "yamnet": yamnet_scores[name],
                    "mlp": mlp_score if name == "siren" else None,
                    "top": yamnet_top,
                    "yamnet_top3": cur_yamnet_top3,
                    "mlp_top3": cur_mlp_top3,
                }

        for name in DETECTORS:
            st = source.smoothers[name].state
            if st == "possible" and source.possible_start[name] is None:
                source.possible_start[name] = time.monotonic()
                source.window_scores[name] = []
            if st in ("possible", "confirmed"):
                source.window_scores[name].append({
                    "t": round(time.monotonic() - (source.possible_start[name] or time.monotonic()), 2),
                    "score": round(effective[name], 3),
                })
            if st == "idle":
                source.possible_start[name] = None
                source.window_scores[name] = []

        # Build and broadcast score message
        score_msg: dict = {
            "type": "score", "top": yamnet_top,
            "source": source.name, "source_id": source.source_id,
        }
        for name in DETECTORS:
            score_msg[f"{name}_score"] = round(effective[name], 3)
            score_msg[f"{name}_state"] = source.smoothers[name].state
            score_msg[f"{name}_triggers"] = source.smoothers[name].trigger_count
            score_msg[f"{name}_possible_needed"] = source.smoothers[name].possible_needed
            score_msg[f"{name}_confirmed_needed"] = source.smoothers[name].confirmed_needed
            score_msg[f"{name}_cooldown"] = round(source.smoothers[name].cooldown_remaining, 1)
            score_msg[f"{name}_refractory"] = round(source.smoothers[name].refractory_remaining, 1)
        score_msg["yamnet"] = round(yamnet_scores.get("siren", 0.0), 3)
        score_msg["mlp"] = round(mlp_score, 3) if mlp_score is not None else None
        score_msg["yamnet_top3"] = cur_yamnet_top3
        if cur_mlp_top3:
            score_msg["mlp_top3"] = cur_mlp_top3

        if dsp_result and dsp_result["ready"]:
            score_msg["dsp"] = dsp_result["probs"]
            score_msg["dsp_label"] = dsp_result["label"]
            score_msg["dsp_confidence"] = round(dsp_result["confidence"], 3)
            score_msg["dsp_skipped_yamnet"] = False
        if dsp is not None:
            score_msg["dsp_stats"] = _dsp_stats_snapshot(dsp, sources or {source.source_id: source})

        score_msg.update(udp_stats_snapshot(runtime.event_bus.source_names))
        if sources:
            score_msg["source_names"] = {sid: src.name for sid, src in sources.items()}
        runtime.publish(RuntimeMessage.from_legacy(score_msg))

        # Handle state transitions → events + clips
        for name, (_, _, prefix) in DETECTORS.items():
            new_state = state_changes[name]
            if new_state not in ("idle", "possible", "confirmed"):
                continue
            pk = source.peaks[name]
            engine_name = det_engine if (name == "siren" and clf is not None) else "yamnet"
            thr_key = f"{name}_threshold"
            clip_url = None
            if new_state == "confirmed":
                clip_name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                clip_duration = None
                if settings_mod.settings.get("clip_from_possible") and source.possible_start[name] is not None:
                    elapsed = time.monotonic() - source.possible_start[name] + 2.0
                    clip_seconds = settings_mod.settings.get("clip_seconds", CLIP_SECONDS)
                    clip_duration = min(elapsed, clip_seconds)
                source.clip_buf.save(CLIP_DIR / clip_name, last_seconds=clip_duration)
                clip_url = f"/clips/{clip_name}"
                print(f"[{ts}] {name} clip saved: {clip_url} (source: {source.name})", flush=True)
                meta = {
                    "event": name, "confidence": round(pk["score"], 3),
                    "timestamp": ts, "source": source.name, "engine": engine_name,
                    "source_id": source.source_id,
                    "yamnet": round(pk["yamnet"], 3),
                    "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                    "yamnet_top": pk["top"],
                    "yamnet_top3": pk.get("yamnet_top3", []),
                    "mlp_top3": pk.get("mlp_top3", []),
                    "threshold": settings_mod.settings.get(thr_key, 0.15),
                    "triggers": source.smoothers[name].trigger_count,
                    "triggers_needed": settings_mod.settings["confirmed_triggers"],
                }
                if source.window_scores[name]:
                    meta["window_scores"] = source.window_scores[name]
                save_clip_meta(clip_name, meta)
                source.possible_start[name] = None
                source.window_scores[name] = []

            event = {
                "type": "event", "event": name, "state": new_state,
                "confidence": round(pk["score"], 3), "timestamp": ts,
                "source": source.name, "engine": engine_name, "clip_url": clip_url,
                "source_id": source.source_id,
                "yamnet": round(pk["yamnet"], 3),
                "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                "yamnet_top": pk["top"],
                "yamnet_top3": pk.get("yamnet_top3", []),
                "mlp_top3": pk.get("mlp_top3", []),
                "threshold": settings_mod.settings.get(thr_key, 0.15),
                "triggers": source.smoothers[name].trigger_count,
                "triggers_needed": settings_mod.settings["confirmed_triggers"] if new_state == "confirmed" else settings_mod.settings["possible_triggers"],
            }
            runtime.publish(RuntimeMessage.from_legacy(event))
            print(f"EVENT {json.dumps(event)}", flush=True)
            if new_state == "confirmed":
                source.peaks[name] = dict(_EMPTY_PEAK)

    @staticmethod
    def _build_heartbeat(sources: dict[int, SourceState]) -> dict:
        """Build aggregated heartbeat when no source has data."""
        heartbeat: dict = {"type": "score", "top": ""}
        # Aggregate: show best (most urgent) state per detector across all sources
        for det_name in DETECTORS:
            best_state = "idle"
            best_triggers = 0
            for src in sources.values():
                sm = src.smoothers[det_name]
                st = sm.state
                # Priority: confirmed > possible > idle
                if st == "confirmed" or (st == "possible" and best_state != "confirmed"):
                    best_state = st
                if sm.trigger_count > best_triggers:
                    best_triggers = sm.trigger_count
            heartbeat[f"{det_name}_score"] = 0.0
            heartbeat[f"{det_name}_state"] = best_state
            heartbeat[f"{det_name}_triggers"] = best_triggers
            # Use first source for remaining fields (they're per-smoother)
            first = next(iter(sources.values()), None)
            if first:
                sm = first.smoothers[det_name]
                heartbeat[f"{det_name}_possible_needed"] = sm.possible_needed
                heartbeat[f"{det_name}_confirmed_needed"] = sm.confirmed_needed
                heartbeat[f"{det_name}_cooldown"] = round(sm.cooldown_remaining, 1)
                heartbeat[f"{det_name}_refractory"] = round(sm.refractory_remaining, 1)
        heartbeat["source_names"] = {sid: src.name for sid, src in sources.items()}
        return heartbeat
