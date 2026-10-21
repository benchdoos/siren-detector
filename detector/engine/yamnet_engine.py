"""YAMNet + MLP detection engine.

Implements ``EngineInterface``: receives normalized audio frames through
RuntimeHub, runs YAMNet embeddings + an optional MLP classifier, and routes
inference results back through RuntimeHub to the current event policy.

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
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..audio import AudioRingBuffer, ClipBuffer
from ..runtime.contracts import AudioFrame, InferenceResult, RuntimeMessage
from ..runtime.hub import RuntimeHub
from ..config import (
    ANALYSIS_SECONDS,
    CLIP_SECONDS,
    DETECTORS,
    DSP_LABELS_PATH,
    DSP_MODEL_PATH,
    DSP_NORM_STATS_PATH,
    SAMPLE_RATE,
)
from ..transport.udp import udp_stats_snapshot
from .smoother import TemporalSmoother
from .source_state import SourceState
from .event_policy import _EMPTY_PEAK, _dsp_stats_snapshot
from .stages.yamnet_mlp import YAMNetMlpStage
from .manifest import YAMNET_ENGINE_MANIFEST
from ..runtime.policies import EventPolicyContext, load_event_policy
from ..runtime.engine_settings import settings_for_engine
from .. import settings as settings_mod

logger = logging.getLogger(__name__)

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


def _inference_hop_samples(settings: dict) -> int:
    interval = float(settings.get("inference_interval_seconds", 0.5))
    return int(SAMPLE_RATE * min(2.0, max(0.25, interval)))


def _create_source(source_id: int, name: str, settings: dict) -> SourceState:
    """Create a fresh SourceState with buffers and smoothers from current settings."""
    s = settings
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

    ENGINE_MANIFEST = YAMNET_ENGINE_MANIFEST

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
        parser.add_argument(
            "--event-policy", default="default",
            help="Event policy name or local package.module.ClassName",
        )

    def start(self, runtime: RuntimeHub, settings: dict, args: argparse.Namespace) -> None:
        # Keep the current bus for source/clip state and live audio; scores and
        # events go through RuntimeHub to the output layer.
        bus = runtime.event_bus
        engine_settings = settings_for_engine(runtime, settings)
        graph_mode = runtime.engine_graph_active
        print(f"Settings: {engine_settings}", flush=True)

        # Parse source name mappings
        source_names: dict[int, str] = {}
        if args.source_name:
            for mapping in args.source_name:
                if "=" in mapping:
                    sid_str, name = mapping.split("=", 1)
                    source_names[int(sid_str)] = name
        # Default source 0 gets --source name
        source_names.setdefault(0, args.source)

        # ── Load replaceable model stage (shared across sources) ─────────
        model_stage = YAMNetMlpStage.load(
            args.classifier_weights, args.classifier_labels,
        )
        det_engine = model_stage.engine_name

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
            if not graph_mode:
                settings_mod.smoothers = all_smoothers

        def get_source(source_id: int) -> SourceState:
            """Get or register per-source processing state."""
            if source_id in sources:
                return sources[source_id]
            with sources_lock:
                if source_id in sources:
                    return sources[source_id]
                name = source_names.get(source_id, f"source-{source_id}")
                s = _create_source(source_id, name, engine_settings)
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

        policy_context = EventPolicyContext(
            runtime, get_source, sources, dsp, settings=engine_settings,
        )
        event_policy = load_event_policy(args.event_policy, policy_context)
        runtime.set_event_policy(event_policy, engine_id=YAMNET_ENGINE_MANIFEST.engine_id)
        runtime.subscribe_audio(
            on_audio_frame, subscriber_id=YAMNET_ENGINE_MANIFEST.engine_id,
            engine_id=YAMNET_ENGINE_MANIFEST.engine_id,
        )

        print("Ready.", flush=True)

        # ── Unified inference loop with hot-switchable mode ──────────────
        _last_heartbeat = 0.0
        _parallel_stop = threading.Event()
        _parallel_threads: dict[int, threading.Thread] = {}
        _current_mode = ""

        def _source_loop(source: SourceState, stop: threading.Event):
            """Inference loop for a single source (used in parallel mode)."""
            try:
                while not stop.is_set():
                    disabled = engine_settings.get("disabled_sources", [])
                    waveform = source.inference_buf.consume(_inference_hop_samples(engine_settings))
                    if waveform is None:
                        stop.wait(0.05)
                        continue
                    if source.source_id in disabled:
                        continue
                    self._process_source(
                        source, waveform, model_stage, det_engine, runtime,
                        sources, dsp, settings=engine_settings,
                    )
            except Exception as exc:
                runtime.set_component_state(
                    YAMNET_ENGINE_MANIFEST.engine_id, "degraded",
                    f"source {source.source_id}: {type(exc).__name__}: {exc}",
                )
                logger.exception(
                    "Parallel inference worker failed for source %s",
                    source.source_id,
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
            mode = engine_settings.get("detection_mode", "round_robin")

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
                        heartbeat["dsp_stats"] = _dsp_stats_snapshot(dsp, sources, engine_settings)
                    runtime.publish(RuntimeMessage.from_legacy(heartbeat))
                    _last_heartbeat = now
                time.sleep(0.5)
            else:
                # Round-robin: process all sources on main thread
                any_data = False
                disabled = engine_settings.get("disabled_sources", [])
                for source in list(sources.values()):
                    waveform = source.inference_buf.consume(_inference_hop_samples(engine_settings))
                    if waveform is None:
                        continue
                    any_data = True
                    if source.source_id in disabled:
                        continue
                    self._process_source(
                        source, waveform, model_stage, det_engine, runtime,
                        sources, dsp, settings=engine_settings,
                    )

                now = time.monotonic()
                if not any_data:
                    time.sleep(0.05)
                    if now - _last_heartbeat >= 1.0:
                        heartbeat = self._build_heartbeat(sources)
                        heartbeat.update(udp_stats_snapshot(bus.source_names))
                        if dsp is not None:
                            heartbeat["dsp_stats"] = _dsp_stats_snapshot(dsp, sources, engine_settings)
                        runtime.publish(RuntimeMessage.from_legacy(heartbeat))
                        _last_heartbeat = now
                else:
                    _last_heartbeat = now

    # ── Per-source inference ─────────────────────────────────────────────

    @staticmethod
    def _process_source(
        source: SourceState,
        waveform: np.ndarray,
        model_stage: YAMNetMlpStage,
        det_engine: str,
        runtime: RuntimeHub,
        sources: dict | None = None,
        dsp=None,
        settings: dict | None = None,
    ) -> None:
        settings = settings if settings is not None else settings_mod.settings
        waveform = np.clip(waveform, -1.0, 1.0)

        # ── Click removal: suppress impulse noise before inference ──────
        cr_override = settings.get(
            "click_removal_source_overrides", {},
        ).get(str(source.source_id), {})
        cr_enabled = cr_override.get(
            "enabled", settings.get("click_removal_enabled", 1),
        )
        if cr_enabled:
            cr_threshold = cr_override.get(
                "threshold",
                settings.get("click_removal_threshold", 10.0),
            )
            cr_max_width = cr_override.get(
                "max_width",
                settings.get("click_removal_max_width", 5),
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
        dsp_override = settings.get("dsp_source_overrides", {}).get(
            str(source.source_id), {},
        )
        source_dsp_enabled = dsp_override.get(
            "dsp_enabled", settings.get("dsp_enabled", 1),
        )
        if dsp is not None and source_dsp_enabled:
            new_audio = waveform[-_inference_hop_samples(settings):]
            dsp_result = dsp.process(new_audio, source.dsp_feature_buf)

            if dsp_result["ready"]:
                bg_threshold = dsp_override.get(
                    "bg_threshold", settings.get("dsp_bg_threshold", 0.85),
                )
                bg_release_threshold = dsp_override.get(
                    "bg_release_threshold", settings.get("dsp_bg_release_threshold", 0.85),
                )
                bg_confirm_windows = dsp_override.get(
                    "bg_confirm_windows", settings.get("dsp_bg_confirm_windows", 2),
                )
                max_consecutive_skips = dsp_override.get(
                    "max_consecutive_skips",
                    settings.get("dsp_max_consecutive_skips", 20),
                )
                dsp_event_gate = dsp_override.get(
                    "event_gate",
                    settings.get("dsp_event_gate", 0.08),
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
                    runtime.publish_inference(InferenceResult(
                        source_id=source.source_id,
                        engine_id=YAMNET_ENGINE_MANIFEST.engine_id,
                        stage_id="dsp-prefilter",
                        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        scores={f"yamnet:{name}": 0.0 for name in DETECTORS},
                        metadata={"dsp_skipped_yamnet": True, "dsp_result": dsp_result},
                    ))
                    return
        elif dsp is not None:
            source.dsp_feature_buf.clear()
            dsp.reset_source_gate(source.source_id)

        prediction = model_stage.predict(source.source_id, waveform)
        metadata = dict(prediction.metadata)
        metadata.update({
            "det_engine": det_engine,
            "dsp_result": dsp_result,
        })
        runtime.publish_inference(InferenceResult(
            source_id=prediction.source_id,
            engine_id=prediction.engine_id,
            stage_id=prediction.stage_id,
            timestamp=prediction.timestamp,
            scores=prediction.scores,
            metadata=metadata,
        ))

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
