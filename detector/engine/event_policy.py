"""Project event policy for inference results routed by Runtime Hub."""

from __future__ import annotations

import json
import time

from .. import settings as settings_mod
from ..config import CLIP_SECONDS, DETECTORS, SAMPLE_RATE
from ..runtime.contracts import ClipSaveRequest, InferenceResult, RuntimeMessage
from ..runtime.policies import EventPolicyContext
from ..transport.udp import udp_stats_snapshot
from .source_state import SourceState

_EMPTY_PEAK = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": "", "yamnet_top3": [], "mlp_top3": []}


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


class EventPolicy:
    """Turn inference results into smoothed scores, events, and clip records."""

    policy_id = "default"
    api_version = "event-policy.v1"

    def __init__(self, context: EventPolicyContext) -> None:
        self._runtime = context.runtime
        self._get_source = context.get_source
        self._sources = context.sources
        self._dsp = context.dsp

    def process(self, result: InferenceResult) -> None:
        """Apply project event policy to generic model results from Runtime Hub."""
        source = self._get_source(result.source_id)
        runtime = self._runtime
        sources = self._sources
        dsp = self._dsp
        if result.metadata.get("dsp_skipped_yamnet"):
            self._publish_dsp_skipped_score(source, result.metadata["dsp_result"])
            return

        yamnet_scores = {
            name: float(result.scores.get(f"yamnet:{name}", 0.0))
            for name in DETECTORS
        }
        mlp_score = result.scores.get("mlp:siren")
        mlp_top1 = result.metadata.get("mlp_top1")
        yamnet_top = result.metadata.get("yamnet_top", "")
        cur_yamnet_top3 = result.metadata.get("yamnet_top3", [])
        cur_mlp_top3 = result.metadata.get("mlp_top3", [])
        model_has_classifier = bool(result.metadata.get("has_classifier"))
        det_engine = result.metadata.get("det_engine", "yamnet")
        dsp_result = result.metadata.get("dsp_result")
        ts = result.timestamp

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
            engine_name = det_engine if (name == "siren" and model_has_classifier) else "yamnet"
            thr_key = f"{name}_threshold"
            clip_url = None
            if new_state == "confirmed":
                clip_name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                clip_duration = None
                if settings_mod.settings.get("clip_from_possible") and source.possible_start[name] is not None:
                    elapsed = time.monotonic() - source.possible_start[name] + 2.0
                    clip_seconds = settings_mod.settings.get("clip_seconds", CLIP_SECONDS)
                    clip_duration = min(elapsed, clip_seconds)
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
                clip_url = runtime.save_clip(ClipSaveRequest(
                    filename=clip_name,
                    sample_rate_hz=SAMPLE_RATE,
                    samples=source.clip_buf.snapshot(last_seconds=clip_duration),
                    metadata=meta,
                ))
                print(f"[{ts}] {name} clip saved: {clip_url} (source: {source.name})", flush=True)
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

    def _publish_dsp_skipped_score(self, source: SourceState, dsp_result: dict) -> None:
        """Publish background score and decay smoother triggers for a DSP skip."""
        score_msg: dict = {
            "type": "score", "top": "(dsp:background)",
            "source": source.name, "source_id": source.source_id,
            "dsp": dsp_result["probs"],
            "dsp_label": dsp_result["label"],
            "dsp_confidence": round(dsp_result["confidence"], 3),
            "dsp_skipped_yamnet": True,
        }
        for name in DETECTORS:
            smoother = source.smoothers[name]
            score_msg[f"{name}_score"] = 0.0
            score_msg[f"{name}_state"] = smoother.state
            score_msg[f"{name}_triggers"] = smoother.trigger_count
            score_msg[f"{name}_possible_needed"] = smoother.possible_needed
            score_msg[f"{name}_confirmed_needed"] = smoother.confirmed_needed
            score_msg[f"{name}_cooldown"] = round(smoother.cooldown_remaining, 1)
            score_msg[f"{name}_refractory"] = round(smoother.refractory_remaining, 1)
            smoother.push(0.0)
        score_msg.update(udp_stats_snapshot(self._runtime.event_bus.source_names))
        score_msg["dsp_stats"] = _dsp_stats_snapshot(
            self._dsp, self._sources or {source.source_id: source},
        )
        if self._sources:
            score_msg["source_names"] = {sid: src.name for sid, src in self._sources.items()}
        self._runtime.publish(RuntimeMessage.from_legacy(score_msg))

