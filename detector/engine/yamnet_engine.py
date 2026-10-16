"""YAMNet + MLP detection engine.

Implements ``EngineInterface``: receives audio via UDP, runs YAMNet
embeddings + optional MLP classifier, pushes scores and events through
the EventBus.
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
from ..bus import EventBus
from ..config import (
    ANALYSIS_SECONDS,
    CLIP_DIR,
    CLIP_SECONDS,
    DETECTORS,
    INFERENCE_HOP_SAMPLES,
    MODEL_URL,
    SAMPLE_RATE,
)
from ..server.routes_clips import save_clip_meta
from ..transport.udp import udp_receiver, udp_stats_snapshot
from .smoother import TemporalSmoother
from .. import settings as settings_mod


class YAMNetEngine:
    """YAMNet + optional MLP classifier engine."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Register YAMNet-specific CLI arguments."""
        parser.add_argument("--classifier-weights", default="/models/classifier_weights.npz",
                            help="Path to MLP classifier weights (.npz)")
        parser.add_argument("--classifier-labels", default="/models/classifier_labels.json",
                            help="Path to MLP classifier labels (.json)")

    def start(self, bus: EventBus, settings: dict, args: argparse.Namespace) -> None:
        settings_mod.settings.update(settings)
        print(f"Settings: {settings_mod.settings}", flush=True)

        print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
        yamnet = hub.load(MODEL_URL)

        with open(yamnet.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
            class_names = [row["display_name"] for row in csv.DictReader(f)]

        detector_indices: dict[str, list[int]] = {}
        for name, (yamnet_classes, _, _) in DETECTORS.items():
            indices = [i for i, n in enumerate(class_names) if n in yamnet_classes]
            detector_indices[name] = indices
            print(f"Detector '{name}' YAMNet classes: {[class_names[i] for i in indices]}", flush=True)

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

        inference_buf = AudioRingBuffer(capacity=int(ANALYSIS_SECONDS * SAMPLE_RATE))
        bus.clip_buf = ClipBuffer(seconds=settings_mod.settings.get("clip_seconds", CLIP_SECONDS))
        clip_buf = bus.clip_buf

        smoothers: dict[str, TemporalSmoother] = {}
        for name in DETECTORS:
            thr_key = f"{name}_threshold"
            smoothers[name] = TemporalSmoother(
                window_seconds=settings_mod.settings["smoothing_window_seconds"],
                hop_seconds=0.5,
                threshold=settings_mod.settings.get(thr_key, 0.15),
                possible=settings_mod.settings["possible_triggers"],
                confirmed=settings_mod.settings["confirmed_triggers"],
                cooldown_seconds=settings_mod.settings["cooldown_seconds"],
                refractory_seconds=settings_mod.settings["refractory_seconds"],
            )
        settings_mod.smoothers = smoothers

        possible_start: dict[str, float | None] = {name: None for name in DETECTORS}
        window_scores: dict[str, list[dict]] = {name: [] for name in DETECTORS}

        stop = threading.Event()
        threading.Thread(
            target=udp_receiver,
            args=(inference_buf, clip_buf, bus, args.bind, args.port, stop),
            daemon=True,
        ).start()

        print("Ready.", flush=True)

        _last_heartbeat = 0.0
        _empty_peak = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": "", "yamnet_top3": [], "mlp_top3": []}
        peaks: dict[str, dict] = {name: dict(_empty_peak) for name in DETECTORS}

        while True:
            waveform = inference_buf.consume(INFERENCE_HOP_SAMPLES)
            if waveform is None:
                time.sleep(0.05)
                now = time.monotonic()
                if now - _last_heartbeat >= 1.0:
                    heartbeat: dict = {"type": "score", "top": ""}
                    for name in DETECTORS:
                        heartbeat[f"{name}_score"] = 0.0
                        heartbeat[f"{name}_state"] = smoothers[name].state
                        heartbeat[f"{name}_triggers"] = smoothers[name].trigger_count
                        heartbeat[f"{name}_possible_needed"] = smoothers[name].possible_needed
                        heartbeat[f"{name}_confirmed_needed"] = smoothers[name].confirmed_needed
                        heartbeat[f"{name}_cooldown"] = round(smoothers[name].cooldown_remaining, 1)
                        heartbeat[f"{name}_refractory"] = round(smoothers[name].refractory_remaining, 1)
                    heartbeat.update(udp_stats_snapshot())
                    bus.broadcast(heartbeat)
                    _last_heartbeat = now
                continue

            waveform = np.clip(waveform, -1.0, 1.0)
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
            for name, smoother in smoothers.items():
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
                mean_probs = probs.mean(axis=0) if mlp_score is not None else clf.predict(embeddings.numpy()).mean(axis=0)
                mlp_top3_idx = mean_probs.argsort()[::-1][:3]
                cur_mlp_top3 = [
                    {"name": clf.labels[i], "score": round(float(mean_probs[i]), 3)}
                    for i in mlp_top3_idx
                ]

            for name in DETECTORS:
                if smoothers[name].state == "idle":
                    peaks[name] = dict(_empty_peak)
                elif effective[name] > peaks[name]["score"]:
                    peaks[name] = {
                        "score": effective[name],
                        "yamnet": yamnet_scores[name],
                        "mlp": mlp_score if name == "siren" else None,
                        "top": yamnet_top,
                        "yamnet_top3": cur_yamnet_top3,
                        "mlp_top3": cur_mlp_top3,
                    }

            for name in DETECTORS:
                st = smoothers[name].state
                if st == "possible" and possible_start[name] is None:
                    possible_start[name] = time.monotonic()
                    window_scores[name] = []
                if st in ("possible", "confirmed"):
                    window_scores[name].append({
                        "t": round(time.monotonic() - (possible_start[name] or time.monotonic()), 2),
                        "score": round(effective[name], 3),
                    })
                if st == "idle":
                    possible_start[name] = None
                    window_scores[name] = []

            score_msg: dict = {"type": "score", "top": yamnet_top}
            for name in DETECTORS:
                score_msg[f"{name}_score"] = round(effective[name], 3)
                score_msg[f"{name}_state"] = smoothers[name].state
                score_msg[f"{name}_triggers"] = smoothers[name].trigger_count
                score_msg[f"{name}_possible_needed"] = smoothers[name].possible_needed
                score_msg[f"{name}_confirmed_needed"] = smoothers[name].confirmed_needed
                score_msg[f"{name}_cooldown"] = round(smoothers[name].cooldown_remaining, 1)
                score_msg[f"{name}_refractory"] = round(smoothers[name].refractory_remaining, 1)
            score_msg["yamnet"] = round(yamnet_scores.get("siren", 0.0), 3)
            score_msg["mlp"] = round(mlp_score, 3) if mlp_score is not None else None
            score_msg["yamnet_top3"] = cur_yamnet_top3
            if cur_mlp_top3:
                score_msg["mlp_top3"] = cur_mlp_top3

            score_msg.update(udp_stats_snapshot())
            bus.broadcast(score_msg)
            _last_heartbeat = time.monotonic()

            for name, (_, _, prefix) in DETECTORS.items():
                new_state = state_changes[name]
                if new_state not in ("idle", "possible", "confirmed"):
                    continue
                pk = peaks[name]
                engine_name = det_engine if (name == "siren" and clf is not None) else "yamnet"
                thr_key = f"{name}_threshold"
                clip_url = None
                if new_state == "confirmed":
                    clip_name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                    clip_duration = None
                    if settings_mod.settings.get("clip_from_possible") and possible_start[name] is not None:
                        elapsed = time.monotonic() - possible_start[name] + 2.0
                        clip_seconds = settings_mod.settings.get("clip_seconds", CLIP_SECONDS)
                        clip_duration = min(elapsed, clip_seconds)
                    clip_buf.save(CLIP_DIR / clip_name, last_seconds=clip_duration)
                    clip_url = f"/clips/{clip_name}"
                    print(f"[{ts}] {name} clip saved: {clip_url}", flush=True)
                    meta = {
                        "event": name, "confidence": round(pk["score"], 3),
                        "timestamp": ts, "source": args.source, "engine": engine_name,
                        "yamnet": round(pk["yamnet"], 3),
                        "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                        "yamnet_top": pk["top"],
                        "yamnet_top3": pk.get("yamnet_top3", []),
                        "mlp_top3": pk.get("mlp_top3", []),
                        "threshold": settings_mod.settings.get(thr_key, 0.15),
                        "triggers": smoothers[name].trigger_count,
                        "triggers_needed": settings_mod.settings["confirmed_triggers"],
                    }
                    if window_scores[name]:
                        meta["window_scores"] = window_scores[name]
                    save_clip_meta(clip_name, meta)
                    possible_start[name] = None
                    window_scores[name] = []

                event = {
                    "type": "event", "event": name, "state": new_state,
                    "confidence": round(pk["score"], 3), "timestamp": ts,
                    "source": args.source, "engine": engine_name, "clip_url": clip_url,
                    "yamnet": round(pk["yamnet"], 3),
                    "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                    "yamnet_top": pk["top"],
                    "yamnet_top3": pk.get("yamnet_top3", []),
                    "mlp_top3": pk.get("mlp_top3", []),
                    "threshold": settings_mod.settings.get(thr_key, 0.15),
                    "triggers": smoothers[name].trigger_count,
                    "triggers_needed": settings_mod.settings["confirmed_triggers"] if new_state == "confirmed" else settings_mod.settings["possible_triggers"],
                }
                bus.broadcast(event)
                print(f"EVENT {json.dumps(event)}", flush=True)
                if new_state == "confirmed":
                    peaks[name] = dict(_empty_peak)
