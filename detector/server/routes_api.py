"""REST API routes: settings, feedback, metrics, audits, auth, manual events."""

from __future__ import annotations

import json
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from fastapi.responses import FileResponse

from ..auth import AUTH_ENABLED, check_token, require_admin, require_viewer
from ..config import (
    CLIP_DIR,
    DATASET_DIR,
    DEFAULT_SETTINGS,
    DSP_LABELS_PATH,
    DSP_MODEL_PATH,
    DSP_NORM_STATS_PATH,
    VALID_LABELS,
    SAMPLE_RATE,
)
from .. import settings as settings_mod
from .routes_clips import save_clip_meta, _load_clip_meta

if TYPE_CHECKING:
    from ..bus import EventBus

router = APIRouter()

_bus: EventBus | None = None


def init(bus: EventBus) -> None:
    global _bus
    _bus = bus


@router.get("/api/settings", dependencies=[Depends(require_viewer)])
async def get_settings():
    dsp_paths = (DSP_MODEL_PATH, DSP_NORM_STATS_PATH, DSP_LABELS_PATH)
    missing_dsp_files = [Path(path).name for path in dsp_paths if not Path(path).is_file()]
    return {
        **settings_mod.settings,
        "defaults": DEFAULT_SETTINGS,
        "dsp_available": not missing_dsp_files,
        "dsp_missing_files": missing_dsp_files,
    }


@router.get("/api/auth/status")
async def auth_status():
    return {"auth_enabled": AUTH_ENABLED}


@router.post("/api/auth/check")
async def auth_check(body: dict):
    if not AUTH_ENABLED:
        return {"ok": True}
    token = body.get("token", "")
    if check_token(token):
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/api/settings", dependencies=[Depends(require_admin)])
async def post_settings(body: dict):
    SETTING_RANGES = {
        "siren_threshold": (0.0, 1.0),
        "explosion_threshold": (0.0, 1.0),
        "motorcycle_threshold": (0.0, 1.0),
        "possible_triggers": (1, 60),
        "confirmed_triggers": (1, 60),
        "cooldown_seconds": (0, 600),
        "refractory_seconds": (0, 120),
        "smoothing_window_seconds": (5, 120),
        "inference_interval_seconds": (0.25, 2.0),
        "dsp_bg_threshold": (0.5, 1.0),
        "dsp_bg_release_threshold": (0.5, 1.0),
        "dsp_bg_confirm_windows": (1, 10),
        "yamnet_min_gate": (0.0, 1.0),
        "clip_seconds": (5, 60),
        "clip_from_possible": (0, 1),
    }
    updated = {}
    for k, default_val in DEFAULT_SETTINGS.items():
        if k in body:
            # list-type settings (e.g. disabled_sources)
            if isinstance(default_val, list):
                if not isinstance(body[k], list):
                    raise HTTPException(status_code=400, detail=f"{k} must be a list")
                updated[k] = body[k]
                continue
            # dict-type settings (e.g. source_custom_names)
            if isinstance(default_val, dict):
                if not isinstance(body[k], dict):
                    raise HTTPException(status_code=400, detail=f"{k} must be an object")
                updated[k] = body[k]
                continue
            try:
                val = type(default_val)(body[k])
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}")
            SETTING_CHOICES = {
                "detection_mode": ("round_robin", "parallel"),
            }
            if k in SETTING_CHOICES:
                if val not in SETTING_CHOICES[k]:
                    raise HTTPException(status_code=400, detail=f"{k} must be one of {SETTING_CHOICES[k]}")
            lo, hi = SETTING_RANGES.get(k, (None, None))
            if lo is not None and not (lo <= val <= hi):
                raise HTTPException(status_code=400, detail=f"{k} must be between {lo} and {hi}")
            updated[k] = val
    if not updated:
        raise HTTPException(status_code=400, detail="No valid settings provided")
    settings_mod.settings.update(updated)
    settings_mod.save_settings()
    settings_mod.apply_settings_to_smoothers()
    settings_mod.apply_clip_seconds(_bus)
    print(f"[settings] updated: {updated}", flush=True)
    return settings_mod.settings


@router.post("/api/settings/reset", dependencies=[Depends(require_admin)])
async def reset_settings():
    settings_mod.settings.clear()
    settings_mod.settings.update(DEFAULT_SETTINGS)
    settings_mod.save_settings()
    settings_mod.apply_settings_to_smoothers()
    settings_mod.apply_clip_seconds(_bus)
    print("[settings] reset to defaults", flush=True)
    return settings_mod.settings


@router.get("/api/metrics", dependencies=[Depends(require_viewer)])
async def get_metrics():
    metrics_path = Path("/models/val_metrics.json")
    if not metrics_path.exists():
        return {"error": "no metrics available"}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@router.get("/api/audits", dependencies=[Depends(require_viewer)])
async def get_audits():
    audits = []
    audits_dir = Path("/models/audits")
    if audits_dir.exists():
        files = sorted(audits_dir.glob("audit_*.json"), reverse=True)[:5]
        for f in files:
            try:
                audits.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    if not audits:
        card_path = Path("/models/model_card.json")
        if card_path.exists():
            try:
                audits.append(json.loads(card_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    return {"audits": audits}


def _clean_dataset_artifacts(stem: str) -> int:
    """Remove all dataset files derived from a clip (full copy, segments, gaps).

    Searches all label dirs for files starting with *stem* (.wav).
    Returns the number of files removed.
    """
    removed = 0
    if not DATASET_DIR.exists():
        return removed
    for label_dir in DATASET_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        for f in label_dir.glob(f"{stem}*.wav"):
            f.unlink()
            removed += 1
    return removed


@router.post("/feedback", dependencies=[Depends(require_admin)])
async def feedback(body: dict):
    filename = body.get("filename", "")
    label = body.get("label", "")
    if not filename or label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid filename or label")
    src = (CLIP_DIR / filename).resolve()
    if not src.is_relative_to(CLIP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not src.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    stem = src.stem

    # Remove any previous dataset artifacts for this clip
    removed = _clean_dataset_artifacts(stem)

    # Check if clip has segments — if so, save only the gaps (non-segment parts)
    meta = _load_clip_meta(filename)
    segments = (meta or {}).get("segments", [])

    dst_dir = DATASET_DIR / label
    dst_dir.mkdir(parents=True, exist_ok=True)

    if segments:
        # Re-save segment slices (they may have landed in a different label dir before)
        _save_segment_slices(src, segments)
        _save_gap_slices(src, segments, label, dst_dir)
    else:
        shutil.copy2(str(src), str(dst_dir / src.name))

    _bus.feedback[filename] = label
    info = f"[feedback] {filename} → {label}"
    if removed:
        info += f" (cleaned {removed} old files)"
    if segments:
        info += f" (gaps only, {len(segments)} segments excluded)"
    print(info, flush=True)
    return {"ok": True}


def _save_gap_slices(
    src: Path, segments: list[dict], label: str, dst_dir: Path,
) -> None:
    """Save non-segment portions of *src* WAV into *dst_dir* as *label*."""
    with wave.open(str(src), "rb") as wf:
        params = wf.getparams()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        all_frames = wf.readframes(n_frames)
    frame_size = params.sampwidth * params.nchannels
    duration = n_frames  # in frames

    # Build sorted, merged occupied intervals (in frames)
    occupied = sorted(
        (max(0, int(s["start"] * sr)), min(duration, int(s["end"] * sr)))
        for s in segments
    )
    merged: list[tuple[int, int]] = []
    for s, e in occupied:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Gaps = [0..first_seg_start, between segs, last_seg_end..duration]
    gaps: list[tuple[int, int]] = []
    prev = 0
    for s, e in merged:
        if s > prev:
            gaps.append((prev, s))
        prev = e
    if prev < duration:
        gaps.append((prev, duration))

    stem = src.stem
    min_frames = sr  # skip gaps shorter than 1 second
    for i, (gs, ge) in enumerate(gaps):
        if ge - gs < min_frames:
            continue
        seg_frames = all_frames[gs * frame_size : ge * frame_size]
        gap_name = f"{stem}__gap{i}__{label}.wav"
        with wave.open(str(dst_dir / gap_name), "wb") as out:
            out.setparams(params)
            out.setnframes(0)
            out.writeframes(seg_frames)


def _save_segment_slices(src: Path, segments: list[dict]) -> None:
    """Slice WAV into per-segment files in dataset/{label}/."""
    with wave.open(str(src), "rb") as wf:
        params = wf.getparams()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        all_frames = wf.readframes(n_frames)
    frame_size = params.sampwidth * params.nchannels
    stem = src.stem
    for i, seg in enumerate(segments):
        s_frame = int(seg["start"] * sr)
        e_frame = int(seg["end"] * sr)
        s_frame = max(0, min(s_frame, n_frames))
        e_frame = max(s_frame, min(e_frame, n_frames))
        seg_frames = all_frames[s_frame * frame_size : e_frame * frame_size]
        if not seg_frames:
            continue
        dst_dir = DATASET_DIR / seg["label"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        seg_name = f"{stem}__seg{i}__{seg['label']}.wav"
        with wave.open(str(dst_dir / seg_name), "wb") as out:
            out.setparams(params)
            out.setnframes(0)
            out.writeframes(seg_frames)


SEGMENT_LABELS = {"siren", "explosion", "motorcycle"}


@router.post("/feedback/segments", dependencies=[Depends(require_admin)])
async def feedback_segments(body: dict):
    filename = body.get("filename", "")
    segments = body.get("segments", [])
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    src = (CLIP_DIR / filename).resolve()
    if not src.is_relative_to(CLIP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not src.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    # Validate segments
    clean = []
    for s in segments:
        label = s.get("label", "")
        start = s.get("start")
        end = s.get("end")
        if label not in SEGMENT_LABELS:
            raise HTTPException(status_code=400, detail=f"Invalid segment label: {label}")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise HTTPException(status_code=400, detail="Segment start/end must be numbers")
        if start >= end:
            raise HTTPException(status_code=400, detail="Segment start must be < end")
        clean.append({"label": label, "start": round(start, 2), "end": round(end, 2)})

    # Save segments to clip metadata
    meta = _load_clip_meta(filename) or {}
    meta["segments"] = clean
    save_clip_meta(filename, meta)

    # Clean old artifacts and re-slice
    stem = src.stem
    removed = _clean_dataset_artifacts(stem)

    if clean:
        _save_segment_slices(src, clean)

    # If clip already has a label, re-save gaps too
    current_label = _bus.feedback.get(filename) if _bus else None
    if current_label and current_label in VALID_LABELS:
        dst_dir = DATASET_DIR / current_label
        dst_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            _save_gap_slices(src, clean, current_label, dst_dir)
        else:
            # No segments left — restore full copy
            shutil.copy2(str(src), str(dst_dir / src.name))

    # Update in-memory event_history so WS replay includes segments
    if _bus:
        for ev in _bus.event_history:
            if ev.get("clip_url", "").endswith("/" + filename):
                ev["segments"] = clean
                break

    print(f"[segments] {filename} → {len(clean)} segments"
          + (f" (cleaned {removed} old files)" if removed else ""),
          flush=True)
    return {"ok": True, "count": len(clean)}


@router.get("/api/sources", dependencies=[Depends(require_viewer)])
async def get_sources():
    """List known audio sources."""
    if _bus is None:
        return {"sources": []}
    custom_names = settings_mod.settings.get("source_custom_names", {})
    return {
        "sources": [
            {
                "source_id": sid,
                "name": custom_names.get(str(sid), name),
                "original_name": name,
                "custom_name": custom_names.get(str(sid), ""),
            }
            for sid, name in sorted(_bus.source_names.items())
        ]
    }


@router.post("/api/calibrate", dependencies=[Depends(require_admin)])
async def start_calibration(body: dict):
    """Start background calibration recording for a source.

    Body: ``{"source_id": 0, "duration": 120}``
    """
    if _bus is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    source_id = body.get("source_id", 0)
    duration = body.get("duration", 120)
    if not isinstance(duration, int) or not 10 <= duration <= 900:
        raise HTTPException(status_code=400, detail="duration must be 10-600 seconds")
    if source_id not in _bus.source_names:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    if not _bus.start_calibration(source_id, duration):
        raise HTTPException(status_code=409, detail="Calibration already in progress")
    return {"ok": True, "source_id": source_id, "duration": duration}


@router.get("/api/calibrate", dependencies=[Depends(require_viewer)])
async def calibration_status():
    """Check calibration progress."""
    if _bus is None:
        return {"active": False}
    status = _bus.calibration_status()
    if status is None:
        return {"active": False}
    return {"active": True, **status}


@router.post("/api/calibrate/stop", dependencies=[Depends(require_admin)])
async def stop_calibration():
    """Stop calibration and save recorded audio as background."""
    if _bus is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    status = _bus.calibration_status()
    if status is None:
        raise HTTPException(status_code=404, detail="No calibration in progress")
    source_name = _bus.source_names.get(status["source_id"], f"source-{status['source_id']}")
    path = _bus.finish_calibration(source_name)
    if path is None:
        raise HTTPException(status_code=400, detail="Not enough audio recorded (need at least 1s)")
    return {"ok": True, "path": path, "source": source_name}


@router.post("/api/calibrate/cancel", dependencies=[Depends(require_admin)])
async def cancel_calibration():
    """Cancel calibration and discard recorded audio."""
    if _bus is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    status = _bus.calibration_status()
    if status is None:
        raise HTTPException(status_code=404, detail="No calibration in progress")
    _bus.cancel_calibration()
    return {"ok": True}


@router.get("/api/calibrate/recordings", dependencies=[Depends(require_viewer)])
async def list_calibration_recordings():
    """List all background calibration WAV files with metadata."""
    bg_dir = DATASET_DIR / "background"
    if not bg_dir.exists():
        return {"recordings": []}
    recordings = []
    for source_dir in sorted(bg_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        for wav in sorted(source_dir.glob("*.wav")):
            stat = wav.stat()
            try:
                with wave.open(str(wav), "rb") as wf:
                    duration = wf.getnframes() / wf.getframerate()
            except Exception:
                duration = 0
            recordings.append({
                "source": source_name,
                "filename": wav.name,
                "path": f"{source_name}/{wav.name}",
                "duration": round(duration, 1),
                "size_kb": round(stat.st_size / 1024),
                "recorded_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            })
    recordings.sort(key=lambda r: r["recorded_at"], reverse=True)
    return {"recordings": recordings}


@router.delete("/api/calibrate/recordings", dependencies=[Depends(require_admin)])
async def delete_calibration_recording(body: dict):
    """Delete a calibration recording. Body: {"path": "source_name/filename.wav"}"""
    rel = body.get("path", "")
    if not rel or ".." in rel:
        raise HTTPException(status_code=400, detail="Invalid path")
    bg_dir = DATASET_DIR / "background"
    target = (bg_dir / rel).resolve()
    if not target.is_relative_to(bg_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    # Also remove cached spectrogram
    spec_cache = target.with_suffix(".spec")
    target.unlink()
    if spec_cache.exists():
        spec_cache.unlink()
    # Clean up empty source dir
    parent = target.parent
    if parent != bg_dir and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    print(f"[calibrate] deleted {rel}", flush=True)
    return {"ok": True}


@router.get("/calibrate-audio/{source_name}/{filename}")
async def serve_calibration_audio(
    source_name: str, filename: str,
    token: str = "", authorization: str = Header(""),
):
    """Serve a calibration WAV file."""
    if AUTH_ENABLED:
        hdr_token = authorization.removeprefix("Bearer ").strip()
        if not ((hdr_token and check_token(hdr_token)) or (token and check_token(token))):
            raise HTTPException(status_code=401, detail="Authentication required")
    bg_dir = DATASET_DIR / "background"
    path = (bg_dir / source_name / filename).resolve()
    if not path.is_relative_to(bg_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(str(path), media_type="audio/wav")


@router.get("/calibrate-audio/{source_name}/{filename}/spectrogram")
async def calibration_spectrogram(source_name: str, filename: str):
    """Return spectrogram for a calibration recording."""
    bg_dir = DATASET_DIR / "background"
    path = (bg_dir / source_name / filename).resolve()
    if not path.is_relative_to(bg_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    from .routes_clips import generate_spectrogram
    return generate_spectrogram(path)


@router.post("/api/manual-event", dependencies=[Depends(require_admin)])
async def manual_event(body: dict | None = Body(None)):
    """Save the current clip buffer as a manual (user-triggered) event.

    Optionally accepts ``{"source_id": N}`` to pick a specific source.
    Defaults to source 0.
    """
    if _bus is None or not _bus.clip_bufs:
        raise HTTPException(status_code=503, detail="Detector not ready")
    source_id = (body or {}).get("source_id", 0)
    clip_buf = _bus.clip_bufs.get(source_id)
    if clip_buf is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    ts = datetime.now(timezone.utc)
    clip_name = f"manual-{ts.strftime('%Y%m%d-%H%M%S')}.wav"
    clip_buf.save(CLIP_DIR / clip_name)
    clip_url = f"/clips/{clip_name}"
    ts_iso = ts.isoformat(timespec="seconds")
    meta = {
        "event": "unknown", "confidence": 0.0,
        "timestamp": ts_iso, "source": "manual", "engine": "user",
        "source_id": source_id,
    }
    save_clip_meta(clip_name, meta)
    ev = {
        "type": "event", "event": "unknown", "state": "confirmed",
        "confidence": 0.0, "timestamp": ts_iso,
        "source": "manual", "engine": "user", "clip_url": clip_url,
        "source_id": source_id,
        "yamnet": 0.0, "mlp": None, "yamnet_top": "",
        "yamnet_top3": [], "mlp_top3": [],
    }
    _bus.broadcast(ev)
    print(f"[{ts_iso}] manual event saved: {clip_url} (source: {source_id})", flush=True)
    return {"ok": True, "clip_url": clip_url}
