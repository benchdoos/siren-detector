"""REST API routes: settings, feedback, metrics, audits, auth, manual events."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from ..auth import AUTH_ENABLED, check_token, require_admin, require_viewer
from ..config import CLIP_DIR, DATASET_DIR, DEFAULT_SETTINGS, VALID_LABELS
from .. import settings as settings_mod
from .routes_clips import save_clip_meta

if TYPE_CHECKING:
    from ..bus import EventBus

router = APIRouter()

_bus: EventBus | None = None


def init(bus: EventBus) -> None:
    global _bus
    _bus = bus


@router.get("/api/settings", dependencies=[Depends(require_viewer)])
async def get_settings():
    return {**settings_mod.settings, "defaults": DEFAULT_SETTINGS}


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
        "yamnet_min_gate": (0.0, 1.0),
        "clip_seconds": (5, 60),
        "clip_from_possible": (0, 1),
    }
    updated = {}
    for k, default_val in DEFAULT_SETTINGS.items():
        if k in body:
            try:
                val = type(default_val)(body[k])
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}")
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
    dst_dir = DATASET_DIR / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / src.name))
    _bus.feedback[filename] = label
    print(f"[feedback] {filename} → {label}", flush=True)
    return {"ok": True}


@router.post("/api/manual-event", dependencies=[Depends(require_admin)])
async def manual_event():
    """Save the current clip buffer as a manual (user-triggered) event."""
    if _bus is None or _bus.clip_buf is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    ts = datetime.now(timezone.utc)
    clip_name = f"manual-{ts.strftime('%Y%m%d-%H%M%S')}.wav"
    _bus.clip_buf.save(CLIP_DIR / clip_name)
    clip_url = f"/clips/{clip_name}"
    ts_iso = ts.isoformat(timespec="seconds")
    meta = {
        "event": "unknown", "confidence": 0.0,
        "timestamp": ts_iso, "source": "manual", "engine": "user",
    }
    save_clip_meta(clip_name, meta)
    ev = {
        "type": "event", "event": "unknown", "state": "confirmed",
        "confidence": 0.0, "timestamp": ts_iso,
        "source": "manual", "engine": "user", "clip_url": clip_url,
        "yamnet": 0.0, "mlp": None, "yamnet_top": "",
        "yamnet_top3": [], "mlp_top3": [],
    }
    _bus.broadcast(ev)
    print(f"[{ts_iso}] manual event saved: {clip_url}", flush=True)
    return {"ok": True, "clip_url": clip_url}
