"""WebSocket routes: /ws (events + scores) and /ws/audio (live PCM)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket

from ..auth import ws_check_token
from ..config import CLIP_DIR

if TYPE_CHECKING:
    from ..bus import EventBus

router = APIRouter()

_bus: EventBus | None = None


def init(bus: EventBus) -> None:
    global _bus
    _bus = bus


@router.websocket("/ws/audio")
async def ws_audio_endpoint(ws: WebSocket):
    """Stream raw float32 PCM audio to the browser for live listening."""
    if not ws_check_token(ws):
        await ws.close(code=4401, reason="Authentication required")
        return
    await ws.accept()
    # Optional source_id filter: /ws/audio?source_id=1
    sid_raw = ws.query_params.get("source_id")
    sid_filter: int | None = int(sid_raw) if sid_raw is not None else None
    queue: asyncio.Queue = asyncio.Queue()
    _bus.audio_queues[ws] = (queue, sid_filter)
    try:
        async def sender():
            while True:
                data = await queue.get()
                await ws.send_bytes(data)

        async def receiver():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception:
        pass
    finally:
        _bus.audio_queues.pop(ws, None)


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json(_bus.last_score)
    for ev in list(_bus.event_history):
        clip_url = ev.get("clip_url")
        if not clip_url:
            continue
        filename = clip_url.split("/")[-1]
        resolved = (CLIP_DIR / filename).resolve()
        if not resolved.is_relative_to(CLIP_DIR.resolve()) or not resolved.exists():
            continue
        ev_out = dict(ev)
        if filename in _bus.feedback:
            ev_out["label"] = _bus.feedback[filename]
        await ws.send_json(ev_out)

    queue: asyncio.Queue = asyncio.Queue()
    _bus.client_queues[ws] = queue

    async def sender():
        while True:
            msg = await queue.get()
            await ws.send_json(msg)

    async def receiver():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

    try:
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception:
        pass
    finally:
        _bus.client_queues.pop(ws, None)
