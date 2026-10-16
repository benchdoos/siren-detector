"""Admin token authentication for FastAPI endpoints."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import Header, HTTPException, WebSocket

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() not in ("0", "false", "no")
ADMIN_TOKEN: str = ""
_TOKEN_HASH: bytes = b""

if AUTH_ENABLED:
    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
    if not ADMIN_TOKEN:
        ADMIN_TOKEN = secrets.token_urlsafe(32)
        print(f"Generated ADMIN_TOKEN: {ADMIN_TOKEN}", flush=True)
        print("Set ADMIN_TOKEN env var to use a fixed token.", flush=True)
    _TOKEN_HASH = hashlib.sha256(ADMIN_TOKEN.encode()).digest()
else:
    print("AUTH_ENABLED=false — admin auth disabled, all users have full access.", flush=True)


def check_token(token: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(token.encode()).digest(),
        _TOKEN_HASH,
    )


async def require_admin(authorization: str = Header("")):
    if not AUTH_ENABLED:
        return
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not check_token(token):
        raise HTTPException(status_code=401, detail="Admin token required")


async def require_viewer(authorization: str = Header("")):
    """Read-level auth: same token, but separate dependency for clarity."""
    if not AUTH_ENABLED:
        return
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not check_token(token):
        raise HTTPException(status_code=401, detail="Authentication required")


def ws_check_token(ws: WebSocket) -> bool:
    """Check token from WebSocket query param ?token=..."""
    if not AUTH_ENABLED:
        return True
    token = ws.query_params.get("token", "")
    return bool(token) and check_token(token)
