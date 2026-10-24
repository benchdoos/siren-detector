"""Authentication and role checks for HTTP and WebSocket routes."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from http.cookies import SimpleCookie
from dataclasses import dataclass
from enum import IntEnum

from fastapi import HTTPException, Request, WebSocket

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


class Role(IntEnum):
    VIEWER = 1
    EDITOR = 2
    ADMIN = 3


def role_from_oidc_claims(claims: dict) -> Role | None:
    value = claims
    for part in OIDC_ROLE_CLAIM.split("."):
        value = value.get(part, {}) if isinstance(value, dict) else {}
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set)):
        return None
    mapped = {
        OIDC_ROLE_MAP[str(item).strip().lower()]
        for item in values
        if str(item).strip().lower() in OIDC_ROLE_MAP
    }
    return next((role for role in (Role.ADMIN, Role.EDITOR, Role.VIEWER)
                 if role.name.lower() in mapped), None)


@dataclass(frozen=True)
class Principal:
    role: Role
    subject: str = "token-admin"
    csrf_token: str = ""
    session: bool = False


OIDC_ISSUER_URL = os.environ.get("OIDC_ISSUER_URL", "").rstrip("/")
OIDC_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")
OIDC_ROLE_CLAIM = os.environ.get("OIDC_ROLE_CLAIM", "roles")
OIDC_ROLE_MAP = {
    k.strip().lower(): v.strip().lower()
    for item in os.environ.get("OIDC_ROLE_MAP", "viewer=viewer,editor=editor,admin=admin").split(",")
    if "=" in item for k, v in [item.split("=", 1)]
}
OIDC_ENABLED = AUTH_ENABLED and bool(OIDC_ISSUER_URL and OIDC_CLIENT_ID and OIDC_CLIENT_SECRET)
OIDC_COOKIE_SECURE = os.environ.get("OIDC_COOKIE_SECURE", "true").lower() not in ("0", "false", "no")
OIDC_SESSION_SECRET = os.environ.get("OIDC_SESSION_SECRET", "")
_sessions: dict[str, tuple[Principal, float]] = {}
SESSION_TTL_SECONDS = 12 * 60 * 60


def check_token(token: str) -> bool:
    return bool(_TOKEN_HASH) and hmac.compare_digest(
        hashlib.sha256(token.encode()).digest(), _TOKEN_HASH,
    )


def principal_from_token(authorization: str) -> Principal | None:
    token = authorization.removeprefix("Bearer ").strip()
    return Principal(Role.ADMIN) if token and check_token(token) else None


async def get_principal(request: Request) -> Principal | None:
    if not AUTH_ENABLED:
        return Principal(Role.ADMIN, "anonymous")
    principal = principal_from_token(request.headers.get("authorization", ""))
    if principal is None:
        principal = principal_from_token(f"Bearer {request.query_params.get('token', '')}")
    if principal:
        return principal
    return session_principal(request.headers.get("cookie", ""))


async def _require(role: Role, request: Request) -> Principal:
    principal = await get_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if principal.role < role:
        raise HTTPException(status_code=403, detail="Insufficient role")
    if principal.session and role >= Role.EDITOR:
        supplied = request.headers.get("x-csrf-token", "")
        if not supplied or not hmac.compare_digest(supplied, principal.csrf_token):
            raise HTTPException(status_code=403, detail="CSRF token required")
    return principal


async def require_viewer(request: Request) -> Principal:
    return await _require(Role.VIEWER, request)


async def require_editor(request: Request) -> Principal:
    return await _require(Role.EDITOR, request)


async def require_admin(request: Request) -> Principal:
    return await _require(Role.ADMIN, request)


def ws_principal(ws: WebSocket) -> Principal | None:
    if not AUTH_ENABLED:
        return Principal(Role.ADMIN, "anonymous")
    principal = principal_from_token(ws.headers.get("authorization", ""))
    if principal is None:
        principal = principal_from_token(f"Bearer {ws.query_params.get('token', '')}")
    if principal:
        return principal
    return session_principal(ws.headers.get("cookie", ""))


def ws_check_token(ws: WebSocket) -> bool:
    """Compatibility helper; accepts any authenticated viewer principal."""
    if not AUTH_ENABLED:
        return True
    principal = principal_from_token(f"Bearer {ws.query_params.get('token', '')}")
    return principal is not None or ws_principal(ws) is not None


def session_principal(cookie_header: str) -> Principal | None:
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return None
    session_id = cookie.get("siren_session")
    if not session_id:
        return None
    entry = _sessions.get(session_id.value)
    if not entry:
        return None
    principal, expires = entry
    if expires <= time.time():
        _sessions.pop(session_id.value, None)
        return None
    return principal


def create_session(subject: str, role: Role) -> tuple[str, Principal]:
    now = time.time()
    for expired in [key for key, (_, expires) in _sessions.items() if expires <= now]:
        _sessions.pop(expired, None)
    session_id = secrets.token_urlsafe(32)
    principal = Principal(role, subject, secrets.token_urlsafe(32), session=True)
    _sessions[session_id] = (principal, now + SESSION_TTL_SECONDS)
    return session_id, principal


def delete_session(cookie_header: str) -> None:
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return
    session_id = cookie.get("siren_session")
    if session_id:
        _sessions.pop(session_id.value, None)
