"""FastAPI application: mounts routes, static files, and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import secrets
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .. import auth
from . import routes_api, routes_clips, routes_ws

if TYPE_CHECKING:
    from ..bus import EventBus
    from ..runtime.hub import RuntimeHub

_bus: EventBus | None = None


def create_app(bus: EventBus, runtime: RuntimeHub | None = None) -> FastAPI:
    global _bus
    _bus = bus

    # Initialise sub-modules with bus reference
    routes_clips.init(bus)
    routes_api.init(bus, runtime)
    routes_ws.init(bus)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await routes_clips.restore_clips(bus)
        yield

    application = FastAPI(lifespan=lifespan)

    oidc_config_present = any((auth.OIDC_ISSUER_URL, auth.OIDC_CLIENT_ID, auth.OIDC_CLIENT_SECRET))
    if auth.AUTH_ENABLED and oidc_config_present and not auth.OIDC_ENABLED:
        raise RuntimeError("OIDC_ISSUER_URL, OIDC_CLIENT_ID, and OIDC_CLIENT_SECRET must all be set")
    if auth.OIDC_ENABLED:
        if not auth.OIDC_SESSION_SECRET:
            raise RuntimeError("OIDC_SESSION_SECRET is required when OIDC is enabled")
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            name="identity",
            client_id=auth.OIDC_CLIENT_ID,
            client_secret=auth.OIDC_CLIENT_SECRET,
            server_metadata_url=f"{auth.OIDC_ISSUER_URL}/.well-known/openid-configuration",
            client_kwargs={"scope": "openid profile email"},
        )
        application.add_middleware(
            SessionMiddleware,
            secret_key=auth.OIDC_SESSION_SECRET,
            session_cookie="oidc_state",
            max_age=600,
            same_site="lax",
            https_only=auth.OIDC_COOKIE_SECURE,
        )

        @application.get("/auth/login")
        async def oidc_login(request: Request):
            return await oauth.identity.authorize_redirect(
                request, request.url_for("oidc_callback"),
            )

        @application.get("/auth/callback", name="oidc_callback")
        async def oidc_callback(request: Request):
            try:
                token = await oauth.identity.authorize_access_token(request)
            except Exception as exc:
                raise HTTPException(status_code=401, detail="OIDC login failed") from exc
            claims = token.get("userinfo") or {}
            subject = claims.get("sub")
            if not subject:
                raise HTTPException(status_code=401, detail="OIDC identity has no subject")
            role = auth.role_from_oidc_claims(claims)
            if role is None:
                raise HTTPException(status_code=403, detail="No permitted OIDC role")
            sid, _principal = auth.create_session(str(subject), role)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                "siren_session", sid, max_age=auth.SESSION_TTL_SECONDS,
                httponly=True, secure=auth.OIDC_COOKIE_SECURE,
                samesite="lax", path="/",
            )
            return response

        @application.post("/auth/logout")
        async def oidc_logout(request: Request):
            cookie = request.headers.get("cookie", "")
            principal = auth.session_principal(cookie)
            if principal:
                supplied = request.headers.get("x-csrf-token", "")
                if not supplied or not secrets.compare_digest(supplied, principal.csrf_token):
                    raise HTTPException(status_code=403, detail="CSRF token required")
            auth.delete_session(cookie)
            response = RedirectResponse("/", status_code=303)
            response.delete_cookie("siren_session", path="/")
            return response

    _static = Path(__file__).parent.parent / "static"
    application.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @application.get("/")
    async def index():
        return FileResponse(_static / "index.html")

    application.include_router(routes_clips.router)
    application.include_router(routes_api.router)
    application.include_router(routes_ws.router)

    return application
