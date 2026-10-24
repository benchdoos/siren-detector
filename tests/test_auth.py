"""Authorization roles, OIDC claim mapping, and session enforcement."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from starlette.requests import Request

from detector import auth


def request_with(*headers: tuple[bytes, bytes]) -> Request:
    return Request({
        "type": "http", "method": "POST", "path": "/feedback",
        "headers": list(headers), "query_string": b"",
        "server": ("testserver", 80), "scheme": "http", "client": ("test", 1),
    })


class AuthorizationTests(unittest.TestCase):
    def test_oidc_claim_mapping_supports_nested_keycloak_roles(self) -> None:
        with patch.object(auth, "OIDC_ROLE_CLAIM", "realm_access.roles"), patch.object(
            auth, "OIDC_ROLE_MAP", {"siren-viewer": "viewer", "siren-editor": "editor"},
        ):
            self.assertEqual(
                auth.role_from_oidc_claims({"realm_access": {"roles": ["siren-viewer", "siren-editor"]}}),
                auth.Role.EDITOR,
            )

    def test_unknown_oidc_role_is_not_granted_access(self) -> None:
        self.assertIsNone(auth.role_from_oidc_claims({"roles": ["unrelated-role"]}))

    @unittest.skipUnless(auth.AUTH_ENABLED, "session authorization is disabled")
    def test_editor_session_requires_csrf_and_viewers_cannot_label(self) -> None:
        sid, editor = auth.create_session("editor-1", auth.Role.EDITOR)
        cookie = f"siren_session={sid}".encode()
        good = request_with((b"cookie", cookie), (b"x-csrf-token", editor.csrf_token.encode()))
        bad = request_with((b"cookie", cookie))
        self.assertIs(asyncio.run(auth.require_editor(good)), editor)
        with self.assertRaises(auth.HTTPException) as error:
            asyncio.run(auth.require_editor(bad))
        self.assertEqual(error.exception.status_code, 403)

        viewer_sid, _ = auth.create_session("viewer-1", auth.Role.VIEWER)
        viewer_request = request_with((b"cookie", f"siren_session={viewer_sid}".encode()))
        with self.assertRaises(auth.HTTPException) as error:
            asyncio.run(auth.require_editor(viewer_request))
        self.assertEqual(error.exception.status_code, 403)

    def test_admin_token_remains_valid_for_cli_compatibility(self) -> None:
        if auth.AUTH_ENABLED:
            self.assertTrue(auth.check_token(auth.ADMIN_TOKEN))
            self.assertEqual(auth.principal_from_token(f"Bearer {auth.ADMIN_TOKEN}").role, auth.Role.ADMIN)


if __name__ == "__main__":
    unittest.main()
