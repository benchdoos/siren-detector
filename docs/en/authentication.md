# Authentication and roles

The web UI supports three modes. Configure them in `hub/.env`, then recreate the detector container.

| Mode | Settings | Access |
|---|---|---|
| No authentication | `AUTH_ENABLED=false` | Everyone has full access. Use only on a trusted network. |
| Token | `AUTH_ENABLED=true`, `ADMIN_TOKEN=...` | One admin bearer token. If omitted, a token is generated at startup and printed in the logs. |
| OIDC | `AUTH_ENABLED=true` and `OIDC_*` settings | Sign in through Keycloak or another compatible OpenID Connect provider; roles determine access. |

`ADMIN_TOKEN` remains valid when OIDC is enabled. Keep it for CLI commands and emergency access. Do not share it or expose it in URLs.

## Roles

| Role | Permissions |
|---|---|
| `viewer` | View scores, events, clips, audio, and available metrics. |
| `editor` | Everything in `viewer`, plus label events and annotate clip segments. |
| `admin` | Everything in `editor`, plus read and change settings, manage calibration, and configure engines. |

The server checks permissions on APIs, WebSockets, clip/audio downloads, and spectrograms. Hiding controls in the UI is for convenience and does not replace server checks. `/` serves the sign-in shell; data and live streams require authentication. The settings page and its data APIs are restricted to administrators.

## Configure Keycloak

Create an OIDC client in Keycloak for the web application. Use a confidential client with the Authorization Code flow. Add this exact Redirect URI:

```text
https://<public-host>/auth/callback
```

Set these application variables:

```dotenv
AUTH_ENABLED=true
ADMIN_TOKEN=<long-random-token-for-cli-and-fallback-login>
OIDC_ISSUER_URL=https://<keycloak-host>/realms/<realm-name>
OIDC_CLIENT_ID=siren-detector
OIDC_CLIENT_SECRET=<client-secret>
OIDC_SESSION_SECRET=<long-random-application-secret>
OIDC_ROLE_CLAIM=realm_access.roles
OIDC_ROLE_MAP=siren-viewer=viewer,siren-editor=editor,siren-admin=admin
OIDC_COOKIE_SECURE=true
```

`OIDC_ROLE_CLAIM` is the path to a string or list of roles in the ID token/userinfo. Add the claim to the ID token with a Keycloak mapper or client scope. For example, Keycloak realm roles use `realm_access.roles`. `OIDC_ROLE_MAP` maps provider role names to application roles; the example uses custom `siren-viewer`, `siren-editor`, and `siren-admin` roles. Users without a matching role are denied.

Use unique random secrets. `OIDC_SESSION_SECRET` is required when OIDC is enabled. The session cookie is `HttpOnly`, `SameSite=Lax`, and `Secure` when `OIDC_COOKIE_SECURE=true`. Use HTTPS and keep the detector port private behind a TLS reverse proxy. For local HTTP development only, set `OIDC_COOKIE_SECURE=false`.

OIDC sessions are stored in process memory and support one worker. Sessions expire after 12 hours; users must sign in again after a detector restart. The admin token remains available to CLI tools.

To return to token-only access, remove the `OIDC_*` settings, keep `AUTH_ENABLED=true`, and configure `ADMIN_TOKEN`. To disable access checks entirely, set `AUTH_ENABLED=false`.

## Retraining and CLI

Browser authorization does not change the local `make retrain` workflow. It continues to use `ADMIN_TOKEN` to poll `/api/calibrate` and stop calibration before syncing the dataset. Configure the same token in `hub/.env` and the local project environment.

See the [API reference](api.md) for protected routes and [Operations](operations.md) for deployment details.
