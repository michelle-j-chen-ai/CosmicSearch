"""Machine auth token for outbound DORA gRPC calls.

Python port of the SnapTag/tag-tracker ``auth.go`` machine-credential flow:
long-lived machine ``client_id`` / ``client_secret`` are exchanged at the Applied
accounts API for a short-lived bearer token, which is cached and auto-refreshed.
Unlike a personal JWT, this never permanently expires -- the credentials are
durable and the token refreshes itself, so a deployed service keeps working.

Credentials come from ``MACHINE_CLIENT_ID`` / ``MACHINE_CLIENT_SECRET`` (set as
Apps Platform secrets in the cloud; exported locally for dev). When they are
absent, callers fall back to a static ``URSA_SDK_GRPC_AUTH_TOKEN`` (local dev).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)

_ACCOUNTS_URL = "https://accounts.applied.co/api/machineCredential/get"
# Conservative lifetime: the accounts API does not return the real TTL, and
# tag-tracker observes ~50 min, so refresh at 45 min (with a 60s safety margin).
_TOKEN_TTL_S = 45 * 60

_lock = threading.Lock()
_token = ""
_expiry = 0.0


class MachineAuthUnavailable(RuntimeError):
    """Raised when machine credentials are absent or the exchange fails."""


def available() -> bool:
    """True iff machine client credentials are configured."""
    return bool(
        os.environ.get("MACHINE_CLIENT_ID") and os.environ.get("MACHINE_CLIENT_SECRET")
    )


def _fetch(client_id: str, client_secret: str) -> str:
    payload = json.dumps(
        {"client_id": client_id, "client_secret": client_secret}
    ).encode()
    req = urllib.request.Request(
        _ACCOUNTS_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise MachineAuthUnavailable(
            f"machine token request to accounts.applied.co failed: {exc}"
        ) from exc
    token = data.get("machine_auth_token", "")
    if not token:
        raise MachineAuthUnavailable(
            f"empty machine_auth_token in response: {data.get('message', '')}"
        )
    return token


def get_machine_token() -> str:
    """A valid machine bearer token, fetching/refreshing as needed (thread-safe).

    Returns a cached token until ~1 min before expiry, then re-exchanges the
    durable credentials. If a refresh fails but a previous (stale) token exists,
    that token is returned rather than raising, matching tag-tracker's behavior.
    """
    global _token, _expiry
    with _lock:
        if _token and time.time() < _expiry - 60:
            return _token
        client_id = os.environ.get("MACHINE_CLIENT_ID", "")
        client_secret = os.environ.get("MACHINE_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise MachineAuthUnavailable(
                "MACHINE_CLIENT_ID / MACHINE_CLIENT_SECRET not set"
            )
        try:
            token = _fetch(client_id, client_secret)
        except MachineAuthUnavailable:
            if _token:
                LOGGER.warning("machine token refresh failed; using stale token")
                return _token
            raise
        _token = token
        _expiry = time.time() + _TOKEN_TTL_S
        LOGGER.info("machine auth token acquired (refresh in %ds)", _TOKEN_TTL_S)
        return _token
