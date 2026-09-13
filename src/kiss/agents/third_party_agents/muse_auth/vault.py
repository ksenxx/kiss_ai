# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""authd: the daemon-side credential vault with surrogation.

Stores each connector service's OAuth authorized-user info under
``$KISS_HOME/muse_auth/vault/<service>.json`` (directory 0700, files
0600), mints opaque surrogate tokens bound to a single service, and
resolves surrogates back to a real bearer token — refreshing the
underlying Google credential when it has expired.  Only the daemon
process instantiates this class; agent processes never import it.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.agents.third_party_agents.muse_auth._common import SURROGATE_PREFIX, muse_auth_dir


class CredentialVault:
    """Daemon-owned store of real OAuth credentials plus surrogate minting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # surrogate token -> service name; surrogates are worthless
        # outside the boundary and die with the daemon.
        self._surrogates: dict[str, str] = {}

    def _vault_dir(self) -> Path:
        """Return the vault directory, creating it with 0700 permissions.

        Returns:
            Path to ``$KISS_HOME/muse_auth/vault``.
        """
        path = muse_auth_dir() / "vault"
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
        return path

    def _entry_path(self, service: str) -> Path:
        """Return the vault file for *service*.

        Args:
            service: Connector service name (e.g. ``"gmail"``).

        Returns:
            Path to ``<vault>/<service>.json``.
        """
        return self._vault_dir() / f"{service}.json"

    def has_credentials(self, service: str) -> bool:
        """Return whether the vault holds a credential for *service*.

        Args:
            service: Connector service name.

        Returns:
            True when an enrollment exists.
        """
        return self._entry_path(service).exists()

    def store(self, service: str, authorized_user_info: dict[str, Any], scopes: list[str]) -> None:
        """Persist a service's real OAuth credential into the vault.

        Args:
            service: Connector service name.
            authorized_user_info: Google authorized-user JSON dict
                (token, refresh_token, client_id, client_secret, ...).
            scopes: OAuth scopes the credential was granted.
        """
        with self._lock:
            payload = {"authorized_user_info": authorized_user_info, "scopes": scopes}
            write_private_file(self._entry_path(service), json.dumps(payload))

    def clear(self, service: str) -> None:
        """Delete a service's vault entry and invalidate its surrogates.

        Args:
            service: Connector service name.
        """
        with self._lock:
            path = self._entry_path(service)
            if path.exists():
                path.unlink()
            self._surrogates = {s: svc for s, svc in self._surrogates.items() if svc != service}

    def mint_surrogate(self, service: str) -> str:
        """Mint a fresh surrogate token bound to *service*.

        Args:
            service: Connector service name; must be enrolled.

        Returns:
            An opaque ``muse-sgt.<service>.<hex>`` token.

        Raises:
            KeyError: When the service has no vault credential.
        """
        surrogate = f"{SURROGATE_PREFIX}{service}.{secrets.token_hex(24)}"
        with self._lock:
            # Check-and-mint under the lock so a concurrent clear()
            # cannot leave a freshly minted surrogate alive for a
            # credential that was just removed.
            if not self._entry_path(service).exists():
                raise KeyError(f"no vault credential for service '{service}'")
            self._surrogates[surrogate] = service
        return surrogate

    def surrogate_service(self, surrogate: str) -> str | None:
        """Return the service a minted surrogate is bound to.

        Args:
            surrogate: Token presented by an agent.

        Returns:
            The bound service name, or ``None`` for unknown surrogates.
        """
        with self._lock:
            return self._surrogates.get(surrogate)

    def resolve_token(self, service: str) -> str:
        """Return a currently valid real bearer token for *service*.

        Loads the vault credential, refreshes it against Google's token
        endpoint when expired, and persists the refreshed credential.

        Args:
            service: Connector service name.

        Returns:
            The real OAuth2 access token.

        Raises:
            KeyError: When the service has no vault credential.
            RuntimeError: When the stored credential is unusable and
                cannot be refreshed.
        """
        with self._lock:
            path = self._entry_path(service)
            if not path.exists():
                raise KeyError(f"no vault credential for service '{service}'")
            payload = json.loads(path.read_text())
            info = payload["authorized_user_info"]
            if info.get("kind") == "bearer":
                # Plain bearer-token services (Notion, GitHub, ...):
                # the stored token is the credential itself.
                return str(info["token"])
            scopes = payload.get("scopes") or None
            creds = Credentials.from_authorized_user_info(info, scopes)
            if not creds.valid:
                if not (creds.expired and creds.refresh_token):
                    raise RuntimeError(f"vault credential for '{service}' is not refreshable")
                creds.refresh(Request())
                payload["authorized_user_info"] = json.loads(creds.to_json())
                write_private_file(path, json.dumps(payload))
            return str(creds.token)
