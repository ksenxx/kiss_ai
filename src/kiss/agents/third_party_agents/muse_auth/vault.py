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

import requests
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

    def store(
        self,
        service: str,
        authorized_user_info: dict[str, Any],
        scopes: list[str],
        hosts: list[str] | None = None,
        insecure_hosts: list[str] | None = None,
    ) -> None:
        """Persist a service's real OAuth credential into the vault.

        Args:
            service: Connector service name.
            authorized_user_info: Google authorized-user JSON dict
                (token, refresh_token, client_id, client_secret, ...),
                or ``{"kind": "bearer"|"header", ...}`` for plain-token
                services.
            scopes: OAuth scopes the credential was granted.
            hosts: Extra hostnames the credential may be spent against
                (consent-time allowlist extension, e.g. a self-hosted
                Firecrawl instance); merged with the built-in hosts by
                Sentinel.
            insecure_hosts: Hostnames the credential may reach over
                plain HTTP (consent-time exception for services the
                user pointed at an ``http://`` base URL, e.g. a LAN
                Home Assistant instance); implicitly part of the host
                allowlist.
        """
        with self._lock:
            payload: dict[str, Any] = {
                "authorized_user_info": authorized_user_info,
                "scopes": scopes,
                # Every store starts a new generation: in-flight
                # requests pinned to the previous generation abort
                # instead of spending this credential (token refreshes
                # rewrite the payload in place, preserving the
                # generation, so they never abort anything).
                "generation": secrets.token_hex(8),
            }
            if hosts:
                payload["hosts"] = list(hosts)
            if insecure_hosts:
                payload["insecure_hosts"] = list(insecure_hosts)
            write_private_file(self._entry_path(service), json.dumps(payload))
            # Surrogates minted against the old credential/host scope
            # die with it, so an old handle can never spend the new
            # token (or ship the old token under the new host scope).
            self._surrogates = {s: svc for s, svc in self._surrogates.items() if svc != service}

    def _payload(self, service: str) -> dict[str, Any]:
        """Load a service's vault payload, tolerating absence.

        Args:
            service: Connector service name.

        Returns:
            The stored payload dict, or ``{}`` when the service is not
            enrolled or the file is unreadable.
        """
        path = self._entry_path(service)
        if not path.exists():
            return {}
        try:
            return dict(json.loads(path.read_text()))
        except (OSError, ValueError):
            return {}

    def enrolled_hosts(self, service: str) -> tuple[str, ...]:
        """Return the extra hosts enrolled with a service's credential.

        Insecure enrollment hosts are included: consenting to reach a
        host over plain HTTP implies the host is allowed at all.

        Args:
            service: Connector service name.

        Returns:
            Hostnames stored at enrollment time, or ``()`` when the
            service is not enrolled or declared none.
        """
        with self._lock:
            payload = self._payload(service)
        return tuple(str(h) for h in payload.get("hosts", [])) + tuple(
            str(h) for h in payload.get("insecure_hosts", [])
        )

    def enrolled_insecure_hosts(self, service: str) -> tuple[str, ...]:
        """Return the hosts the credential may reach over plain HTTP.

        Args:
            service: Connector service name.

        Returns:
            The consent-time insecure hostnames, or ``()``.
        """
        with self._lock:
            payload = self._payload(service)
        return tuple(str(h) for h in payload.get("insecure_hosts", []))

    def generation(self, service: str) -> str:
        """Return the current credential generation for *service*.

        Args:
            service: Connector service name.

        Returns:
            The generation nonce written by :meth:`store` (pre-upgrade
            entries report ``""``, which stays stable across reads).
        """
        with self._lock:
            return str(self._payload(service).get("generation", ""))

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

    def resolve_credential(
        self, service: str, generation: str | None = None
    ) -> tuple[str, str, str]:
        """Return the real credential and where the boundary must place it.

        ``{"kind": "header"}`` credentials (e.g. Brave Search's
        ``X-Subscription-Token``) are sent verbatim in their declared
        header; ``{"kind": "query"}`` credentials (BlueBubbles'
        ``password``, Synology's webhook ``token``) are spliced into the
        request URL's query string; every other kind resolves through
        :meth:`resolve_token` into ``Authorization: Bearer <token>``.

        Args:
            service: Connector service name.
            generation: When given, the vault generation the caller's
                request is pinned to; resolution fails if the credential
                was replaced since (a token refresh keeps the
                generation, so it never trips this check).

        Raises:
            KeyError: When the service has no vault credential.
            RuntimeError: When the stored credential is unusable or was
                replaced after the caller's request started.

        Returns:
            ``(placement, name, value)`` where *placement* is
            ``"header"`` or ``"query"``, *name* is the header or query
            parameter name, and *value* is the real credential.
        """
        with self._lock:
            path = self._entry_path(service)
            if not path.exists():
                raise KeyError(f"no vault credential for service '{service}'")
            info = json.loads(path.read_text())["authorized_user_info"]
        if info.get("kind") == "header":
            # Header/query-kind credentials never call resolve_token,
            # so the generation is checked here; other kinds are checked
            # inside resolve_token (avoiding a redundant double-check).
            with self._lock:
                self._check_generation(service, generation)
            return "header", str(info["header"]), str(info["token"])
        if info.get("kind") == "query":
            with self._lock:
                self._check_generation(service, generation)
            return "query", str(info["param"]), str(info["token"])
        return "header", "Authorization", f"Bearer {self.resolve_token(service, generation)}"

    def _check_generation(self, service: str, generation: str | None) -> None:
        """Raise if the stored generation no longer matches *generation*.

        Must be called with the lock held.

        Args:
            service: Connector service name.
            generation: The generation the caller's request is pinned
                to; ``None`` skips the check.

        Raises:
            RuntimeError: When the credential was replaced since.
        """
        if generation is None:
            return
        if str(self._payload(service).get("generation", "")) != generation:
            raise RuntimeError(
                f"the '{service}' credential was replaced after this request started"
            )

    def resolve_token(self, service: str, generation: str | None = None) -> str:
        """Return a currently valid real bearer token for *service*.

        Loads the vault credential, refreshes it against Google's token
        endpoint when expired, and persists the refreshed credential.

        Args:
            service: Connector service name.
            generation: When given, the vault generation the caller's
                request is pinned to; resolution fails if the credential
                was replaced since.

        Returns:
            The real OAuth2 access token.

        Raises:
            KeyError: When the service has no vault credential.
            RuntimeError: When the stored credential is unusable, cannot
                be refreshed, or was replaced after the caller's request
                started.
        """
        with self._lock:
            path = self._entry_path(service)
            if not path.exists():
                raise KeyError(f"no vault credential for service '{service}'")
            payload = json.loads(path.read_text())
            self._check_generation(service, generation)
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
                # The refresh carries the refresh token to Google's
                # token endpoint: like the boundary itself, it must be
                # immune to ambient proxy/netrc environment configs.
                session = requests.Session()
                session.trust_env = False
                with session:
                    creds.refresh(Request(session=session))
                payload["authorized_user_info"] = json.loads(creds.to_json())
                write_private_file(path, json.dumps(payload))
            return str(creds.token)
