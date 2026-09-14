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

import contextlib
import json
import math
import secrets
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.agents.third_party_agents.muse_auth._common import SURROGATE_PREFIX, muse_auth_dir

# The only token-endpoint error strings ever relayed into an error
# message: the RFC 6749 / RFC 8628 authorization-server error codes.  A
# shape-based rule is not enough — a reflected secret can itself look
# like a plausible identifier.
_OAUTH_ERROR_CODES = frozenset(
    {
        "access_denied",
        "authorization_pending",
        "expired_token",
        "invalid_client",
        "invalid_grant",
        "invalid_request",
        "invalid_scope",
        "server_error",
        "slow_down",
        "temporarily_unavailable",
        "unauthorized_client",
        "unsupported_grant_type",
        "unsupported_response_type",
    }
)


# Longest plausible access-token lifetime (30 days).  A cached expiry
# beyond this — from a buggy earlier build or a hostile token endpoint
# — is treated as malformed on both the write and the read path.
_MAX_CACHE_LIFETIME = 30 * 86400.0


class _TokenEndpointRedirectError(Exception):
    """Internal: the OAuth token endpoint answered with a 3xx status."""

    def __init__(self, status: int) -> None:
        super().__init__(str(status))
        self.status = status


class CredentialVault:
    """Daemon-owned store of real OAuth credentials plus surrogate minting."""

    def __init__(self) -> None:
        # Reentrant so callers can compose multi-step critical sections
        # via :meth:`locked` around the single-step vault methods.
        self._lock = threading.RLock()
        # surrogate token -> service name; surrogates are worthless
        # outside the boundary and die with the daemon.
        self._surrogates: dict[str, str] = {}

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the vault lock across a multi-step critical section.

        The lock is reentrant, so the vault methods called inside the
        section acquire it again without deadlocking.  Two compositions
        need this:

        * the daemon's atomic store-if-absent — the presence check,
          candidate validation, and conditional write form ONE critical
          section, so a concurrent authoritative store either lands
          before the section (making the candidate irrelevant,
          ``created=False``) or after it, never in between;
        * the boundary's transport-write gate — the credential
          generation is re-checked and the credential-bearing request
          head is written to the (already-connected) socket under the
          lock, so a rotation can never complete between the check and
          the moment the credential bytes are emitted.

        Never hold this lock across a wait for a network PEER (a
        response read): the peer could itself trigger a rotation whose
        store would then deadlock against this lock.

        Yields:
            None while the lock is held.
        """
        with self._lock:
            yield

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
        only_if_absent: bool = False,
    ) -> bool:
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
            only_if_absent: When True, store ATOMICALLY only if the
                service has no credential yet (auto-migration's
                store-if-absent); an existing credential is left
                untouched.  The absence check and the write happen under
                one lock, so a concurrent writer cannot slip a credential
                in between them.

        Returns:
            True when this call wrote the credential; False only when
            *only_if_absent* was set and a credential already existed.
        """
        with self._lock:
            if only_if_absent and self._entry_path(service).exists():
                return False
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
        return True

    def sweep_stale_pending(self, max_age_seconds: float = 300.0) -> None:
        """Delete abandoned candidate-validation scratch vault files.

        Candidate validation enrolls a ``<root>-pending-<hex>`` file and
        removes it in a ``finally``; a validation caller killed before
        that ``finally`` runs would leave the file behind.  These
        scratch entries are short-lived by construction, so any pending
        file older than *max_age_seconds* is swept.

        Args:
            max_age_seconds: Age past which a pending file is removed.
        """
        import time

        from kiss.agents.third_party_agents.muse_auth._common import scratch_root

        cutoff = time.time() - max_age_seconds
        with self._lock:
            for path in self._vault_dir().glob("*-pending-*.json"):
                # Only the exact token-exchange scratch grammar, so a
                # legitimate per-workspace service whose name contains
                # "-pending-" (e.g. slack-pending-<hash>) is never swept.
                if not scratch_root(path.stem):
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue

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

    def placement(self, service: str) -> str:
        """Return where a service's credential travels, without the value.

        Lets the boundary know — before Sentinel sees the URL — whether
        the request URL itself embeds the credential (so path-placed
        surrogates can be redacted from decisions and the audit log).

        Args:
            service: Connector service name.

        Returns:
            ``"header"``, ``"query"``, or ``"path"`` (unenrolled and
            token-resolving kinds report ``"header"``, where the
            eventual ``Authorization: Bearer`` credential travels).
        """
        with self._lock:
            info = self._payload(service).get("authorized_user_info", {})
        kind = info.get("kind") if isinstance(info, dict) else None
        return kind if kind in ("query", "path") else "header"

    def resolve_credential(
        self, service: str, generation: str | None = None
    ) -> tuple[str, str, str]:
        """Return the real credential and where the boundary must place it.

        ``{"kind": "header"}`` credentials (e.g. Brave Search's
        ``X-Subscription-Token``) are sent verbatim in their declared
        header; ``{"kind": "query"}`` credentials (BlueBubbles'
        ``password``, Synology's webhook ``token``) are spliced into the
        request URL's query string; ``{"kind": "path"}`` credentials
        (Telegram's ``/bot<token>/...``) replace the surrogate inside
        the URL path; every other kind resolves through
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
        if info.get("kind") == "path":
            with self._lock:
                self._check_generation(service, generation)
            # Path placement has no header/parameter name: the value
            # replaces the surrogate inside the URL path itself.
            return "path", "", str(info["token"])
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

    def _client_credentials_token(
        self, service: str, path: Path, payload: dict[str, Any]
    ) -> str:
        """Return a valid access token for an OAuth2 client-credentials entry.

        Performs the token exchange HERE in the daemon — the agent
        process never sees the client_secret or the acquired token —
        and caches the result inside the vault payload until shortly
        before it expires.  Must be called with the vault lock held;
        rewriting the payload in place preserves the credential
        generation, exactly like a Google token refresh.

        Args:
            service: Connector service name.
            path: The service's vault file.
            payload: The loaded vault payload (mutated with the cache).

        Returns:
            The acquired access token.

        Raises:
            RuntimeError: When the exchange fails.  Messages never
                contain the client_secret: transport errors are reduced
                to the exception class plus the (credential-free) token
                URL, and endpoint refusals carry only a redacted,
                truncated error code.
        """
        info = payload["authorized_user_info"]
        cache = payload.get("cached_token") or {}
        try:
            expires_at = float(cache.get("expires_at", 0.0))
        except (TypeError, ValueError):
            expires_at = 0.0
        # Reject a non-finite/NaN OR implausibly-far-future persisted
        # expiry (an ``inf`` or a ``1e308`` written by an earlier build)
        # before trusting the cache: it must never read as valid
        # forever.  The ceiling mirrors the write-path clamp (30 days).
        # 60s skew: never hand out a token about to expire mid-request.
        now = time.time()
        if (
            cache.get("access_token")
            and math.isfinite(expires_at)
            and now < expires_at - 60.0
            and expires_at <= now + _MAX_CACHE_LIFETIME + 60.0
        ):
            return str(cache["access_token"])
        token_url = str(info["token_url"])
        client_secret = str(info["client_secret"])
        form = {
            "grant_type": "client_credentials",
            "client_id": str(info["client_id"]),
            "client_secret": client_secret,
        }
        if info.get("token_scope"):
            form["scope"] = str(info["token_scope"])
        # Like the boundary and the Google refresh, the exchange must be
        # immune to ambient proxy/netrc environment configuration.  It
        # must also never follow a redirect: only the stored token_url
        # was pinned at enrollment, and a 307/308 would forward the
        # secret-bearing POST body to an unvalidated origin.
        session = requests.Session()
        session.trust_env = False
        try:
            with session:
                resp = session.post(token_url, data=form, timeout=30.0, allow_redirects=False)
            if 300 <= resp.status_code < 400:
                raise _TokenEndpointRedirectError(resp.status_code)
            data = resp.json() if resp.content else {}
        except _TokenEndpointRedirectError as redirect:
            raise RuntimeError(
                f"the token endpoint at {token_url} answered with a redirect "
                f"(HTTP {redirect.status}); refusing to follow it with the "
                f"'{service}' client credentials"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"{type(e).__name__} exchanging '{service}' client credentials at {token_url}"
            ) from None
        token = data.get("access_token") if isinstance(data, dict) else None
        if not resp.ok or not token:
            # Never relay arbitrary endpoint text (it can reflect the
            # request form verbatim or in reversible encodings): expose
            # the error only when it is a known OAuth error code.
            hint = data.get("error") if isinstance(data, dict) else None
            if isinstance(hint, str) and hint in _OAUTH_ERROR_CODES:
                detail = hint
            else:
                detail = f"HTTP {resp.status_code}"
            raise RuntimeError(
                f"the token endpoint at {token_url} refused the '{service}' "
                f"client-credentials exchange: {detail}"
            )
        try:
            lifetime = float(data.get("expires_in", 3600.0))
        except (TypeError, ValueError):
            lifetime = 3600.0
        # A non-finite/non-positive/absurd lifetime must not create a
        # cache entry that never expires (or non-RFC-8259 JSON).
        if not math.isfinite(lifetime) or not 0.0 < lifetime <= _MAX_CACHE_LIFETIME:
            lifetime = 3600.0
        payload["cached_token"] = {
            "access_token": str(token),
            "expires_at": time.time() + lifetime,
        }
        write_private_file(path, json.dumps(payload))
        return str(token)

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
            if info.get("kind") == "oauth2_client_credentials":
                return self._client_credentials_token(service, path, payload)
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
