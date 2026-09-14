# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sentinel: the sole permission authority for connector egress.

Evaluates every outbound connector request against a per-service host
allowlist and an allow/deny/ask policy, consults user grants
(once / session / time-bounded / perpetual — Muse's grant scopes), and
appends a JSON-line audit record for every decision to
``$KISS_HOME/muse_auth/audit.jsonl``.

Policy lives in ``$KISS_HOME/muse_auth/policy.json``::

    {
      "defaults": {"read": "allow", "write": "ask"},
      "services": {
        "google_drive": {"write": "allow", "extra_hosts": ["127.0.0.1"]}
      }
    }

Grants live in ``$KISS_HOME/muse_auth/grants.json`` (except
session-scoped grants, which die with the daemon).  Files are re-read
on every decision so ``grant``/``revoke`` CLI runs take effect
immediately.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kiss.agents.third_party_agents._channel_agent_utils import write_private_file
from kiss.agents.third_party_agents.muse_auth._common import (
    _safe_port,
    builtin_hosts,
    canonical_host,
    canonical_host_entry,
    host_port_entry,
    is_loopback_host,
    muse_auth_dir,
    policy_service,
    request_action,
    url_origin,
)

_DEFAULT_POLICY = {"read": "allow", "write": "ask"}


@dataclass
class Decision:
    """Outcome of a Sentinel evaluation.

    Attributes:
        verdict: ``"allow"``, ``"deny"``, or ``"ask"``.
        reason: Human-readable explanation (safe to show the agent).
    """

    verdict: str
    reason: str


def grant_command(service: str, action: str) -> str:
    """Return the CLI command a user runs to approve an asked action.

    Args:
        service: Connector service name.
        action: Action class (``"read"`` or ``"write"``).

    Returns:
        The exact shell command string.
    """
    return (
        "uv run python -m kiss.agents.third_party_agents.muse_auth "
        f"grant {service} {action} --scope once|session|perpetual [--ttl SECONDS]"
    )


class Sentinel:
    """Policy engine, grant store, and audit logger for the daemon."""

    def __init__(
        self,
        hosts_provider: Callable[[str], tuple[str, ...]],
        insecure_hosts_provider: Callable[[str], tuple[str, ...]],
    ) -> None:
        self._lock = threading.Lock()
        # Session-scoped grants live only in daemon memory.
        self._session_grants: list[dict[str, Any]] = []
        # Returns the hosts enrolled with a service's vault credential
        # (the daemon passes CredentialVault.enrolled_hosts).
        self._hosts_provider = hosts_provider
        # Returns the hosts the user consented to reach over plain HTTP
        # at enrollment time (CredentialVault.enrolled_insecure_hosts).
        self._insecure_hosts_provider = insecure_hosts_provider

    def _policy_path(self) -> Path:
        """Return the policy file path.

        Returns:
            Path to ``$KISS_HOME/muse_auth/policy.json``.
        """
        return muse_auth_dir() / "policy.json"

    def _grants_path(self) -> Path:
        """Return the persistent grants file path.

        Returns:
            Path to ``$KISS_HOME/muse_auth/grants.json``.
        """
        return muse_auth_dir() / "grants.json"

    def _load_policy(self) -> dict[str, Any]:
        """Load the policy file, tolerating absence and malformed JSON.

        Returns:
            The policy dict (``{}`` when missing or unreadable).
        """
        try:
            return dict(json.loads(self._policy_path().read_text()))
        except Exception:
            return {}

    def _load_grants(self) -> list[dict[str, Any]]:
        """Load persistent grants, tolerating absence and malformed JSON.

        Returns:
            List of grant dicts.
        """
        try:
            data = json.loads(self._grants_path().read_text())
            return list(data.get("grants", []))
        except Exception:
            return []

    def _save_grants(self, grants: list[dict[str, Any]]) -> None:
        """Persist *grants* atomically with private permissions.

        Args:
            grants: Grant dicts to store.
        """
        write_private_file(self._grants_path(), json.dumps({"grants": grants}, indent=2))

    def allowed_hosts(self, service: str) -> tuple[str, ...]:
        """Return the effective host allowlist for *service*.

        Built-in hosts from :func:`builtin_hosts`, plus any
        ``extra_hosts`` configured for the service in the policy file,
        plus hosts enrolled alongside the vault credential (e.g. a
        self-hosted Firecrawl instance).

        Args:
            service: Connector service name.

        Returns:
            Tuple of allowed hostnames (lowercase).
        """
        services = self._load_policy().get("services", {})
        # A candidate-validation scratch name (``telegram-pending-<hex>``)
        # inherits the LIVE root service's host extensions; per-workspace
        # services keep their own isolated policy (policy_service is the
        # identity for them).
        svc_policy = services.get(service) or services.get(policy_service(service), {})
        extra = svc_policy.get("extra_hosts", [])
        enrolled = self._hosts_provider(service)
        # Canonicalize policy/enrolled hosts (lowercase, drop a trailing
        # DNS dot, normalize any :port) so an FQDN- or leading-zero-port
        # entry matches a request origin canonicalized the same way.
        return (
            builtin_hosts(service)
            + tuple(canonical_host_entry(h) for h in extra)
            + tuple(canonical_host_entry(h) for h in enrolled)
        )

    def origin_allowed(self, service: str, url: str) -> bool:
        """Return whether *url*'s origin is on the service's allowlist.

        Allowlist entries are either bare hostnames (built-in service
        hosts, policy ``extra_hosts`` — any port) or port-pinned
        ``host:port`` origins (self-hosted enrollments, where the same
        hostname on another port is a different server).

        Args:
            service: Connector service name.
            url: The effective request URL.

        Returns:
            True when the URL's host (or host:port origin) is allowed.
        """
        parsed = urlparse(url)
        host, port = url_origin(parsed.scheme, parsed.hostname, _safe_port(parsed))
        entries = self.allowed_hosts(service)
        return host in entries or host_port_entry(host, port) in entries

    def _insecure_origin(self, service: str, url: str) -> bool:
        """Return whether *url* may carry the credential over plain HTTP.

        Args:
            service: Connector service name.
            url: The effective request URL.

        Returns:
            True when the URL's host or host:port origin was enrolled as
            a consent-time insecure host.
        """
        parsed = urlparse(url)
        host, port = url_origin(parsed.scheme, parsed.hostname, _safe_port(parsed))
        entries = tuple(canonical_host_entry(h) for h in self._insecure_hosts_provider(service))
        return host in entries or host_port_entry(host, port) in entries

    def add_grant(self, service: str, action: str, scope: str, ttl: float = 0.0) -> str:
        """Record a user approval as a strict capability.

        Args:
            service: Connector service name.
            action: Action class (``"read"`` or ``"write"``).
            scope: ``"once"``, ``"session"``, ``"perpetual"``, or ``"ttl"``.
            ttl: Lifetime in seconds; required when *scope* is ``"ttl"``.

        Returns:
            The new grant's ID.

        Raises:
            ValueError: On an unknown scope or a ``ttl`` scope without
                a positive ttl.
        """
        if scope not in ("once", "session", "perpetual", "ttl"):
            raise ValueError(f"unknown grant scope '{scope}'")
        if scope == "ttl" and not (math.isfinite(ttl) and ttl > 0):
            raise ValueError("ttl scope requires a positive finite --ttl")
        entry: dict[str, Any] = {
            "id": secrets.token_hex(8),
            "service": service,
            "action": action,
            "scope": scope,
            "created_at": time.time(),
        }
        if scope == "ttl":
            entry["expires_at"] = time.time() + ttl
        with self._lock:
            if scope == "session":
                self._session_grants.append(entry)
            else:
                grants = self._load_grants()
                grants.append(entry)
                self._save_grants(grants)
        return str(entry["id"])

    def revoke_grants(self, service: str, action: str = "") -> int:
        """Remove grants for a service (optionally only one action class).

        Args:
            service: Connector service name.
            action: Action class filter; empty removes all actions.

        Returns:
            Number of grants removed.
        """

        def keeps(g: dict[str, Any]) -> bool:
            return not (g["service"] == service and (not action or g["action"] == action))

        with self._lock:
            persistent = self._load_grants()
            kept = [g for g in persistent if keeps(g)]
            removed = len(persistent) - len(kept)
            if removed:
                self._save_grants(kept)
            before = len(self._session_grants)
            self._session_grants = [g for g in self._session_grants if keeps(g)]
            removed += before - len(self._session_grants)
        return removed

    def _consume_grant(self, service: str, action: str) -> dict[str, Any] | None:
        """Find a live grant for (service, action), consuming ``once`` grants.

        Args:
            service: Connector service name.
            action: Action class.

        Returns:
            The matching grant dict, or ``None``.
        """
        now = time.time()

        def matches(g: dict[str, Any]) -> bool:
            if g["service"] != service or g["action"] != action:
                return False
            return not (g["scope"] == "ttl" and float(g.get("expires_at", 0)) < now)

        with self._lock:
            for g in self._session_grants:
                if matches(g):
                    return g
            grants = self._load_grants()
            live = [g for g in grants if not (g["scope"] == "ttl" and
                                              float(g.get("expires_at", 0)) < now)]
            for g in live:
                if matches(g):
                    if g["scope"] == "once":
                        live.remove(g)
                    if live != grants:
                        self._save_grants(live)
                    return g
            if live != grants:
                self._save_grants(live)
        return None

    def decide(self, service: str, method: str, url: str, effective_url: str = "") -> Decision:
        """Evaluate one concrete outbound request.

        Args:
            service: Connector service the surrogate is bound to.
            method: HTTP method of the request.
            url: Full request URL as the client asked for it (checked
                for userinfo and scheme).
            effective_url: The URL the request will actually be sent to
                after normalization (checked for the destination host);
                defaults to *url*.  Passing the post-normalization URL
                closes urlparse-vs-transport parser differentials.

        Returns:
            A :class:`Decision`; ``ask`` verdicts carry instructions
            for obtaining a grant.
        """
        parsed = urlparse(url)
        effective = urlparse(effective_url or url)
        # Classify on the effective path so a parser-differential raw
        # URL cannot masquerade a write API method as a read.
        action = request_action(service, method, effective.path)
        host = canonical_host(effective.hostname or "")
        deny_reason = ""
        if parsed.username or parsed.password or effective.username or effective.password:
            deny_reason = "URLs with userinfo are not allowed at the boundary"
        elif not self.origin_allowed(service, effective_url or url):
            deny_reason = f"host '{host}' is not in the '{service}' connector's allowlist"
        elif effective.scheme != "https" and not (
            is_loopback_host(host)
            or (
                effective.scheme == "http"
                and self._insecure_origin(service, effective_url or url)
            )
        ):
            # Never hand a real credential to a plaintext transport —
            # unless the user enrolled this exact host from an http://
            # base URL (consent-scoped exception, e.g. a LAN Home
            # Assistant instance that has no TLS).
            deny_reason = f"non-HTTPS scheme '{effective.scheme}' to non-loopback host '{host}'"
        if deny_reason:
            decision = Decision("deny", deny_reason)
            self._audit(service, action, method, effective_url or url, decision, grant_id="")
            return decision
        policy = self._load_policy()
        services = policy.get("services", {})
        # Exact service rule wins; otherwise a candidate-validation
        # scratch name inherits its root service's rule (so validation
        # under ``telegram-pending-<hex>`` obeys an explicit ``telegram``
        # deny), then the global default.  Per-workspace services keep
        # their own policy (policy_service is the identity for them).
        rule = (
            services.get(service, {}).get(action)
            or services.get(policy_service(service), {}).get(action)
            or policy.get("defaults", {}).get(action)
            or _DEFAULT_POLICY[action]
        )
        grant_id = ""
        if rule == "deny":
            decision = Decision("deny", f"policy denies '{action}' actions for '{service}'")
        elif rule == "allow":
            decision = Decision("allow", f"policy allows '{action}' actions for '{service}'")
        else:
            # Consume grants under the policy identity: a candidate
            # validating as ``telegram-pending-<hex>`` is approved by
            # the user's ``telegram`` grant (the random scratch name
            # keeps the credential isolated but must not need its own,
            # un-grantable, grant).
            used = self._consume_grant(policy_service(service), action)
            if used is not None:
                grant_id = str(used["id"])
                decision = Decision(
                    "allow", f"approved by {used['scope']} grant {grant_id}"
                )
            else:
                # Name the GRANTABLE policy identity in the remediation
                # command: a random scratch name is un-grantable (a
                # retry mints a new one), so the user must grant the
                # root service that policy_service resolves to.
                grantable = policy_service(service)
                decision = Decision(
                    "ask",
                    f"'{action}' on '{grantable}' requires user approval. "
                    "Ask the user to run: " + grant_command(grantable, action),
                )
        # Audit the effective (post-normalization) URL so the record
        # reflects the host/path actually contacted, not a parser-
        # differential raw form.
        self._audit(service, action, method, effective_url or url, decision, grant_id)
        return decision

    def _audit(
        self,
        service: str,
        action: str,
        method: str,
        url: str,
        decision: Decision,
        grant_id: str,
    ) -> None:
        """Append one audit record; never contains credentials.

        Args:
            service: Connector service name.
            action: Action class.
            method: HTTP method.
            url: Request URL (query string stripped).
            decision: The verdict being recorded.
            grant_id: ID of the grant used, or empty.
        """
        parsed = urlparse(url)
        record = {
            "ts": time.time(),
            "service": service,
            "action": action,
            "method": method.upper(),
            "host": (parsed.hostname or "").lower(),
            "path": parsed.path,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "grant_id": grant_id,
        }
        directory = muse_auth_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(directory / "audit.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
