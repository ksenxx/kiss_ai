# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Microsoft Teams Agent — channel agent with MS Teams Graph API tools.

Provides authenticated access to Microsoft Teams through Microsoft Graph.
Connects like the Muse app: ``authenticate_msteams(tenant_id, client_id)``
starts the Microsoft identity platform device code flow for a public
app registration and hands back ``https://microsoft.com/devicelogin``
plus a short code; the user signs in and consents in their own browser
and ``finish_msteams_auth()`` stores the delegated token pair (the Muse
daemon refreshes it; no client secret).  The app-only client-credentials
flow remains available by passing ``client_secret``.  Stores config in
``~/.kiss/third_party_agents/msteams/config.json``.

Usage::

    agent = MSTeamsAgent()
    agent.run(prompt_template="List all teams I'm a member of")
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    config_file_lock,
    write_private_file,
)
from kiss.agents.third_party_agents._device_auth import (
    ConsentSession,
    DeviceFlowProvider,
    DeviceFlowSession,
    TokenGrant,
    connect_prompt,
    consent_required,
)

_MSTEAMS_DIR = Path.home() / ".kiss" / "third_party_agents" / "msteams"
_config = ChannelConfig(_MSTEAMS_DIR, ("tenant_id", "client_id", "client_secret"))
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_DEFAULT_LOGIN_BASE = "https://login.microsoftonline.com"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
# Delegated Graph permissions the device code sign-in asks the user to
# consent to; ``offline_access`` yields the refresh token the daemon
# uses to renew the one-hour access token.
_DEFAULT_DELEGATED_SCOPES = (
    "offline_access User.Read Team.ReadBasic.All Channel.ReadBasic.All "
    "ChannelMessage.Read.All ChannelMessage.Send Chat.ReadWrite TeamMember.Read.All"
)
_NOT_AUTHENTICATED = (
    "Not authenticated with MS Teams. Call authenticate_msteams(tenant_id=..., "
    "client_id=...) to sign in the way the Muse app connects: it returns "
    "https://microsoft.com/devicelogin plus a short code for the user to enter "
    "in their OWN browser after signing in and consenting; then call "
    "finish_msteams_auth(). The app registration (https://portal.azure.com > App "
    "registrations) must allow public client flows and carry delegated Microsoft "
    "Graph permissions such as Team.ReadBasic.All, ChannelMessage.Send and "
    "Chat.ReadWrite. Never ask for the user's Microsoft password or 2FA code. "
    "Alternatively pass client_secret for the app-only client-credentials flow."
)

# Azure tenant IDs are GUIDs or verified domains (contoso.onmicrosoft.com):
# one URL path segment.  Rejecting anything else keeps the composed token
# URL's host and path shape fixed (no `/`, `?`, `#`, or whitespace).
_TENANT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


def _login_base() -> str:
    """Return the Azure AD login base URL, resolved per call.

    ``MSTEAMS_LOGIN_BASE`` lets tests point the daemon-side token
    exchange at a loopback token endpoint (the Muse daemon only
    accepts the real pinned host or loopback).

    Returns:
        The login base URL without a trailing slash.
    """
    return os.environ.get("MSTEAMS_LOGIN_BASE", "") or _DEFAULT_LOGIN_BASE


def _client_credential_info(tenant_id: str, client_id: str, client_secret: str) -> dict[str, str]:
    """Build the vault payload for the Azure AD client-credentials flow.

    The Muse daemon performs the token exchange itself at the network
    boundary (POST ``token_url``), so this payload carries everything
    the exchange needs and the agent process never sees the acquired
    Graph token.

    Args:
        tenant_id: Azure tenant ID (validated against _TENANT_ID_RE).
        client_id: Azure app client ID.
        client_secret: Azure app client secret.

    Returns:
        An ``oauth2_client_credentials`` vault credential dict.
    """
    return {
        "kind": "oauth2_client_credentials",
        "token_url": f"{_login_base()}/{tenant_id}/oauth2/v2.0/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "token_scope": _GRAPH_SCOPE,
    }


def _device_provider(tenant_id: str) -> DeviceFlowProvider:
    """Return the tenant's device-code endpoints under :func:`_login_base`.

    Args:
        tenant_id: Azure tenant ID or alias (validated by the caller).

    Returns:
        The provider with ``/oauth2/v2.0/devicecode`` and
        ``/oauth2/v2.0/token`` for the tenant.
    """
    base = f"{_login_base()}/{tenant_id}/oauth2/v2.0"
    return DeviceFlowProvider(device_url=f"{base}/devicecode", token_url=f"{base}/token")


def _raw_config() -> dict[str, Any]:
    """Return the legacy config parsed WITHOUT type coercion.

    ``ChannelConfig.load_metadata`` stringifies every value, which would
    turn a JSON boolean ``true`` into the credential string ``"True"``;
    migration must see the real JSON types so a malformed config is
    rejected, not coerced.

    Returns:
        The parsed config dict, or ``{}`` when missing or not an object.
    """
    try:
        data = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _scrub_config_secret(expected: str | None = None) -> None:
    """Remove a vault-migrated ``client_secret`` from config.json.

    Finishes the Muse migration automatically: the non-secret
    ``tenant_id``/``client_id``/``bot_id`` metadata survives (the
    backend still needs it to rebuild the vault payload on rotation),
    and the file is deleted when nothing else was stored.

    The whole read-compare-replace cycle runs under
    :func:`config_file_lock`, which every config writer shares, so it
    is a true compare-and-swap: a newer secret a concurrent writer
    lands either arrives before the read (the comparison sees it and
    the scrub backs off) or after the replacement (it survives), never
    in between.

    Args:
        expected: When given, the exact secret that was migrated; the
            key is scrubbed only if the config still holds that value,
            so a newer secret a concurrent writer placed there since
            the migration is left untouched.
    """
    with config_file_lock(_config.path):
        try:
            cfg = json.loads(_config.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(cfg, dict) or "client_secret" not in cfg:
            return
        if expected is not None and cfg.get("client_secret") != expected:
            return
        kept = {k: str(v) for k, v in cfg.items() if k != "client_secret" and v}
        # Raw primitives: config_file_lock is not reentrant, so the
        # locked save_json_config/clear_json_config must not be used.
        if kept:
            write_private_file(_config.path, json.dumps(kept, indent=2))
        elif _config.path.exists():  # pragma: no branch - read above proved it exists
            _config.path.unlink()


def _get_access_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Get an OAuth2 access token via client credentials flow."""
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    resp = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=30,
    )
    data = resp.json()
    return str(data.get("access_token", ""))


class MSTeamsChannelBackend(ToolMethodBackend):
    """Channel backend for Microsoft Teams via Graph API."""

    def __init__(self, graph_base: str = _GRAPH_BASE) -> None:
        self._tenant_id: str = ""
        self._client_id: str = ""
        self._client_secret: str = ""
        self._bot_id: str = ""
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._connection_info: str = ""
        self._graph_base: str = graph_base
        self._http: Any = requests
        self._muse: bool = False

    def _token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._muse:
            # ``_access_token`` holds a surrogate that never expires
            # agent-side; the daemon runs (and caches) the real
            # client-credentials exchange at the network boundary.
            return self._access_token
        if time.time() >= self._token_expiry - 60:  # pragma: no branch
            self._access_token = _get_access_token(
                self._tenant_id, self._client_id, self._client_secret
            )
            self._token_expiry = time.time() + 3600
        return self._access_token

    def _wire_muse(self) -> bool:
        """Acquire an MS Teams surrogate and wire the boundary session.

        On the FIRST migration (the vault holds no ``msteams``
        credential yet) the legacy config's tenant/client IDs and
        ``client_secret`` seed an ``oauth2_client_credentials`` vault
        entry, and the ``client_secret`` is scrubbed from
        ``config.json`` afterwards (the non-secret
        ``tenant_id``/``client_id``/``bot_id`` survive).  Once the vault
        holds a credential it is authoritative: the config candidate is
        neither validated nor applied — a stale config value (malformed
        or not) must never block or clobber a working vault credential —
        and rotations go through ``authenticate_msteams``, which
        validates the candidate before replacing anything.  No network
        round trip happens here; the daemon exchanges the secret for a
        Graph token lazily.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _raw_config()
        migrated = False
        client_secret = cfg.get("client_secret")
        # VAULT FIRST: only when no credential is enrolled yet does the
        # legacy config candidate matter, so its local validation runs
        # only then — a stale malformed config must not disable an
        # authoritative vault credential.
        handle = mint_surrogate("msteams")
        if handle is None:
            local_error = ""
            tenant_id = cfg.get("tenant_id")
            client_id = cfg.get("client_id")
            if client_secret and tenant_id and client_id:
                # Validate types and shape BEFORE any credential state
                # change: a non-string (JSON true -> "True") or
                # malformed tenant must not partially migrate the vault.
                if not all(isinstance(v, str) for v in (tenant_id, client_id, client_secret)):
                    local_error = "MS Teams config has non-string credentials."
                elif not _TENANT_ID_RE.fullmatch(tenant_id):
                    local_error = f"MS Teams config has an invalid tenant_id {tenant_id!r}"
                else:
                    # Store-if-absent is ATOMIC in the daemon (presence
                    # check, candidate validation, and write form one
                    # vault critical section), so a concurrent
                    # authoritative writer can never be clobbered — or
                    # failed — by this stale config candidate.
                    migrated = store_credentials(
                        "msteams",
                        _client_credential_info(tenant_id, client_id, client_secret),
                        [],
                        only_if_absent=True,
                    )
            # Mint again even when the local candidate was rejected: a
            # concurrent writer may have enrolled the authoritative
            # credential since the first mint, and it wins.
            handle = mint_surrogate("msteams")
            if handle is None:
                self._connection_info = (
                    local_error or "No MS Teams credential in the Muse vault or config."
                )
                return False
        if migrated and isinstance(client_secret, str):
            # Compare-and-scrub: only remove the secret we migrated, so
            # a newer secret a concurrent writer placed in config
            # between the store and here is not deleted.
            _scrub_config_secret(expected=client_secret)
        self._tenant_id = str(cfg.get("tenant_id") or "")
        self._client_id = str(cfg.get("client_id") or "")
        self._bot_id = str(cfg.get("bot_id") or "")
        self._access_token = handle.token
        self._http = MuseBoundarySession("msteams")
        self._muse = True
        return True

    def _muse_probe(self) -> tuple[bool, str]:
        """Prove the daemon can exchange this credential for a Graph token.

        See :func:`_graph_probe`.

        Returns:
            ``(ok, message)``.
        """
        return _graph_probe("msteams", self._access_token, self._graph_base)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.get(
            f"{self._graph_base}{path}", headers=self._headers(), params=params, timeout=30
        )
        result: dict[str, Any] = resp.json()
        # Muse mode marks HTTP errors so tools can surface Sentinel
        # denials and Graph failures; legacy behavior is unchanged.
        if self._muse and resp.status_code >= 400:
            result["ok"] = False
        return result

    def _post(self, path: str, body: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.post(
            f"{self._graph_base}{path}", headers=self._headers(), json=body, timeout=30
        )
        result: dict[str, Any] = resp.json() if resp.content else {"ok": True}
        if self._muse and resp.status_code >= 400:
            result["ok"] = False
        return result

    def connect(self) -> bool:
        """Authenticate with Microsoft Graph API."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
        from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

        if muse_auth_enabled():
            # Vault-first surrogate wiring; the probe below runs one
            # Graph read through the daemon boundary (audited), which
            # forces the daemon-side client-credentials exchange.  A
            # malformed legacy-config secret makes the daemon refuse
            # enrollment (MuseAuthError); fail closed with a bool.
            try:
                if not self._wire_muse():
                    return False
            except MuseAuthError as e:
                self._connection_info = f"MS Teams auth failed: {e}"
                return False
            ok, message = self._muse_probe()
            self._connection_info = message if ok else f"MS Teams auth failed: {message}"
            return ok
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No MS Teams config found."
            return False
        self._tenant_id = cfg["tenant_id"]
        self._client_id = cfg["client_id"]
        self._client_secret = cfg["client_secret"]
        self._bot_id = cfg.get("bot_id", "")
        try:
            token = self._token()
            if not token:  # pragma: no branch
                self._connection_info = "MS Teams auth failed: no token"
                return False
            self._connection_info = "Authenticated with Microsoft Teams"
            return True
        except Exception as e:
            self._connection_info = f"MS Teams auth failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll MS Teams channel for new messages."""
        if not channel_id or ":" not in channel_id:  # pragma: no branch
            return [], oldest
        team_id, chan_id = channel_id.split(":", 1)
        try:
            params: dict[str, Any] = {"$top": limit, "$orderby": "lastModifiedDateTime asc"}
            if oldest and oldest != "0":
                params["$filter"] = f"lastModifiedDateTime gt {oldest}"
            url = f"/teams/{team_id}/channels/{chan_id}/messages"
            result = self._get(url, params=params)
            msgs = result.get("value", [])
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for msg in msgs:  # pragma: no branch
                last_modified = msg.get("lastModifiedDateTime", "")
                new_oldest = last_modified
                msg_id = msg.get("id", "")
                body = msg.get("body", {})
                messages.append(
                    {
                        "ts": msg_id,
                        "thread_ts": msg_id,
                        "user": msg.get("from", {}).get("user", {}).get("id", ""),
                        "text": body.get("content", ""),
                        "id": msg_id,
                        "last_modified": last_modified,
                    }
                )
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Teams channel message."""
        if ":" not in channel_id:  # pragma: no branch
            return
        team_id, chan_id = channel_id.split(":", 1)
        if thread_ts:  # pragma: no branch
            self._post(
                f"/teams/{team_id}/channels/{chan_id}/messages/{thread_ts}/replies",
                {"body": {"content": text, "contentType": "html"}},
            )
        else:
            self._post(
                f"/teams/{team_id}/channels/{chan_id}/messages",
                {"body": {"content": text, "contentType": "html"}},
            )

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message is from the bot."""
        return bool(msg.get("user", "") == self._bot_id)

    def list_teams(self, limit: int = 20) -> str:
        """List Microsoft Teams the bot/user is a member of.

        Args:
            limit: Maximum teams to return. Default: 20.

        Returns:
            JSON string with team list (id, displayName, description).
        """
        try:
            result = self._get("/me/joinedTeams", params={"$top": limit})
            teams = [
                {"id": t.get("id", ""), "name": t.get("displayName", "")}
                for t in result.get("value", [])
            ]
            return json.dumps({"ok": True, "teams": teams}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_team(self, team_id: str) -> str:
        """Get details about a Microsoft Team.

        Args:
            team_id: Team ID.

        Returns:
            JSON string with team details.
        """
        try:
            result = self._get(f"/teams/{team_id}")
            return json.dumps({"ok": True, **result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_third_party_agents(self, team_id: str) -> str:
        """List channels in a Microsoft Team.

        Args:
            team_id: Team ID.

        Returns:
            JSON string with channel list (id, displayName, membershipType).
        """
        try:
            result = self._get(f"/teams/{team_id}/channels")
            third_party_agents = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("displayName", ""),
                    "type": c.get("membershipType", ""),
                    "description": c.get("description", ""),
                }
                for c in result.get("value", [])
            ]
            payload = {"ok": True, "third_party_agents": third_party_agents}
            return json.dumps(payload, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_channel_messages(self, team_id: str, channel_id: str, top: int = 20) -> str:
        """List messages in a Teams channel.

        Args:
            team_id: Team ID.
            channel_id: Channel ID.
            top: Maximum messages to return. Default: 20.

        Returns:
            JSON string with message list.
        """
        try:
            result = self._get(
                f"/teams/{team_id}/channels/{channel_id}/messages",
                params={"$top": top},
            )
            messages = [
                {
                    "id": m.get("id", ""),
                    "from": m.get("from", {}).get("user", {}).get("displayName", ""),
                    "body": m.get("body", {}).get("content", ""),
                    "created": m.get("createdDateTime", ""),
                }
                for m in result.get("value", [])
            ]
            return json.dumps({"ok": True, "messages": messages}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_channel_message(
        self, team_id: str, channel_id: str, content: str, content_type: str = "html"
    ) -> str:
        """Post a message to a Teams channel.

        Args:
            team_id: Team ID.
            channel_id: Channel ID.
            content: Message content.
            content_type: "html" or "text". Default: "html".

        Returns:
            JSON string with ok status and message id.
        """
        try:
            result = self._post(
                f"/teams/{team_id}/channels/{channel_id}/messages",
                {"body": {"content": content, "contentType": content_type}},
            )
            if result.get("ok") is False:
                return json.dumps({"ok": False, "error": str(result.get("error", result))[:500]})
            return json.dumps({"ok": True, "id": result.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def reply_to_message(self, team_id: str, channel_id: str, message_id: str, content: str) -> str:
        """Reply to a Teams channel message.

        Args:
            team_id: Team ID.
            channel_id: Channel ID.
            message_id: Parent message ID.
            content: Reply content.

        Returns:
            JSON string with ok status and reply id.
        """
        try:
            result = self._post(
                f"/teams/{team_id}/channels/{channel_id}/messages/{message_id}/replies",
                {"body": {"content": content, "contentType": "html"}},
            )
            if result.get("ok") is False:
                return json.dumps({"ok": False, "error": str(result.get("error", result))[:500]})
            return json.dumps({"ok": True, "id": result.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_chats(self, top: int = 20) -> str:
        """List chats for the authenticated user.

        Args:
            top: Maximum chats to return. Default: 20.

        Returns:
            JSON string with chat list.
        """
        try:
            result = self._get("/me/chats", params={"$top": top})
            chats = [
                {"id": c.get("id", ""), "topic": c.get("topic", ""), "type": c.get("chatType", "")}
                for c in result.get("value", [])
            ]
            return json.dumps({"ok": True, "chats": chats}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_chat_message(self, chat_id: str, content: str, content_type: str = "text") -> str:
        """Post a message to a Teams chat.

        Args:
            chat_id: Chat ID.
            content: Message content.
            content_type: "text" or "html". Default: "text".

        Returns:
            JSON string with ok status and message id.
        """
        try:
            result = self._post(
                f"/me/chats/{chat_id}/messages",
                {"body": {"content": content, "contentType": content_type}},
            )
            if result.get("ok") is False:
                return json.dumps({"ok": False, "error": str(result.get("error", result))[:500]})
            return json.dumps({"ok": True, "id": result.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_team_members(self, team_id: str, top: int = 50) -> str:
        """List members of a Microsoft Team.

        Args:
            team_id: Team ID.
            top: Maximum members to return. Default: 50.

        Returns:
            JSON string with member list.
        """
        try:
            result = self._get(f"/teams/{team_id}/members", params={"$top": top})
            members = [
                {
                    "id": m.get("id", ""),
                    "display_name": m.get("displayName", ""),
                    "email": m.get("email", ""),
                    "roles": m.get("roles", []),
                }
                for m in result.get("value", [])
            ]
            return json.dumps({"ok": True, "members": members}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _graph_probe(service: str, surrogate: str, graph_base: str) -> tuple[bool, str]:
    """Run one Graph read through the boundary to validate a credential.

    Client-credentials validation cannot happen agent-side (the secret
    lives only in the vault), so one small Graph read flows through the
    audited boundary, forcing the daemon-side token exchange.  A real
    Graph response — even a 403 for a missing app permission — proves
    the exchange succeeded (mirroring the legacy "token acquired"
    success criterion), but a 401 means Graph rejected the acquired
    token itself; a failed exchange or a Sentinel denial reports its
    reason.

    Args:
        service: Vault service the surrogate is bound to (``"msteams"``
            or the scratch ``"msteams-pending"``).
        surrogate: The surrogate bearer to present.
        graph_base: Graph API base URL.

    Returns:
        ``(ok, message)``.
    """
    from kiss.agents.third_party_agents.muse_auth.client import (
        MuseAuthError,
        MuseBoundarySession,
    )

    try:
        resp = MuseBoundarySession(service).get(
            f"{graph_base}/teams",
            headers={"Authorization": f"Bearer {surrogate}"},
            params={"$top": 1},
            timeout=30,
        )
    except MuseAuthError as e:
        return False, str(e)
    try:
        data = resp.json() if resp.content else {}
    except ValueError:
        data = {}
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("status") == "MUSE_AUTH_DENIED":
        return False, str(error.get("message", "denied by Muse-auth policy"))
    if resp.status_code == 401:
        # Graph documents 401 as missing/invalid authentication.
        return False, "Microsoft Graph rejected the acquired token (HTTP 401)"
    # Any other answer (including a 5xx outage) arrived because the
    # daemon-side token exchange succeeded — a failed exchange surfaces
    # as a MuseAuthError above — which is the legacy success criterion.
    return True, "Authenticated with Microsoft Teams (Muse-auth)"


def _probe_candidate_credentials(backend: MSTeamsChannelBackend, info: dict[str, str]) -> str:
    """Validate candidate client credentials without touching the live entry.

    The candidate is enrolled under the scratch service
    ``msteams-pending`` (which inherits the Graph host allowlist and
    the pinned token-endpoint list), one probe read forces the
    daemon-side exchange, and the scratch entry is removed again — so
    a rejected rotation can never destroy an existing working
    ``msteams`` vault credential.

    Args:
        backend: The agent's MS Teams backend (supplies the Graph base).
        info: The candidate ``oauth2_client_credentials`` payload.

    Returns:
        ``""`` on success, else the failure detail.
    """
    import contextlib
    import secrets

    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        mint_surrogate,
        store_credentials,
    )

    # A per-attempt unique scratch service (``msteams-pending-<hex>``)
    # so two concurrent validations can never probe or clear each
    # other's candidate; it resolves to ``msteams`` for host, token
    # endpoint, and policy inheritance via service_root.
    scratch = f"msteams-pending-{secrets.token_hex(8)}"
    store_credentials(scratch, info, [])
    try:
        handle = mint_surrogate(scratch)
        if handle is None:  # pragma: no cover - defense in depth
            return "could not mint a scratch validation surrogate"
        ok, message = _graph_probe(scratch, handle.token, backend._graph_base)
        return "" if ok else message
    finally:
        with contextlib.suppress(Exception):
            clear_credentials(scratch)


def _invalid_ids_reason(tenant_id: str, client_id: str, secret: str = "") -> str:
    """Pre-validate the identifiers before any credential state change.

    The tenant becomes one path segment of the daemon's token URL, and
    every credential value must be a clean header-safe string.

    Args:
        tenant_id: Azure tenant ID or alias.
        client_id: Azure app client ID.
        secret: Optional client secret to validate as well.

    Returns:
        A JSON error string, or ``""`` when the values are acceptable.
    """
    from kiss.agents.third_party_agents.muse_auth._common import valid_credential_value

    if not _TENANT_ID_RE.fullmatch(tenant_id):
        return json.dumps(
            {
                "ok": False,
                "error": "tenant_id must be a GUID or verified domain "
                "(letters, digits, '.', '_', '-').",
            }
        )
    if not valid_credential_value(client_id) or (secret and not valid_credential_value(secret)):
        return json.dumps(
            {
                "ok": False,
                "error": "client_id/client_secret contain control characters or stray whitespace.",
            }
        )
    return ""


def _muse_authenticate(
    backend: MSTeamsChannelBackend,
    tenant_id: str,
    client_id: str,
    info: dict[str, Any],
    bot_id: str,
) -> str:
    """Enroll an MS Teams credential into the Muse vault and validate it.

    *info* is either the ``oauth2_client_credentials`` payload built from
    a client secret or the ``oauth2_refresh_token`` pair a device code
    sign-in produced.  The candidate is validated FIRST, against a
    scratch ``-pending`` enrollment, so a rejected credential mutates
    nothing — neither the config nor an existing working vault
    credential.  Only proven credentials replace the live enrollment:
    the non-secret ``tenant_id``/``client_id``/``bot_id`` metadata is
    written to ``config.json``, and the secret material goes straight
    into the vault, never to disk outside it.  If the swap itself fails
    midway, the pre-call config bytes are restored — the vault is never
    cleared, because whichever credential it holds at that point (the
    untouched old one or the just-validated new one) is worth keeping.

    Args:
        backend: The agent's MS Teams backend to (re)wire.
        tenant_id: Azure tenant ID (already validated).
        client_id: Azure app client ID (already validated).
        info: The vault ``authorized_user_info`` payload.
        bot_id: Optional bot user ID kept as config metadata.

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents.muse_auth.client import store_credentials

    try:
        failure = _probe_candidate_credentials(backend, info)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
    if failure:
        return json.dumps({"ok": False, "error": failure})
    try:
        prev_raw: str | None = _config.path.read_text()
    except OSError:
        prev_raw = None
    try:
        meta = {"tenant_id": tenant_id, "client_id": client_id}
        if bot_id:
            meta["bot_id"] = bot_id
        _config.save(meta)
        store_credentials("msteams", info, [])
        if backend._wire_muse():  # pragma: no branch - credential was just stored
            backend._connection_info = "Authenticated with Microsoft Teams (Muse-auth)"
            return json.dumps({"ok": True, "message": "MS Teams credentials saved (Muse-auth)."})
        error = json.dumps(  # pragma: no cover - defense in depth
            {"ok": False, "error": backend._connection_info}
        )
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Restore the pre-call config bytes; a failed swap must not leave
    # half-migrated state (the restored bytes are exactly what was
    # already on disk, so no new secret lands in the file).
    with contextlib.suppress(Exception):
        if prev_raw is None:
            _config.clear()
        else:
            # Atomic 0600 restore, serialized against every other
            # config writer via the shared config lock.
            with config_file_lock(_config.path):
                write_private_file(_config.path, prev_raw)
    backend._access_token = ""
    backend._muse = False
    backend._http = requests
    backend._tenant_id = ""
    backend._client_id = ""
    backend._client_secret = ""
    return error


class MSTeamsAgent(BaseChannelAgent):
    """Channel agent with Microsoft Teams Graph API tools."""

    channel_system_prompt = connect_prompt(
        "msteams",
        "Microsoft Teams",
        "authenticate_msteams(tenant_id=..., client_id=...) without a client_secret",
        "The app registration (https://portal.azure.com > App registrations) must "
        "allow public client flows and carry delegated Microsoft Graph permissions; "
        "the token then acts as the signed-in user. Passing client_secret instead "
        "configures the app-only flow directly.",
    ).lstrip()

    def __init__(self) -> None:
        super().__init__("MS Teams Agent")
        self._backend = MSTeamsChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # session (no network round trip); the client_secret never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._access_token = ""
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._tenant_id = cfg["tenant_id"]
            self._backend._client_id = cfg["client_id"]
            self._backend._client_secret = cfg["client_secret"]
            self._backend._bot_id = cfg.get("bot_id", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        if self._backend._muse:
            return bool(self._backend._access_token)
        return bool(self._backend._client_id)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_msteams_auth() -> str:
            """Check if MS Teams credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._muse:
                # The secret lives in the vault; the probe forces the
                # daemon-side token exchange through the boundary.
                ok, message = agent._backend._muse_probe()
                if ok:
                    return json.dumps(
                        {"ok": True, "message": "MS Teams authenticated (Muse-auth)."}
                    )
                return json.dumps({"ok": False, "error": message})
            if not agent._backend._client_id:  # pragma: no branch
                return _NOT_AUTHENTICATED
            try:
                token = agent._backend._token()
                if token:  # pragma: no branch
                    return json.dumps({"ok": True, "message": "MS Teams authenticated."})
                return json.dumps({"ok": False, "error": "Could not obtain access token."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_msteams(
            tenant_id: str,
            client_id: str,
            client_secret: str = "",
            bot_id: str = "",
            scopes: str = "",
        ) -> str:
            """Connect MS Teams by browser sign-in or with app credentials.

            Without ``client_secret`` this starts the Microsoft identity
            platform device code flow for the public app registration and
            returns a ``consent_required`` answer: give the user the
            verification URL and code (ask_user_question) to complete in
            their OWN browser, then call finish_msteams_auth().  The
            resulting delegated Graph token acts as the signed-in user.
            With ``client_secret`` the app-only client-credentials flow is
            configured directly.

            Args:
                tenant_id: Azure tenant ID (GUID, verified domain, or
                    "organizations"/"common").
                client_id: Application (client) ID of the app registration.
                    For the device code flow the registration must allow
                    public client flows and hold the delegated Graph
                    permissions.
                client_secret: Optional client secret for the app-only flow.
                bot_id: Optional bot user ID for message filtering.
                scopes: Space-separated delegated scopes for the device
                    code flow (default covers teams, channels, chats and
                    members; include offline_access for refresh).

            Returns:
                A consent_required JSON answer, a validation result, or an
                error message.
            """
            for val, name in [(tenant_id, "tenant_id"), (client_id, "client_id")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            tenant_id, client_id = tenant_id.strip(), client_id.strip()
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if client_secret.strip():
                # App credentials supersede any browser sign-in still
                # pending; drop it so a late approval cannot overwrite them.
                ConsentSession.cancel_active("msteams")
                if muse_auth_enabled():
                    reason = _invalid_ids_reason(tenant_id, client_id, client_secret.strip())
                    if reason:
                        return reason
                    return _muse_authenticate(
                        agent._backend,
                        tenant_id,
                        client_id,
                        _client_credential_info(tenant_id, client_id, client_secret.strip()),
                        bot_id.strip(),
                    )
                agent._backend._tenant_id = tenant_id
                agent._backend._client_id = client_id
                agent._backend._client_secret = client_secret.strip()
                agent._backend._bot_id = bot_id.strip()
                try:
                    token = agent._backend._token()
                    if not token:  # pragma: no branch
                        return json.dumps({"ok": False, "error": "Could not obtain access token."})
                    _config.save(
                        {
                            "tenant_id": tenant_id,
                            "client_id": client_id,
                            "client_secret": client_secret.strip(),
                            "bot_id": bot_id.strip(),
                        }
                    )
                    return json.dumps({"ok": True, "message": "MS Teams credentials saved."})
                except Exception as e:
                    return json.dumps({"ok": False, "error": str(e)})
            if not muse_auth_enabled():
                return json.dumps(
                    {
                        "ok": False,
                        "error": "The device code sign-in stores a refreshable delegated "
                        "token in the Muse vault, which is disabled (KISS_MUSE_AUTH=0). "
                        "Enable Muse-auth, or pass client_secret for the app-only flow.",
                    }
                )
            reason = _invalid_ids_reason(tenant_id, client_id)
            if reason:
                return reason
            try:
                session = DeviceFlowSession(
                    "msteams",
                    _device_provider(tenant_id),
                    client_id,
                    scopes.strip() or _DEFAULT_DELEGATED_SCOPES,
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})
            # The metadata the finish step needs rides on the session; the
            # stored configuration is untouched until the sign-in lands.
            session.options["tenant_id"] = tenant_id
            session.options["bot_id"] = bot_id.strip()
            session.register()
            return json.dumps(consent_required("msteams", "Microsoft Teams", session))

        def finish_msteams_auth() -> str:
            """Complete a browser sign-in started by authenticate_msteams().

            Call after the user reports that they entered the code and
            consented; the delegated token pair is validated with a Graph
            read through the Muse boundary and stored in the vault, where
            the daemon refreshes it with the public client ID.

            Returns:
                The validation result, a pending status while the user has
                not consented yet, or an error message.
            """
            session, status = ConsentSession.finish("msteams")
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "The user has not consented yet; ask them to finish "
                        "the sign-in, then call this tool again.",
                    }
                )
            if not isinstance(session, DeviceFlowSession) or session.result is None:
                return json.dumps({"ok": False, "error": f"MS Teams sign-in failed: {status}"})
            grant = TokenGrant.from_session(session)
            tenant_id = str(session.options.get("tenant_id") or "")
            if not grant.refresh_token:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "Microsoft issued no refresh token; include "
                        "offline_access in the scopes and sign in again.",
                    }
                )
            info = grant.vault_credential(
                session.provider.token_url, session.client_id, refresh_scope=session.scope
            )
            return _muse_authenticate(
                agent._backend,
                tenant_id,
                session.client_id,
                info,
                str(session.options.get("bot_id") or ""),
            )

        def clear_msteams_auth() -> str:
            """Clear the stored MS Teams credentials.

            Returns:
                Status message.
            """
            ConsentSession.cancel_active("msteams")
            _config.clear()
            agent._backend._client_id = ""
            agent._backend._client_secret = ""
            agent._backend._tenant_id = ""
            agent._backend._access_token = ""
            agent._backend._token_expiry = 0.0
            agent._backend._muse = False
            agent._backend._http = requests
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("msteams")
            return "MS Teams authentication cleared."

        return [check_msteams_auth, authenticate_msteams, finish_msteams_auth, clear_msteams_auth]


def _make_backend() -> MSTeamsChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MSTeamsChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    if muse_auth_enabled():
        try:
            wired = backend._wire_muse()
        except MuseAuthError:
            wired = False
        if wired:
            return backend
        print("Not authenticated. Run: kiss-msteams -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-msteams -t 'authenticate'")
        sys.exit(1)
    backend._tenant_id = cfg["tenant_id"]
    backend._client_id = cfg["client_id"]
    backend._client_secret = cfg["client_secret"]
    backend._bot_id = cfg.get("bot_id", "")
    return backend


def main() -> None:
    """Run the MSTeamsAgent from the command line with chat persistence."""
    channel_main(
        MSTeamsAgent,
        "kiss-msteams",
        channel_name="MS Teams",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Microsoft Teams channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return MSTeamsAgent()._get_tools()


if __name__ == "__main__":
    main()
