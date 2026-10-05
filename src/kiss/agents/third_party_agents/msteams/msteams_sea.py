# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Microsoft Teams Agent — channel agent with MS Teams Graph API tools.

Provides authenticated access to Microsoft Teams through Microsoft Graph.
Connects like the Muse app: ``authenticate_msteams()`` starts the
Microsoft identity platform device code flow for the KISS-owned
multi-tenant public app (``oauth_client_id("msteams")``, overridable
with ``$KISS_MSTEAMS_CLIENT_ID``) against the ``organizations`` tenant
(or a given single tenant) and hands back
``https://microsoft.com/devicelogin`` plus a short code; the user signs
in and clicks Accept in their own browser and ``finish_msteams_auth()``
stores the delegated token pair in the Muse vault, whose daemon
refreshes it (no client secret exists).  Every Graph call acts as the
signed-in user.  Stores non-secret metadata (``tenant_id``, optional
``bot_id``) in ``~/.kiss/third_party_agents/msteams/config.json``.

Usage::

    agent = MSTeamsAgent()
    agent.run(prompt_template="List all teams I'm a member of")
"""

from __future__ import annotations

import json
import os
import re
import sys
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
from kiss.agents.third_party_agents._oauth_apps import (
    missing_client_id_error,
    oauth_client_id,
)
from kiss.core.config import kiss_home

_MSTEAMS_DIR = kiss_home() / "third_party_agents" / "msteams"
_config = ChannelConfig(_MSTEAMS_DIR, ("tenant_id",))
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_DEFAULT_LOGIN_BASE = "https://login.microsoftonline.com"
# The KISS app is multi-tenant: ``organizations`` lets a work or school
# account from any directory sign in.
_DEFAULT_TENANT = "organizations"
# Delegated Graph permissions the device code sign-in asks the user to
# consent to; ``offline_access`` yields the refresh token the daemon
# uses to renew the one-hour access token.
_DEFAULT_DELEGATED_SCOPES = (
    "offline_access User.Read Team.ReadBasic.All Channel.ReadBasic.All "
    "ChannelMessage.Read.All ChannelMessage.Send Chat.ReadWrite TeamMember.Read.All"
)
_MUSE_REQUIRED = (
    "MS Teams sign-in stores a refreshable delegated token in the Muse vault, "
    "which is disabled (KISS_MUSE_AUTH=0); enable Muse-auth to connect."
)
_NOT_AUTHENTICATED = (
    "Not authenticated with MS Teams. Call authenticate_msteams() to sign in the "
    "way the Muse app connects: it opens https://microsoft.com/devicelogin for the "
    "user (follow its 'instructions') with a short code they enter after signing "
    "in with their work or school account and clicking Accept; then call "
    "finish_msteams_auth(). Never ask for the user's Microsoft password or 2FA code."
)

# Azure tenant IDs are GUIDs or verified domains (contoso.onmicrosoft.com):
# one URL path segment.  Rejecting anything else keeps the composed token
# URL's host and path shape fixed (no `/`, `?`, `#`, or whitespace).
_TENANT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


def description() -> str:
    """Return the one-sentence help text shown by ``/msteams help``."""
    return (
        "Lists teams, channels, chats and members and reads or posts channel and chat "
        "messages in Microsoft Teams through Microsoft Graph as the user signed in with the "
        'device code flow; use it with `run_agent(agent="msteams", task="...")` or the '
        "`kiss-msteams -t '...'` CLI."
    )


def _login_base() -> str:
    """Return the Azure AD login base URL, resolved per call.

    ``MSTEAMS_LOGIN_BASE`` lets tests point the daemon-side token
    exchange at a loopback token endpoint (the Muse daemon only
    accepts the real pinned host or loopback).

    Returns:
        The login base URL without a trailing slash.
    """
    return os.environ.get("MSTEAMS_LOGIN_BASE", "") or _DEFAULT_LOGIN_BASE


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
    """Return the config metadata parsed WITHOUT type coercion.

    Returns:
        The parsed config dict, or ``{}`` when missing or not an object.
    """
    try:
        data = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class MSTeamsChannelBackend(ToolMethodBackend):
    """Channel backend for Microsoft Teams via Graph API."""

    def __init__(self, graph_base: str = _GRAPH_BASE) -> None:
        self._tenant_id: str = ""
        self._bot_id: str = ""
        self._access_token: str = ""
        self._connection_info: str = ""
        self._graph_base: str = graph_base
        self._http: Any = requests

    def _wire_muse(self) -> bool:
        """Acquire an MS Teams surrogate and wire the boundary session.

        The delegated token pair lives in the Muse vault (stored by
        ``finish_msteams_auth``); this process only holds a surrogate
        and the daemon refreshes the access token at the boundary.  No
        network round trip happens here.

        Returns:
            True when the backend holds a surrogate and boundary session.
        """
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
        )

        handle = mint_surrogate("msteams")
        if handle is None:
            self._connection_info = "No MS Teams credential in the Muse vault."
            return False
        cfg = _raw_config()
        self._tenant_id = str(cfg.get("tenant_id") or "")
        self._bot_id = str(cfg.get("bot_id") or "")
        self._access_token = handle.token
        self._http = MuseBoundarySession("msteams")
        return True

    def _muse_probe(self) -> tuple[bool, str]:
        """Prove the daemon can use this credential for a Graph read.

        See :func:`_graph_probe`.

        Returns:
            ``(ok, message)``.
        """
        return _graph_probe("msteams", self._access_token, self._graph_base)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.get(
            f"{self._graph_base}{path}", headers=self._headers(), params=params, timeout=30
        )
        result: dict[str, Any] = resp.json()
        # Mark HTTP errors so tools can surface Sentinel denials and
        # Graph failures.
        if resp.status_code >= 400:
            result["ok"] = False
        return result

    def _post(self, path: str, body: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = self._http.post(
            f"{self._graph_base}{path}", headers=self._headers(), json=body, timeout=30
        )
        result: dict[str, Any] = resp.json() if resp.content else {"ok": True}
        if resp.status_code >= 400:
            result["ok"] = False
        return result

    def connect(self) -> bool:
        """Wire the vault credential and validate it with one Graph read.

        Returns:
            True when the stored delegated token pair works.
        """
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
        from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

        if not muse_auth_enabled():
            self._connection_info = _MUSE_REQUIRED
            return False
        try:
            if not self._wire_muse():
                return False
        except MuseAuthError as e:
            self._connection_info = f"MS Teams auth failed: {e}"
            return False
        ok, message = self._muse_probe()
        self._connection_info = message if ok else f"MS Teams auth failed: {message}"
        return ok

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

    The token pair lives only in the vault, so one small Graph read
    flows through the audited boundary (the daemon refreshes an expired
    access token first).  A real Graph response — even a 403 for a
    missing permission — proves the credential works, but a 401 means
    Graph rejected the token itself; a failed refresh or a Sentinel
    denial reports its reason.

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
    # Any other answer (including a 5xx outage) arrived with a token
    # Graph accepted; a failed refresh surfaces as a MuseAuthError above.
    return True, "Authenticated with Microsoft Teams (Muse-auth)"


def _probe_candidate_credentials(backend: MSTeamsChannelBackend, info: dict[str, str]) -> str:
    """Validate a candidate token pair without touching the live entry.

    The candidate is enrolled under the scratch service
    ``msteams-pending`` (which inherits the Graph host allowlist and
    the pinned token-endpoint list), one probe read exercises it
    through the daemon, and the scratch entry is removed again — so
    a rejected rotation can never destroy an existing working
    ``msteams`` vault credential.

    Args:
        backend: The agent's MS Teams backend (supplies the Graph base).
        info: The candidate ``oauth2_refresh_token`` payload.

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


def _muse_authenticate(
    backend: MSTeamsChannelBackend,
    tenant_id: str,
    info: dict[str, Any],
    bot_id: str,
) -> str:
    """Enroll an MS Teams token pair into the Muse vault and validate it.

    *info* is the ``oauth2_refresh_token`` pair a device code sign-in
    produced.  The candidate is validated FIRST, against a
    scratch ``-pending`` enrollment, so a rejected credential mutates
    nothing — neither the config nor an existing working vault
    credential.  Only proven credentials replace the live enrollment:
    the non-secret ``tenant_id``/``bot_id`` metadata is
    written to ``config.json``, and the secret material goes straight
    into the vault, never to disk outside it.  If the swap itself fails
    midway, the pre-call config bytes are restored — the vault is never
    cleared, because whichever credential it holds at that point (the
    untouched old one or the just-validated new one) is worth keeping.

    Args:
        backend: The agent's MS Teams backend to (re)wire.
        tenant_id: Tenant the user signed in to (already validated).
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
        meta = {"tenant_id": tenant_id}
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
    backend._http = requests
    backend._tenant_id = ""
    return error


class MSTeamsAgent(BaseChannelAgent):
    """Channel agent with Microsoft Teams Graph API tools."""

    channel_system_prompt = connect_prompt(
        "msteams",
        "Microsoft Teams",
        "authenticate_msteams()",
        "It uses the KISS Microsoft Entra app, so the user only signs in with a "
        "work or school account and clicks Accept; Graph calls then act as that "
        "user. Pass tenant_id=... only to restrict sign-in to one directory.",
    ).lstrip()

    def __init__(self) -> None:
        super().__init__("MS Teams Agent")
        self._backend = MSTeamsChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if not muse_auth_enabled():
            return
        from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

        # Wire a vault surrogate and the boundary session (no network
        # round trip).  A daemon failure leaves the agent constructible
        # (fail closed, tokenless) so its authenticate/clear tools stay
        # available.
        try:
            self._backend._wire_muse()
        except MuseAuthError as e:
            self._backend._access_token = ""
            self._backend._connection_info = f"Muse-auth wiring failed: {e}"

    def _is_authenticated(self) -> bool:
        """Return True if the backend holds a vault surrogate."""
        return bool(self._backend._access_token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_msteams_auth() -> str:
            """Check if MS Teams is connected and the credential works.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._access_token:
                return _NOT_AUTHENTICATED
            ok, message = agent._backend._muse_probe()
            if ok:
                return json.dumps({"ok": True, "message": "MS Teams authenticated (Muse-auth)."})
            return json.dumps({"ok": False, "error": message})

        def authenticate_msteams(tenant_id: str = "", bot_id: str = "", scopes: str = "") -> str:
            """Connect MS Teams by browser sign-in (device code flow).

            Starts the Microsoft identity platform device code flow with
            the KISS-owned multi-tenant app ($KISS_MSTEAMS_CLIENT_ID
            overrides its client ID) and returns a ``consent_required``
            answer whose ``instructions`` say how to hand the page and
            code to the user (ask_user_question); the USER completes it,
            then call finish_msteams_auth().  The resulting delegated Graph
            token acts as the signed-in user.

            Args:
                tenant_id: Optional Azure tenant (GUID or verified domain)
                    to restrict sign-in to one directory; default
                    "organizations" accepts any work or school account.
                bot_id: Optional user ID whose messages poll mode ignores.
                scopes: Space-separated delegated scopes (default covers
                    teams, channels, chats and members, plus
                    offline_access for refresh).

            Returns:
                A consent_required JSON answer or an error message.
            """
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if not muse_auth_enabled():
                return json.dumps({"ok": False, "error": _MUSE_REQUIRED})
            client_id = oauth_client_id("msteams")
            if not client_id:
                return json.dumps(
                    {"ok": False, "error": missing_client_id_error("msteams", "Microsoft Teams")}
                )
            tenant_id = tenant_id.strip() or _DEFAULT_TENANT
            if not _TENANT_ID_RE.fullmatch(tenant_id):
                return json.dumps(
                    {
                        "ok": False,
                        "error": "tenant_id must be a GUID or verified domain "
                        "(letters, digits, '.', '_', '-').",
                    }
                )
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
            agent._backend._tenant_id = ""
            agent._backend._access_token = ""
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


def main() -> None:
    """Run the MSTeamsAgent from the command line with chat persistence."""
    channel_main(
        MSTeamsAgent,
        "kiss-msteams",
        channel_name="MS Teams",
        make_backend=_make_backend,
    )


def add_to_tools() -> list:
    """Return the Microsoft Teams channel tools (``kiss.server.sorcar.run`` agent-script contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``extension_agent_path``: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return MSTeamsAgent()._get_tools()


def settings() -> dict:
    """Run as a ``channel`` worker (``kiss.server.sorcar.run`` agent-script contract).

    No git lifecycle, nothing inherited from the calling task, the
    channel preamble in the system prompt (see
    :mod:`kiss.agents.sorcar.sea_settings`).
    """
    return {"kind": "channel"}


def add_to_system_prompt() -> str:
    """Return the channel guidance appended to the run's system prompt."""
    return MSTeamsAgent.channel_system_prompt


if __name__ == "__main__":
    main()
