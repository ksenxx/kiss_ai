"""Microsoft Teams Agent — StatefulSorcarAgent extension with MS Teams Graph API tools.

Provides authenticated access to Microsoft Teams via Azure AD client credentials.
Stores config in ``~/.kiss/channels/msteams/config.json``.

Usage::

    agent = MSTeamsAgent()
    agent.run(prompt_template="List all teams I'm a member of")
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_MSTEAMS_DIR = Path.home() / ".kiss" / "channels" / "msteams"
_config = ChannelConfig(_MSTEAMS_DIR, ("tenant_id", "client_id"))
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


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

    def __init__(self) -> None:
        self._tenant_id: str = ""
        self._client_id: str = ""
        self._client_secret: str = ""
        self._bot_id: str = ""
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._connection_info: str = ""

    def _token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if time.time() >= self._token_expiry - 60:  # pragma: no branch
            self._access_token = _get_access_token(
                self._tenant_id, self._client_id, self._client_secret
            )
            self._token_expiry = time.time() + 3600
        return self._access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.get(
            f"{_GRAPH_BASE}{path}", headers=self._headers(), params=params, timeout=30
        )
        return resp.json()  # type: ignore[no-any-return]

    def _post(self, path: str, body: dict | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
        resp = requests.post(f"{_GRAPH_BASE}{path}", headers=self._headers(), json=body, timeout=30)
        return resp.json() if resp.content else {"ok": True}  # type: ignore[no-any-return]

    def connect(self) -> bool:
        """Authenticate with Microsoft Graph API."""
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
        # channel_id format: "team_id:channel_id"
        if not channel_id or ":" not in channel_id:  # pragma: no branch
            return [], oldest
        team_id, chan_id = channel_id.split(":", 1)
        try:
            params: dict[str, Any] = {"$top": limit, "$orderby": "lastModifiedDateTime asc"}
            if oldest:  # pragma: no branch
                params["$filter"] = f"lastModifiedDateTime gt {oldest}"
            result = self._get(f"/teams/{team_id}/channels/{chan_id}/messages", params=params)
            msgs = result.get("value", [])
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for msg in msgs:  # pragma: no branch
                ts = msg.get("lastModifiedDateTime", "")
                new_oldest = ts
                body = msg.get("body", {})
                messages.append(
                    {
                        "ts": ts,
                        "user": msg.get("from", {}).get("user", {}).get("id", ""),
                        "text": body.get("content", ""),
                        "id": msg.get("id", ""),
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

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        oldest = ""

        def poll() -> list[dict[str, Any]]:
            nonlocal oldest
            msgs, oldest = self.poll_messages(channel_id, oldest)
            return msgs

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=3.0,
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

    def list_channels(self, team_id: str) -> str:
        """List channels in a Microsoft Team.

        Args:
            team_id: Team ID.

        Returns:
            JSON string with channel list (id, displayName, membershipType).
        """
        try:
            result = self._get(f"/teams/{team_id}/channels")
            channels = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("displayName", ""),
                    "type": c.get("membershipType", ""),
                    "description": c.get("description", ""),
                }
                for c in result.get("value", [])
            ]
            return json.dumps({"ok": True, "channels": channels}, indent=2)[:8000]
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


class MSTeamsAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Microsoft Teams Graph API tools."""

    def __init__(self) -> None:
        super().__init__("MS Teams Agent")
        self._backend = MSTeamsChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._tenant_id = cfg["tenant_id"]
            self._backend._client_id = cfg["client_id"]
            self._backend._client_secret = cfg["client_secret"]
            self._backend._bot_id = cfg.get("bot_id", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._client_id)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_msteams_auth() -> str:
            """Check if MS Teams credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if not agent._backend._client_id:  # pragma: no branch
                return (
                    "Not authenticated with MS Teams. Use authenticate_msteams() to configure.\n"
                    "You need: tenant_id, client_id, client_secret from Azure portal."
                )
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
            client_secret: str,
            bot_id: str = "",
        ) -> str:
            """Store and validate MS Teams Azure AD credentials.

            Args:
                tenant_id: Azure tenant ID.
                client_id: Azure app client ID.
                client_secret: Azure app client secret.
                bot_id: Optional bot user ID for message filtering.

            Returns:
                Validation result or error message.
            """
            cred_pairs = [
                (tenant_id, "tenant_id"),
                (client_id, "client_id"),
                (client_secret, "client_secret"),
            ]
            for val, name in cred_pairs:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            agent._backend._tenant_id = tenant_id.strip()
            agent._backend._client_id = client_id.strip()
            agent._backend._client_secret = client_secret.strip()
            agent._backend._bot_id = bot_id.strip()
            try:
                token = agent._backend._token()
                if not token:  # pragma: no branch
                    return json.dumps({"ok": False, "error": "Could not obtain access token."})
                _config.save(
                    {
                        "tenant_id": tenant_id.strip(),
                        "client_id": client_id.strip(),
                        "client_secret": client_secret.strip(),
                        "bot_id": bot_id.strip(),
                    }
                )
                return json.dumps({"ok": True, "message": "MS Teams credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_msteams_auth() -> str:
            """Clear the stored MS Teams credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._client_id = ""
            agent._backend._client_secret = ""
            agent._backend._tenant_id = ""
            return "MS Teams authentication cleared."

        return [check_msteams_auth, authenticate_msteams, clear_msteams_auth]


def _make_backend() -> MSTeamsChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MSTeamsChannelBackend()
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


if __name__ == "__main__":
    main()
