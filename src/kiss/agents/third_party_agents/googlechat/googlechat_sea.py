# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Chat Agent — channel agent with Google Chat API tools.

Provides access to Google Chat as the user, through Composio (see
:mod:`._composio_google`), or as a Chat bot with a service account
stored at ``$KISS_HOME/third_party_agents/googlechat/service_account.json``.
Composio has no managed Google Chat app, so user sign-in needs a custom
Composio auth config whose ID is set in
``KISS_COMPOSIO_AUTH_CONFIG_GOOGLECHAT``.

Usage::

    agent = GoogleChatAgent()
    agent.run(prompt_template="List all spaces I'm a member of")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    write_private_file,
)
from kiss.agents.third_party_agents._composio_google import ComposioHttp, connected_account_id
from kiss.agents.third_party_agents._google_workspace_utils import (
    google_auth_prompt,
    make_google_auth_tools,
)
from kiss.core.config import kiss_home

_SERVICE = "googlechat"

_SCOPES = [
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces",
    "https://www.googleapis.com/auth/chat.memberships",
]


class GooglechatSea(BaseSea):
    """The ``/googlechat`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/googlechat help``."""
        return (
            "Lists, gets and creates Google Chat spaces, lists their members, and reads, posts, "
            "edits and deletes messages as the signed-in user (via Composio) or as a Chat bot "
            'with a service account; use `run_agent(agent="googlechat", task="...")` or the '
            "`kiss-gchat -t '...'` CLI."
        )

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the Google Chat channel tools (the SEA ``tools`` method).

        Called by the kiss-web daemon when this module's path is passed as
        the API's ``extension_agent_path``: builds a fresh agent from the
        credentials recorded under ``$KISS_HOME`` and returns its
        authentication and backend tools.
        """
        return tools + GoogleChatAgent()._get_tools()

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run as a ``channel`` worker (``kiss.server.sorcar.run`` agent-script contract).

        No git lifecycle, nothing inherited from the calling task, the
        channel preamble in the system prompt (see
        :mod:`kiss.agents.sorcar.sea_settings`).
        """
        return settings | {"kind": "channel"}

    def system_prompt(self, system_prompt: str) -> str:
        """Return the channel guidance appended to the run's system prompt."""
        return system_prompt + "\n\n" + GoogleChatAgent.channel_system_prompt


def _gchat_dir() -> Path:
    """Return the Google Chat credential directory, honoring ``KISS_HOME``.

    Returns:
        Path to ``$KISS_HOME/third_party_agents/googlechat`` (defaults to
        ``$KISS_HOME/third_party_agents/googlechat``).
    """
    return kiss_home() / "third_party_agents" / "googlechat"


def _service_account_path() -> Path:
    """Return the path to the service account JSON file."""
    return _gchat_dir() / "service_account.json"


def _load_service(sa_path: str = "") -> Any:
    """Load a Google Chat API service from a service account or Composio.

    A service account (Chat bot) wins when its JSON file exists;
    otherwise the recorded Composio connection is used.

    Args:
        sa_path: Path to a service account JSON file. If empty, the
            default ``service_account.json`` is used when present.

    Returns:
        Google Chat API service resource, or None on failure.
    """
    from googleapiclient.discovery import build

    sa_file = Path(sa_path) if sa_path else _service_account_path()
    if sa_file.exists():
        try:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                str(sa_file), scopes=_SCOPES
            )
            return build("chat", "v1", credentials=creds)
        except Exception:
            pass

    if sa_path or not connected_account_id(_SERVICE):
        return None
    return build("chat", "v1", http=ComposioHttp(_SERVICE), static_discovery=True)


class GoogleChatChannelBackend(ToolMethodBackend):
    """Channel backend for Google Chat API."""

    def __init__(self) -> None:
        self._service: Any = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Google Chat."""
        service = _load_service()
        if not service:  # pragma: no branch
            self._connection_info = "No Google Chat credentials found."
            return False
        self._service = service
        self._connection_info = "Authenticated with Google Chat"
        return True

    def find_channel(self, name: str) -> str | None:
        """Find a Google Chat space by display name."""
        if not self._service:  # pragma: no branch
            return None
        try:
            resp = self._service.spaces().list(pageSize=100).execute()
            for space in resp.get("spaces", []):  # pragma: no branch
                if space.get("displayName") == name:  # pragma: no branch
                    return str(space["name"])
        except Exception:
            pass
        return None

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll a Google Chat space for new messages."""
        if not self._service or not channel_id:  # pragma: no branch
            return [], oldest
        try:
            kwargs: dict[str, Any] = {
                "parent": channel_id,
                "pageSize": limit,
                "orderBy": "createTime asc",
            }
            if oldest and oldest != "0":
                kwargs["filter"] = f'createTime > "{oldest}"'
            resp = self._service.spaces().messages().list(**kwargs).execute()
            raw_msgs = resp.get("messages", [])
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for msg in raw_msgs:  # pragma: no branch
                ts = msg.get("createTime", "")
                new_oldest = ts
                messages.append(
                    {
                        "ts": ts,
                        "user": msg.get("sender", {}).get("name", ""),
                        "text": msg.get("text", ""),
                        "name": msg.get("name", ""),
                        "thread": msg.get("thread", {}).get("name", ""),
                        "thread_ts": msg.get("thread", {}).get("name", ""),
                    }
                )
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Google Chat message."""
        if not self._service:  # pragma: no branch
            return
        body: dict[str, Any] = {"text": text}
        if thread_ts:  # pragma: no branch
            body["thread"] = {"name": thread_ts}
        self._service.spaces().messages().create(parent=channel_id, body=body).execute()

    def list_spaces(self, page_size: int = 20, page_token: str = "") -> str:
        """List Google Chat spaces (rooms and DMs).

        Args:
            page_size: Maximum spaces to return. Default: 20.
            page_token: Pagination token from a previous response.

        Returns:
            JSON string with space list (name, displayName, type).
        """
        assert self._service is not None
        try:
            kwargs: dict[str, Any] = {"pageSize": page_size}
            if page_token:  # pragma: no branch
                kwargs["pageToken"] = page_token
            resp = self._service.spaces().list(**kwargs).execute()
            spaces = [
                {
                    "name": s.get("name", ""),
                    "display_name": s.get("displayName", ""),
                    "type": s.get("type", ""),
                }
                for s in resp.get("spaces", [])
            ]
            result: dict[str, Any] = {"ok": True, "spaces": spaces}
            if resp.get("nextPageToken"):  # pragma: no branch
                result["next_page_token"] = resp["nextPageToken"]
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_space(self, space_name: str) -> str:
        """Get information about a Google Chat space.

        Args:
            space_name: Space resource name (e.g. "spaces/ABCDEF").

        Returns:
            JSON string with space details.
        """
        assert self._service is not None
        try:
            space = self._service.spaces().get(name=space_name).execute()
            return json.dumps({"ok": True, "space": space}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_members(self, space_name: str, page_size: int = 20, page_token: str = "") -> str:
        """List members of a Google Chat space.

        Args:
            space_name: Space resource name.
            page_size: Maximum members to return. Default: 20.
            page_token: Pagination token.

        Returns:
            JSON string with member list.
        """
        assert self._service is not None
        try:
            kwargs: dict[str, Any] = {"parent": space_name, "pageSize": page_size}
            if page_token:  # pragma: no branch
                kwargs["pageToken"] = page_token
            resp = self._service.spaces().members().list(**kwargs).execute()
            members = resp.get("memberships", [])
            result: dict[str, Any] = {"ok": True, "members": members}
            if resp.get("nextPageToken"):  # pragma: no branch
                result["next_page_token"] = resp["nextPageToken"]
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
        space_name: str,
        page_size: int = 20,
        page_token: str = "",
        filter: str = "",
    ) -> str:
        """List messages in a Google Chat space.

        Args:
            space_name: Space resource name (e.g. "spaces/ABCDEF").
            page_size: Maximum messages to return. Default: 20.
            page_token: Pagination token.
            filter: Optional filter (e.g. 'createTime > "2024-01-01T00:00:00Z"').

        Returns:
            JSON string with message list.
        """
        assert self._service is not None
        try:
            kwargs: dict[str, Any] = {
                "parent": space_name,
                "pageSize": page_size,
                "orderBy": "createTime desc",
            }
            if page_token:  # pragma: no branch
                kwargs["pageToken"] = page_token
            if filter:  # pragma: no branch
                kwargs["filter"] = filter
            resp = self._service.spaces().messages().list(**kwargs).execute()
            messages = [
                {
                    "name": m.get("name", ""),
                    "text": m.get("text", ""),
                    "sender": m.get("sender", {}).get("displayName", ""),
                    "create_time": m.get("createTime", ""),
                    "thread": m.get("thread", {}).get("name", ""),
                }
                for m in resp.get("messages", [])
            ]
            result: dict[str, Any] = {"ok": True, "messages": messages}
            if resp.get("nextPageToken"):  # pragma: no branch
                result["next_page_token"] = resp["nextPageToken"]
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_message(self, message_name: str) -> str:
        """Get a specific Google Chat message.

        Args:
            message_name: Message resource name (e.g. "spaces/X/messages/Y").

        Returns:
            JSON string with message details.
        """
        assert self._service is not None
        try:
            msg = self._service.spaces().messages().get(name=message_name).execute()
            return json.dumps({"ok": True, "message": msg}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, space_name: str, text: str, thread_key: str = "") -> str:
        """Send a message to a Google Chat space.

        Args:
            space_name: Space resource name (e.g. "spaces/ABCDEF").
            text: Message text.
            thread_key: Optional thread key to reply in an existing thread.

        Returns:
            JSON string with ok status and message name.
        """
        assert self._service is not None
        try:
            body: dict[str, Any] = {"text": text}
            if thread_key:  # pragma: no branch
                body["thread"] = {"name": thread_key}
            msg = self._service.spaces().messages().create(parent=space_name, body=body).execute()
            return json.dumps(
                {
                    "ok": True,
                    "name": msg.get("name", ""),
                    "create_time": msg.get("createTime", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def update_message(self, message_name: str, text: str) -> str:
        """Update an existing Google Chat message.

        Args:
            message_name: Message resource name.
            text: New message text.

        Returns:
            JSON string with ok status.
        """
        assert self._service is not None
        try:
            self._service.spaces().messages().update(
                name=message_name,
                body={"text": text},
                updateMask="text",
            ).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, message_name: str) -> str:
        """Delete a Google Chat message.

        Args:
            message_name: Message resource name.

        Returns:
            JSON string with ok status.
        """
        assert self._service is not None
        try:
            self._service.spaces().messages().delete(name=message_name).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_space(self, display_name: str, space_type: str = "SPACE") -> str:
        """Create a new Google Chat space.

        Args:
            display_name: Space display name.
            space_type: Space type ("SPACE" or "GROUP_CHAT"). Default: "SPACE".

        Returns:
            JSON string with space name and display name.
        """
        assert self._service is not None
        try:
            space = (
                self._service.spaces()
                .create(
                    body={
                        "displayName": display_name,
                        "spaceType": space_type,
                    }
                )
                .execute()
            )
            return json.dumps(
                {
                    "ok": True,
                    "name": space.get("name", ""),
                    "display_name": space.get("displayName", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GoogleChatAgent(BaseChannelAgent):
    """Channel agent with Google Chat API tools.

    Example::

        agent = GoogleChatAgent()
        result = agent.run(prompt_template="List all spaces")
    """

    channel_system_prompt = google_auth_prompt(_SERVICE, "Google Chat") + (
        " Composio has no managed Google Chat app: user sign-in needs a custom "
        "Composio auth config (created with the user's own Google OAuth client "
        "at https://dashboard.composio.dev) whose ID is set in the "
        "KISS_COMPOSIO_AUTH_CONFIG_GOOGLECHAT environment variable. To act as a "
        "Chat bot instead, call authenticate_googlechat_service_account() with "
        "the path of the bot's service account JSON key."
    )

    def __init__(self) -> None:
        super().__init__("Google Chat Agent")
        self._backend = GoogleChatChannelBackend()
        service = _load_service()
        if service:  # pragma: no branch
            self._backend._service = service

    def _is_authenticated(self) -> bool:
        """Return True if a service account is loaded or a Composio connection is set up."""
        return self._backend._service is not None or bool(connected_account_id(_SERVICE))

    def _forget_service_account(self) -> None:
        """Drop the loaded Chat service and delete the stored service-account key."""
        self._backend._service = None
        _service_account_path().unlink(missing_ok=True)

    def _get_auth_tools(self) -> list:
        """Return the Composio sign-in tools plus the service-account tool."""
        agent = self

        def authenticate_googlechat_service_account(service_account_json_path: str = "") -> str:
            """Authenticate Google Chat as a Chat bot with a service account key.

            A key outside the default location is copied there so later
            runs find it.

            Args:
                service_account_json_path: Path to the service account JSON
                    key. If empty, the default service_account.json is used.

            Returns:
                JSON with ok status, or an error message.
            """
            source = Path(service_account_json_path) if service_account_json_path else (
                _service_account_path()
            )
            service = _load_service(str(source))
            if service is None:
                return json.dumps({
                    "ok": False,
                    "error": f"Could not load a service account key from {source}.",
                })
            if source.resolve() != _service_account_path().resolve():
                write_private_file(_service_account_path(), source.read_text())
            agent._backend._service = service
            return json.dumps({"ok": True, "message": "Google Chat service account loaded."})

        return [
            *make_google_auth_tools(
                self, _SERVICE, "Google Chat", self._backend.connect, self._forget_service_account
            ),
            authenticate_googlechat_service_account,
        ]


def _make_backend() -> GoogleChatChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = GoogleChatChannelBackend()
    service = _load_service()
    if not service:  # pragma: no branch
        print("Not authenticated. Run: kiss-gchat -t 'authenticate'")
        sys.exit(1)
    backend._service = service
    return backend


def main() -> None:
    """Run the GoogleChatAgent from the command line with chat persistence."""
    channel_main(
        GoogleChatAgent,
        "kiss-gchat",
        channel_name="Google Chat",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
