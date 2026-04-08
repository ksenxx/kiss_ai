"""Google Chat Agent — StatefulSorcarAgent extension with Google Chat API tools.

Provides authenticated access to Google Chat via Service Account or OAuth2.
Stores credentials in ``~/.kiss/channels/googlechat/``.

Usage::

    agent = GoogleChatAgent()
    agent.run(prompt_template="List all spaces I'm a member of")
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import is_headless_environment, wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)

_GCHAT_DIR = Path.home() / ".kiss" / "channels" / "googlechat"
_SCOPES = [
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces",
    "https://www.googleapis.com/auth/chat.memberships",
]


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def _token_path() -> Path:
    """Return the path to the stored OAuth2 token file."""
    return _GCHAT_DIR / "token.json"


def _credentials_path() -> Path:
    """Return the path to the OAuth2 client credentials file."""
    return _GCHAT_DIR / "credentials.json"


def _service_account_path() -> Path:
    """Return the path to the service account JSON file."""
    return _GCHAT_DIR / "service_account.json"


def _load_service(sa_path: str = "") -> Any:
    """Load a Google Chat API service using service account or OAuth2.

    Args:
        sa_path: Path to service account JSON. If empty, uses OAuth2.

    Returns:
        Google Chat API service resource, or None on failure.
    """
    from googleapiclient.discovery import build

    sa_file = Path(sa_path) if sa_path else _service_account_path()
    if sa_file.exists():  # pragma: no branch
        try:
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                str(sa_file), scopes=_SCOPES
            )
            return build("chat", "v1", credentials=creds)
        except Exception:
            pass

    # Fall back to OAuth2
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_file = _token_path()
    if not token_file.exists():  # pragma: no branch
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(token_file), _SCOPES)
        if creds.valid:  # pragma: no branch
            return build("chat", "v1", credentials=creds)
        if creds.expired and creds.refresh_token:  # pragma: no branch
            creds.refresh(Request())
            token_file.write_text(creds.to_json())
            if sys.platform != "win32":  # pragma: no branch
                token_file.chmod(0o600)
            return build("chat", "v1", credentials=creds)
    except Exception:
        pass
    return None


def _run_oauth_flow() -> Any:
    """Run OAuth2 flow for Google Chat.

    In headless/Docker environments, falls back to ``run_console()`` which
    prints a URL and reads the auth code from stdin instead of opening a
    browser window.

    Returns:
        Google Chat API service resource, or None on failure.
    """
    from typing import cast

    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds_path = _credentials_path()
    if not creds_path.exists():  # pragma: no branch
        return None
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
    if is_headless_environment():  # pragma: no branch
        creds = cast(Credentials, flow.run_console())
    else:
        creds = cast(Credentials, flow.run_local_server(port=0))
    token_file = _token_path()
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json())
    if sys.platform != "win32":  # pragma: no branch
        token_file.chmod(0o600)
    return build("chat", "v1", credentials=creds)


def _clear_config() -> None:
    """Delete the stored Google Chat credentials."""
    for path in [_token_path()]:
        if path.exists():  # pragma: no branch
            path.unlink()


# ---------------------------------------------------------------------------
# GoogleChatChannelBackend
# ---------------------------------------------------------------------------


class GoogleChatChannelBackend(ToolMethodBackend):
    """ChannelBackend implementation for Google Chat API."""

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

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

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

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for Google Chat — bots are added by admins."""

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
            if oldest:  # pragma: no branch
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

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if a message is from a bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    # -------------------------------------------------------------------
    # Google Chat API tool methods
    # -------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GoogleChatAgent
# ---------------------------------------------------------------------------


class GoogleChatAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Google Chat API tools.

    Example::

        agent = GoogleChatAgent()
        result = agent.run(prompt_template="List all spaces")
    """

    def __init__(self) -> None:
        super().__init__("Google Chat Agent")
        self._backend = GoogleChatChannelBackend()
        service = _load_service()
        if service:  # pragma: no branch
            self._backend._service = service

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._service is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_googlechat_auth() -> str:
            """Check if Google Chat credentials are configured and valid.

            Returns:
                Authentication status or instructions for how to authenticate.
            """
            if agent._backend._service is None:  # pragma: no branch
                sa_exists = _service_account_path().exists()
                creds_exists = _credentials_path().exists()
                if sa_exists:  # pragma: no branch
                    return (
                        "Not authenticated. A service_account.json file exists. "
                        "Call authenticate_googlechat() to load it."
                    )
                if creds_exists:  # pragma: no branch
                    return (
                        "Not authenticated. A credentials.json file exists. "
                        "Call authenticate_googlechat() to start OAuth2 flow."
                    )
                return (
                    "Not authenticated with Google Chat. To set up:\n"
                    "Option 1 (Bot): Save service account JSON to "
                    f"{_service_account_path()}\n"
                    "Option 2 (OAuth): Save credentials JSON to "
                    f"{_credentials_path()}\n"
                    "Then call authenticate_googlechat()."
                )
            try:
                resp = agent._backend._service.spaces().list(pageSize=1).execute()
                return json.dumps({"ok": True, "space_count": len(resp.get("spaces", []))})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_googlechat(service_account_json_path: str = "") -> str:
            """Authenticate with Google Chat using service account or OAuth2.

            Args:
                service_account_json_path: Path to service account JSON file.
                    If empty, uses OAuth2 with credentials.json.

            Returns:
                Authentication result or error message.
            """
            service = _load_service(service_account_json_path)
            if service is None and not service_account_json_path:  # pragma: no branch
                service = _run_oauth_flow()
            if service is None:  # pragma: no branch
                return (
                    f"Authentication failed. Ensure credentials exist at "
                    f"{_service_account_path()} or {_credentials_path()}"
                )
            agent._backend._service = service
            try:
                resp = agent._backend._service.spaces().list(pageSize=1).execute()
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Google Chat authentication successful.",
                        "space_count": len(resp.get("spaces", [])),
                    }
                )
            except Exception as e:
                return json.dumps({"ok": True, "message": "Authenticated.", "error": str(e)})

        def clear_googlechat_auth() -> str:
            """Clear the stored Google Chat credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._service = None
            return "Google Chat authentication cleared."

        return [check_googlechat_auth, authenticate_googlechat, clear_googlechat_auth]


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
