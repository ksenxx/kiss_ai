"""Slack channel agent for reading and writing messages via Sorcar.

Manages Slack workspace connections and provides tools for the SorcarAgent
to read messages from and write messages to Slack channels.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from kiss.agents.sorcar.task_history import _KISS_DIR

logger = logging.getLogger(__name__)

_SLACK_CONFIG_DIR = _KISS_DIR / "slack"
_SLACK_WORKSPACES_FILE = _SLACK_CONFIG_DIR / "workspaces.json"


def _load_workspaces() -> dict[str, dict[str, str]]:
    """Load saved Slack workspace configurations.

    Returns:
        Dict mapping workspace name to {"token": "xoxb-...", "name": "workspace"}.
    """
    if not _SLACK_WORKSPACES_FILE.exists():
        return {}
    try:
        data = json.loads(_SLACK_WORKSPACES_FILE.read_text())
        return dict(data) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.debug("Failed to load workspaces", exc_info=True)
        return {}


def _save_workspaces(workspaces: dict[str, dict[str, str]]) -> None:
    """Save Slack workspace configurations to disk.

    Args:
        workspaces: Dict mapping workspace name to config dict.
    """
    _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _SLACK_WORKSPACES_FILE.write_text(json.dumps(workspaces, indent=2))


def _get_client(token: str) -> Any:
    """Create a Slack WebClient from a bot token.

    Args:
        token: Slack bot token (xoxb-...).

    Returns:
        slack_sdk.WebClient instance.
    """
    from slack_sdk import WebClient

    return WebClient(token=token)


def _validate_token(token: str) -> tuple[bool, str]:
    """Validate a Slack bot token by calling auth.test.

    Args:
        token: Slack bot token to validate.

    Returns:
        (is_valid, team_name_or_error_message).
    """
    try:
        client = _get_client(token)
        response = client.auth_test()
        if response["ok"]:
            return True, str(response.get("team", "Unknown"))
        return False, str(response.get("error", "Unknown error"))
    except Exception as e:
        logger.debug("Token validation failed", exc_info=True)
        return False, str(e)


def add_workspace(name: str, token: str) -> str:
    """Add or update a Slack workspace configuration.

    Validates the token and saves if valid.

    Args:
        name: A friendly name for this workspace.
        token: Slack bot token (xoxb-...).

    Returns:
        Status message indicating success or failure.
    """
    if not name.strip():
        return "Error: workspace name cannot be empty"
    if not token.strip():
        return "Error: token cannot be empty"

    valid, team_or_error = _validate_token(token.strip())
    if not valid:
        return f"Error: invalid token - {team_or_error}"

    workspaces = _load_workspaces()
    workspaces[name.strip()] = {"token": token.strip(), "team": team_or_error}
    _save_workspaces(workspaces)
    return f"Workspace '{name.strip()}' added (team: {team_or_error})"


def remove_workspace(name: str) -> str:
    """Remove a Slack workspace configuration.

    Args:
        name: The workspace name to remove.

    Returns:
        Status message indicating success or failure.
    """
    workspaces = _load_workspaces()
    if name not in workspaces:
        return f"Error: workspace '{name}' not found"
    del workspaces[name]
    _save_workspaces(workspaces)
    return f"Workspace '{name}' removed"


def list_workspaces() -> str:
    """List all configured Slack workspaces.

    Returns:
        Formatted string listing workspaces, or a message if none configured.
    """
    workspaces = _load_workspaces()
    if not workspaces:
        return (
            "No Slack workspaces configured.\n"
            "Use add_workspace(name, token) to add one.\n"
            "You need a Slack Bot Token (xoxb-...) with channels:read, "
            "channels:history, chat:write scopes."
        )
    lines = ["Configured Slack workspaces:"]
    for name, info in workspaces.items():
        lines.append(f"  - {name} (team: {info.get('team', 'unknown')})")
    return "\n".join(lines)


def _resolve_workspace_client(workspace: str | None) -> tuple[Any, str]:
    """Get the WebClient for a workspace name (or the only one if just one exists).

    Args:
        workspace: Workspace name, or None to auto-select if only one exists.

    Returns:
        (client, workspace_name) tuple.

    Raises:
        ValueError: If workspace not found or ambiguous.
    """
    workspaces = _load_workspaces()
    if not workspaces:
        raise ValueError(
            "No Slack workspaces configured. "
            "Use add_workspace(name, token) to add one first."
        )
    if workspace is None:
        if len(workspaces) == 1:
            name = next(iter(workspaces))
            return _get_client(workspaces[name]["token"]), name
        raise ValueError(
            f"Multiple workspaces configured: {list(workspaces.keys())}. "
            "Specify which one with the workspace parameter."
        )
    if workspace not in workspaces:
        raise ValueError(
            f"Workspace '{workspace}' not found. "
            f"Available: {list(workspaces.keys())}"
        )
    return _get_client(workspaces[workspace]["token"]), workspace


def list_channels(workspace: str | None = None) -> str:
    """List public channels in a Slack workspace.

    Args:
        workspace: Workspace name. Auto-selects if only one workspace configured.

    Returns:
        Formatted string listing channels.
    """
    try:
        client, ws_name = _resolve_workspace_client(workspace)
    except ValueError as e:
        return str(e)
    try:
        result = client.conversations_list(types="public_channel", limit=200)
        channels = result.get("channels", [])
        if not channels:
            return f"No public channels found in workspace '{ws_name}'"
        lines = [f"Channels in '{ws_name}':"]
        for ch in sorted(channels, key=lambda c: c.get("name", "")):
            name = ch.get("name", "unknown")
            purpose = ch.get("purpose", {}).get("value", "")
            member_count = ch.get("num_members", "?")
            line = f"  #{name} ({member_count} members)"
            if purpose:
                line += f" - {purpose[:80]}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Failed to list channels", exc_info=True)
        return f"Error listing channels: {e}"


def read_messages(
    channel: str,
    workspace: str | None = None,
    limit: int = 20,
) -> str:
    """Read recent messages from a Slack channel.

    Args:
        channel: Channel name (without #) or channel ID.
        workspace: Workspace name. Auto-selects if only one configured.
        limit: Maximum number of messages to retrieve (default 20).

    Returns:
        Formatted string with recent messages.
    """
    try:
        client, ws_name = _resolve_workspace_client(workspace)
    except ValueError as e:
        return str(e)
    try:
        channel_id = _resolve_channel_id(client, channel)
        result = client.conversations_history(channel=channel_id, limit=limit)
        messages = result.get("messages", [])
        if not messages:
            return f"No messages in #{channel} ({ws_name})"
        lines = [f"Recent messages in #{channel} ({ws_name}):"]
        # Messages come newest-first; reverse for chronological order
        for msg in reversed(messages):
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            # Try to get display name
            try:
                user_info = client.users_info(user=user)
                display = user_info["user"]["profile"].get(
                    "display_name"
                ) or user_info["user"].get("real_name", user)
            except Exception:
                display = user
            time_str = ""
            if ts:
                try:
                    time_str = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(float(ts))
                    )
                except (ValueError, OSError):
                    pass
            prefix = f"[{time_str}] {display}" if time_str else display
            lines.append(f"  {prefix}: {text}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Failed to read messages", exc_info=True)
        return f"Error reading messages from #{channel}: {e}"


def send_message(
    channel: str,
    text: str,
    workspace: str | None = None,
) -> str:
    """Send a message to a Slack channel.

    Args:
        channel: Channel name (without #) or channel ID.
        text: Message text to send.
        workspace: Workspace name. Auto-selects if only one configured.

    Returns:
        Status message indicating success or failure.
    """
    if not text.strip():
        return "Error: message text cannot be empty"
    try:
        client, ws_name = _resolve_workspace_client(workspace)
    except ValueError as e:
        return str(e)
    try:
        channel_id = _resolve_channel_id(client, channel)
        result = client.chat_postMessage(channel=channel_id, text=text)
        if result.get("ok"):
            return f"Message sent to #{channel} ({ws_name})"
        return f"Error: {result.get('error', 'unknown error')}"
    except Exception as e:
        logger.debug("Failed to send message", exc_info=True)
        return f"Error sending message to #{channel}: {e}"


def _resolve_channel_id(client: Any, channel: str) -> str:
    """Resolve a channel name to its ID.

    Args:
        client: Slack WebClient.
        channel: Channel name (with or without #) or already an ID.

    Returns:
        Channel ID string.

    Raises:
        ValueError: If channel not found.
    """
    # If it looks like an ID already (starts with C or G), use directly
    ch = channel.lstrip("#")
    if ch and ch[0] in ("C", "G") and ch[1:].isalnum():
        return ch

    # Search through channels
    cursor = None
    while True:
        kwargs: dict[str, Any] = {"types": "public_channel,private_channel", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        result = client.conversations_list(**kwargs)
        for c in result.get("channels", []):
            if c.get("name") == ch:
                return str(c["id"])
        cursor = result.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    raise ValueError(f"Channel '{channel}' not found")


class SlackChannelAgent:
    """Agent that bridges Slack channels with the Sorcar chat window.

    Provides tools for SorcarAgent to manage Slack workspaces, read messages,
    and send messages. Also supports background polling of channels to display
    new messages in the Sorcar chat window.
    """

    def __init__(self) -> None:
        self._poll_threads: dict[str, threading.Event] = {}

    def get_tools(self) -> list:
        """Return the list of tools for use by SorcarAgent.

        Returns:
            List of callable tool functions.
        """
        return [
            list_workspaces,
            add_workspace,
            remove_workspace,
            list_channels,
            read_messages,
            send_message,
        ]

    def start_polling(
        self,
        channel: str,
        workspace: str | None = None,
        interval: float = 10.0,
        callback: Any = None,
    ) -> str:
        """Start polling a Slack channel for new messages in the background.

        Args:
            channel: Channel name to poll.
            workspace: Workspace name.
            interval: Polling interval in seconds.
            callback: Called with (channel, messages_text) when new messages arrive.

        Returns:
            Status message.
        """
        key = f"{workspace or 'default'}:{channel}"
        if key in self._poll_threads:
            return f"Already polling #{channel}"

        stop_event = threading.Event()
        self._poll_threads[key] = stop_event

        def _poll() -> None:
            last_ts = str(time.time())
            while not stop_event.is_set():
                stop_event.wait(interval)
                if stop_event.is_set():
                    break
                try:
                    client, _ = _resolve_workspace_client(workspace)
                    channel_id = _resolve_channel_id(client, channel)
                    result = client.conversations_history(
                        channel=channel_id, oldest=last_ts, limit=50
                    )
                    messages = result.get("messages", [])
                    if messages:
                        last_ts = messages[0].get("ts", last_ts)
                        if callback:
                            text = _format_messages(client, channel, messages)
                            callback(channel, text)
                except Exception:
                    logger.debug("Poll error", exc_info=True)

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        return f"Started polling #{channel} every {interval}s"

    def stop_polling(self, channel: str, workspace: str | None = None) -> str:
        """Stop polling a Slack channel.

        Args:
            channel: Channel name to stop polling.
            workspace: Workspace name.

        Returns:
            Status message.
        """
        key = f"{workspace or 'default'}:{channel}"
        if key not in self._poll_threads:
            return f"Not polling #{channel}"
        self._poll_threads[key].set()
        del self._poll_threads[key]
        return f"Stopped polling #{channel}"

    def stop_all_polling(self) -> None:
        """Stop all active polling threads."""
        for ev in self._poll_threads.values():
            ev.set()
        self._poll_threads.clear()


def _format_messages(client: Any, channel: str, messages: list[dict]) -> str:
    """Format Slack messages for display.

    Args:
        client: Slack WebClient for user lookups.
        channel: Channel name.
        messages: List of message dicts from Slack API.

    Returns:
        Formatted messages string.
    """
    lines = []
    for msg in reversed(messages):
        user = msg.get("user", "unknown")
        text = msg.get("text", "")
        try:
            user_info = client.users_info(user=user)
            display = user_info["user"]["profile"].get(
                "display_name"
            ) or user_info["user"].get("real_name", user)
        except Exception:
            display = user
        lines.append(f"#{channel} [{display}]: {text}")
    return "\n".join(lines)
