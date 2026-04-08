"""iMessage Agent — StatefulSorcarAgent extension with iMessage tools via AppleScript.

macOS only. Uses osascript to send/receive iMessages via the Messages app.
Stores config in ``~/.kiss/channels/imessage/config.json``.

Usage::

    agent = IMessageAgent()
    agent.run(prompt_template="Send 'Hello!' to +14155238886")
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_IMESSAGE_DIR = Path.home() / ".kiss" / "channels" / "imessage"
_config = ChannelConfig(_IMESSAGE_DIR, ())

_PLATFORM_ERROR = json.dumps(
    {
        "ok": False,
        "error": "iMessage tools require macOS with the Messages app.",
    }
)


def _run_osascript(script: str) -> tuple[str, str]:
    """Run an AppleScript and return (stdout, stderr)."""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
    return result.stdout.strip(), result.stderr.strip()


class IMessageChannelBackend(ToolMethodBackend):
    """Channel backend for iMessage via AppleScript."""

    def __init__(self) -> None:
        self._enabled: bool = False
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Check macOS and Messages.app availability."""
        if sys.platform != "darwin":  # pragma: no branch
            self._connection_info = "iMessage requires macOS."
            return False
        try:
            _run_osascript('tell application "Messages" to get name')
            self._enabled = True
            self._connection_info = "iMessage available via Messages.app"
            return True
        except Exception as e:
            self._connection_info = f"iMessage unavailable: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll iMessage via AppleScript (basic implementation)."""
        # AppleScript polling is limited; basic implementation returns empty
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an iMessage."""
        script = f'''tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{channel_id}" of targetService
    send "{text}" to targetBuddy
end tell'''
        _run_osascript(script)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Reply waiting is not supported for AppleScript-based iMessage."""
        return None

    def send_imessage(self, recipient: str, text: str, service: str = "iMessage") -> str:
        """Send an iMessage or SMS to a recipient.

        Args:
            recipient: Phone number or Apple ID email to send to.
            text: Message text.
            service: "iMessage" or "SMS". Default: "iMessage".

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            script = f'''tell application "Messages"
    set targetService to 1st service whose service type = {service}
    set targetBuddy to buddy "{recipient}" of targetService
    send "{text}" to targetBuddy
end tell'''
            _, stderr = _run_osascript(script)
            if stderr:  # pragma: no branch
                return json.dumps({"ok": False, "error": stderr})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_attachment(self, recipient: str, file_path: str, service: str = "iMessage") -> str:
        """Send a file attachment via iMessage.

        Args:
            recipient: Phone number or Apple ID email.
            file_path: Absolute path to the file to send.
            service: "iMessage" or "SMS". Default: "iMessage".

        Returns:
            JSON string with ok status.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            script = f'''tell application "Messages"
    set targetService to 1st service whose service type = {service}
    set targetBuddy to buddy "{recipient}" of targetService
    send POSIX file "{file_path}" to targetBuddy
end tell'''
            _, stderr = _run_osascript(script)
            if stderr:  # pragma: no branch
                return json.dumps({"ok": False, "error": stderr})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_conversations(self) -> str:
        """List recent iMessage conversations.

        Returns:
            JSON string with conversation list.
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        try:
            script = """tell application "Messages"
    set convos to {}
    repeat with c in every chat
        set end of convos to (id of c as string) & "|" & (display name of c as string)
    end repeat
    return convos
end tell"""
            stdout, stderr = _run_osascript(script)
            if stderr:  # pragma: no branch
                return json.dumps({"ok": False, "error": stderr})
            convos = []
            for item in stdout.split(", "):  # pragma: no branch
                parts = item.split("|", 1)
                convos.append(
                    {
                        "id": parts[0].strip(),
                        "name": parts[1].strip() if len(parts) > 1 else "",
                    }
                )
            return json.dumps({"ok": True, "conversations": convos}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_messages(self, recipient: str, limit: int = 20) -> str:
        """Get recent messages with a recipient (basic implementation).

        Args:
            recipient: Phone number or email to get messages for.
            limit: Maximum messages to return. Default: 20.

        Returns:
            JSON string with message list (basic).
        """
        if sys.platform != "darwin":  # pragma: no branch
            return _PLATFORM_ERROR
        return json.dumps(
            {
                "ok": True,
                "note": "Full message history requires direct database access. "
                "Use BlueBubbles for complete iMessage history access.",
                "messages": [],
            }
        )


class IMessageAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with iMessage tools (macOS only)."""

    def __init__(self) -> None:
        super().__init__("iMessage Agent")
        self._backend = IMessageChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._enabled = True

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._enabled and sys.platform == "darwin"

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_imessage_auth() -> str:
            """Check if iMessage is available on this system.

            Returns:
                Availability status or instructions.
            """
            if sys.platform != "darwin":  # pragma: no branch
                return _PLATFORM_ERROR
            if not agent._backend._enabled:  # pragma: no branch
                return (
                    "iMessage not configured. Use authenticate_imessage() to enable. "
                    "Requires macOS with Messages.app."
                )
            try:
                stdout, _ = _run_osascript('tell application "Messages" to get name')
                return json.dumps({"ok": True, "app": stdout})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_imessage() -> str:
            """Enable iMessage access on this macOS system.

            Returns:
                Result or error message.
            """
            if sys.platform != "darwin":  # pragma: no branch
                return _PLATFORM_ERROR
            try:
                stdout, _ = _run_osascript('tell application "Messages" to get name')
                _config.save({"enabled": "true"})
                agent._backend._enabled = True
                return json.dumps({"ok": True, "message": f"iMessage enabled via {stdout}."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_imessage_auth() -> str:
            """Clear the stored iMessage configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._enabled = False
            return "iMessage configuration cleared."

        return [check_imessage_auth, authenticate_imessage, clear_imessage_auth]


def main() -> None:
    """Run the IMessageAgent from the command line with chat persistence."""
    channel_main(IMessageAgent, "kiss-imessage")


if __name__ == "__main__":
    main()
