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

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

_IMESSAGE_DIR = Path.home() / ".kiss" / "channels" / "imessage"

_PLATFORM_ERROR = json.dumps({
    "ok": False,
    "error": "iMessage tools require macOS with the Messages app.",
})


def _config_path() -> Path:
    """Return the path to the stored iMessage config file."""
    return _IMESSAGE_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored iMessage config (minimal, no credentials needed)."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config() -> None:
    """Save iMessage config marker to disk."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": True}, indent=2))


def _clear_config() -> None:
    """Delete the stored iMessage config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


def _run_osascript(script: str) -> tuple[str, str]:
    """Run an AppleScript and return (stdout, stderr)."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip(), result.stderr.strip()


class IMessageChannelBackend:
    """ChannelBackend implementation for iMessage via AppleScript."""

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

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return phone number or email as channel ID."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username as user ID."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op for iMessage."""

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

    def disconnect(self) -> None:
        """Release backend resources before stop or reconnect."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mentions from text."""
        return text

    def send_imessage(
        self, recipient: str, text: str, service: str = "iMessage"
    ) -> str:
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

    def send_attachment(
        self, recipient: str, file_path: str, service: str = "iMessage"
    ) -> str:
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
            script = '''tell application "Messages"
    set convos to {}
    repeat with c in every chat
        set end of convos to (id of c as string) & "|" & (display name of c as string)
    end repeat
    return convos
end tell'''
            stdout, stderr = _run_osascript(script)
            if stderr:  # pragma: no branch
                return json.dumps({"ok": False, "error": stderr})
            convos = []
            for item in stdout.split(", "):  # pragma: no branch
                parts = item.split("|", 1)
                convos.append({
                    "id": parts[0].strip(),
                    "name": parts[1].strip() if len(parts) > 1 else "",
                })
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
        return json.dumps({
            "ok": True,
            "note": "Full message history requires direct database access. "
                    "Use BlueBubbles for complete iMessage history access.",
            "messages": [],
        })

    def get_tool_methods(self) -> list:
        """Return list of bound tool methods for use by the LLM agent."""
        non_tool = frozenset({
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention", "disconnect", "get_tool_methods",
        })
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in non_tool
            and callable(getattr(self, name))
        ]


class IMessageAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with iMessage tools (macOS only)."""

    def __init__(self) -> None:
        super().__init__("iMessage Agent")
        self._backend = IMessageChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._enabled = True

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + iMessage auth tools + API tools."""
        tools = super()._get_tools()
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
                _save_config()
                agent._backend._enabled = True
                return json.dumps({"ok": True, "message": f"iMessage enabled via {stdout}."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_imessage_auth() -> str:
            """Clear the stored iMessage configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._enabled = False
            return "iMessage configuration cleared."

        tools.extend([check_imessage_auth, authenticate_imessage, clear_imessage_auth])

        if agent._backend._enabled and sys.platform == "darwin":  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the IMessageAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-imessage [-m MODEL] [-t TASK] [-n]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    args = parser.parse_args()

    agent = IMessageAgent()
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.new:  # pragma: no branch
        agent.new_chat()
    else:
        agent.resume_chat(task_description)

    model_config: dict[str, Any] = {}
    if args.endpoint:  # pragma: no branch
        model_config["base_url"] = args.endpoint

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }

    start_time = time_mod.time()
    agent.run(**run_kwargs)
    elapsed = time_mod.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
