"""IRC Agent — StatefulSorcarAgent extension with IRC tools.

Connects to IRC servers via the irc library. Stores config in
``~/.kiss/channels/irc/config.json``.

Usage::

    agent = IRCAgent()
    agent.run(prompt_template="Join #general and say hello")
"""

from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import (
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import wait_for_matching_message

_IRC_DIR = Path.home() / ".kiss" / "channels" / "irc"


def _config_path() -> Path:
    """Return the path to the stored IRC config file."""
    return _IRC_DIR / "config.json"


def _load_config() -> dict[str, Any] | None:
    """Load stored IRC config from disk."""
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("server") and data.get("nick"):  # pragma: no branch
            return {
                "server": data["server"],
                "port": int(data.get("port", 6667)),
                "nick": data["nick"],
                "password": data.get("password", ""),
                "use_tls": bool(data.get("use_tls", False)),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(
    server: str,
    nick: str,
    port: int = 6667,
    password: str = "",
    use_tls: bool = False,
) -> None:
    """Save IRC config to disk with restricted permissions."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "server": server.strip(),
        "nick": nick.strip(),
        "port": port,
        "password": password,
        "use_tls": use_tls,
    }, indent=2))
    if sys.platform != "win32":  # pragma: no branch
        path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored IRC config."""
    path = _config_path()
    if path.exists():  # pragma: no branch
        path.unlink()


class IRCChannelBackend:
    """ChannelBackend implementation for IRC via raw socket."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._nick: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._connection_info: str = ""
        self._reader_thread: threading.Thread | None = None

    def connect(self) -> bool:
        """Connect to IRC server."""
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            self._connection_info = "No IRC config found."
            return False
        try:
            self._nick = cfg["nick"]
            sock = socket.create_connection((cfg["server"], cfg["port"]), timeout=30)
            if cfg.get("use_tls"):  # pragma: no branch
                import ssl
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=cfg["server"])
            self.disconnect()
            self._sock = sock
            self._sock.settimeout(1.0)
            if cfg.get("password"):  # pragma: no branch
                self._send_raw(f"PASS {cfg['password']}")
            self._send_raw(f"NICK {cfg['nick']}")
            self._send_raw(f"USER {cfg['nick']} 0 * :{cfg['nick']}")
            self._connection_info = f"Connected to {cfg['server']} as {cfg['nick']}"
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._reader_thread.start()
            time.sleep(1.0)  # Wait for server greeting
            return True
        except Exception as e:
            self._connection_info = f"IRC connection failed: {e}"
            return False

    def _send_raw(self, line: str) -> None:
        """Send a raw IRC line."""
        if self._sock:  # pragma: no branch
            self._sock.sendall(f"{line}\r\n".encode("utf-8", errors="replace"))

    def _read_loop(self) -> None:
        """Background thread reading IRC data."""
        buf = ""
        while self._sock is not None:  # pragma: no branch
            try:
                data = self._sock.recv(4096)
                if not data:  # pragma: no branch
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\r\n" in buf:  # pragma: no branch
                    line, buf = buf.split("\r\n", 1)
                    self._handle_line(line)
            except TimeoutError:
                continue
            except OSError:
                break

    def _handle_line(self, line: str) -> None:
        """Handle a received IRC line."""
        if line.startswith("PING"):  # pragma: no branch
            self._send_raw(f"PONG {line[5:]}")
        elif "PRIVMSG" in line:  # pragma: no branch
            # :nick!user@host PRIVMSG #channel :message
            parts = line.split(" ", 3)
            if len(parts) >= 4:  # pragma: no branch
                prefix = parts[0].lstrip(":")
                nick = prefix.split("!")[0]
                target = parts[2]
                text = parts[3].lstrip(":")
                self._message_queue.put({
                    "ts": str(time.time()),
                    "user": nick,
                    "text": text,
                    "target": target,
                })

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return channel name."""
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return username."""
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """Join an IRC channel."""
        self._send_raw(f"JOIN {channel_id}")
        time.sleep(0.5)

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return buffered IRC messages."""
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:  # pragma: no branch
            msg = self._message_queue.get_nowait()
            if not channel_id or msg.get("target") == channel_id:  # pragma: no branch
                messages.append(msg)
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an IRC PRIVMSG."""
        self._send_raw(f"PRIVMSG {channel_id} :{text}")

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=1.0,
        )

    def disconnect(self) -> None:
        """Close the IRC socket and join the reader thread."""
        sock = self._sock
        self._sock = None
        if sock is not None:  # pragma: no branch
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._reader_thread is not None:  # pragma: no branch
            self._reader_thread.join(timeout=5.0)
            self._reader_thread = None

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot."""
        return bool(msg.get("user", "") == self._nick)

    def strip_bot_mention(self, text: str) -> str:
        """Remove bot mention from text."""
        prefix = f"{self._nick}: "
        if text.startswith(prefix):  # pragma: no branch
            return text[len(prefix):]
        return text

    def connect_irc(
        self,
        server: str,
        port: int = 6667,
        nick: str = "KISSBot",
        realname: str = "KISS Agent",
        password: str = "",
        use_tls: bool = False,
    ) -> str:
        """Connect to an IRC server.

        Args:
            server: IRC server hostname or IP.
            port: Server port. Default: 6667.
            nick: Nickname to use. Default: "KISSBot".
            realname: Real name. Default: "KISS Agent".
            password: Server password. Optional.
            use_tls: Use TLS encryption. Default: False.

        Returns:
            JSON string with ok status.
        """
        try:
            _save_config(server, nick, port, password, use_tls)
            success = self.connect()
            if success:  # pragma: no branch
                return json.dumps({"ok": True, "message": f"Connected to {server} as {nick}"})
            return json.dumps({"ok": False, "error": self._connection_info})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def join_irc_channel(self, channel: str) -> str:
        """Join an IRC channel.

        Args:
            channel: Channel name (e.g. "#general").

        Returns:
            JSON string with ok status.
        """
        try:
            self.join_channel(channel)
            return json.dumps({"ok": True, "channel": channel})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def leave_channel(self, channel: str, reason: str = "") -> str:
        """Leave an IRC channel.

        Args:
            channel: Channel name.
            reason: Optional leave reason.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"PART {channel}" + (f" :{reason}" if reason else ""))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_message(self, channel_or_nick: str, text: str) -> str:
        """Send a message to an IRC channel or user.

        Args:
            channel_or_nick: Target channel (e.g. "#general") or nick.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"PRIVMSG {channel_or_nick} :{text}")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_notice(self, channel_or_nick: str, text: str) -> str:
        """Send a NOTICE to an IRC channel or user.

        Args:
            channel_or_nick: Target channel or nick.
            text: Notice text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"NOTICE {channel_or_nick} :{text}")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_topic(self, channel: str) -> str:
        """Get the topic of an IRC channel.

        Args:
            channel: Channel name.

        Returns:
            JSON string with ok status (topic comes via server response).
        """
        try:
            self._send_raw(f"TOPIC {channel}")
            return json.dumps({"ok": True, "note": "Topic will appear in server messages"})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_topic(self, channel: str, topic: str) -> str:
        """Set the topic of an IRC channel.

        Args:
            channel: Channel name.
            topic: New topic text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"TOPIC {channel} :{topic}")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def kick_user(self, channel: str, nick: str, reason: str = "") -> str:
        """Kick a user from an IRC channel.

        Args:
            channel: Channel name.
            nick: Nickname to kick.
            reason: Optional kick reason.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"KICK {channel} {nick}" + (f" :{reason}" if reason else ""))
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def whois(self, nick: str) -> str:
        """Get WHOIS information about a user.

        Args:
            nick: Nickname to look up.

        Returns:
            JSON string with ok status (data comes via server response).
        """
        try:
            self._send_raw(f"WHOIS {nick}")
            return json.dumps({"ok": True, "note": "WHOIS info will appear in server messages"})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def identify_nickserv(self, password: str) -> str:
        """Identify to NickServ.

        Args:
            password: NickServ password.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_raw(f"PRIVMSG NickServ :IDENTIFY {password}")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

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


class IRCAgent(StatefulSorcarAgent):
    """StatefulSorcarAgent extended with IRC tools."""

    def __init__(self) -> None:
        super().__init__("IRC Agent")
        self._backend = IRCChannelBackend()
        cfg = _load_config()
        if cfg:  # pragma: no branch
            self._backend._nick = cfg.get("nick", "")

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + IRC auth tools + IRC API tools."""
        tools = super()._get_tools()
        agent = self

        def check_irc_auth() -> str:
            """Check if IRC is configured and connected.

            Returns:
                Connection status or instructions.
            """
            if not agent._backend._nick:  # pragma: no branch
                return (
                    "Not configured for IRC. Use authenticate_irc(server=..., nick=...) "
                    "to configure and connect."
                )
            return json.dumps({
                "ok": True,
                "nick": agent._backend._nick,
                "connected": agent._backend._sock is not None,
            })

        def authenticate_irc(
            server: str,
            nick: str,
            port: int = 6667,
            password: str = "",
            use_tls: bool = False,
        ) -> str:
            """Configure and connect to an IRC server.

            Args:
                server: IRC server hostname.
                nick: Nickname to use.
                port: Server port. Default: 6667.
                password: Server password. Optional.
                use_tls: Use TLS. Default: False.

            Returns:
                Connection result or error message.
            """
            for val, name in [(server, "server"), (nick, "nick")]:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            _save_config(server, nick, port, password, use_tls)
            agent._backend._nick = nick.strip()
            success = agent._backend.connect()
            if success:  # pragma: no branch
                return json.dumps({"ok": True, "message": f"Connected to {server} as {nick}"})
            return json.dumps({"ok": False, "error": agent._backend._connection_info})

        def clear_irc_auth() -> str:
            """Clear the stored IRC configuration.

            Returns:
                Status message.
            """
            _clear_config()
            agent._backend._nick = ""
            if agent._backend._sock:  # pragma: no branch
                try:
                    agent._backend._sock.close()
                except Exception:
                    pass
            agent._backend._sock = None
            return "IRC configuration cleared."

        tools.extend([check_irc_auth, authenticate_irc, clear_irc_auth])

        if agent._backend._nick:  # pragma: no branch
            tools.extend(agent._backend.get_tool_methods())

        return tools


def main() -> None:
    """Run the IRCAgent from the command line with chat persistence."""
    import sys
    import time as time_mod

    if len(sys.argv) <= 1:  # pragma: no branch
        print("Usage: kiss-irc [-m MODEL] [-t TASK] [-n] [--daemon]")
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument("-n", "--new", action="store_true", help="Start a new chat session")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--daemon-channel", default="", help="IRC channel to monitor")
    parser.add_argument("--allow-users", default="", help="Comma-separated nicks to allow")
    args = parser.parse_args()

    if args.daemon:  # pragma: no branch
        from kiss.channels.background_agent import ChannelDaemon

        backend = IRCChannelBackend()
        cfg = _load_config()
        if not cfg:  # pragma: no branch
            print("Not configured. Run: kiss-irc -t 'authenticate'")
            sys.exit(1)
        backend._nick = cfg["nick"]
        backend.connect()
        allow_users = [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name="IRC Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            poll_interval=1.0,
            allow_users=allow_users,
        )
        print("Starting IRC daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = IRCAgent()
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
