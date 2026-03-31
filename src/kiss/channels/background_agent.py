"""Background channel daemon — polls ChannelBackend and triggers agents."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels import ChannelBackend

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3.0
_RECONNECT_BASE = 2.0
_RECONNECT_MAX = 60.0
_RECONNECT_ATTEMPTS = 5
_STALE_THRESHOLD = 300.0


class ChannelDaemon:
    """Background daemon that monitors a ChannelBackend and triggers agents.

    Runs a polling loop that detects inbound messages and spawns a
    StatefulSorcarAgent to respond to each conversation.
    """

    def __init__(
        self,
        backend: ChannelBackend,
        channel_name: str,
        agent_name: str,
        extra_tools: list | None = None,
        model_name: str = "",
        max_budget: float = 5.0,
        work_dir: str = "",
        poll_interval: float = _POLL_INTERVAL,
        allow_users: list[str] | None = None,
    ) -> None:
        self._backend = backend
        self._channel_name = channel_name
        self._agent_name = agent_name
        self._extra_tools = extra_tools or []
        self._model_name = model_name
        self._max_budget = max_budget
        self._work_dir = work_dir or str(Path.home() / ".kiss" / "daemon_work")
        self._poll_interval = poll_interval
        self._allow_users = set(allow_users) if allow_users else None
        self._sender_locks: dict[str, threading.Lock] = {}
        self._sender_chat_ids: dict[str, str] = {}
        self._last_event_at: float = time.time()
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Start the daemon loop. Blocks until stop() is called or fatal error."""
        reconnect_delay = _RECONNECT_BASE
        attempts = 0
        while not self._stop_event.is_set():
            try:
                self._connect_and_poll()
                reconnect_delay = _RECONNECT_BASE
                attempts = 0
            except Exception as e:
                attempts += 1
                if _RECONNECT_ATTEMPTS > 0 and attempts >= _RECONNECT_ATTEMPTS:
                    logger.error("Max reconnect attempts reached: %s", e)
                    raise
                logger.warning(
                    "Channel error (attempt %d): %s. Retrying in %.1fs",
                    attempts,
                    e,
                    reconnect_delay,
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX)

    def stop(self) -> None:
        """Signal the daemon to stop after the current poll cycle."""
        self._stop_event.set()

    def _connect_and_poll(self) -> None:
        """Connect to channel and run the polling loop."""
        if not self._backend.connect():
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)

        channel_id = ""
        if self._channel_name:
            channel_id = self._backend.find_channel(self._channel_name) or ""
            if not channel_id:
                raise RuntimeError(f"Channel not found: {self._channel_name!r}")
            self._backend.join_channel(channel_id)
            logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

        oldest = str(time.time())
        self._last_event_at = time.time()

        while not self._stop_event.is_set():
            if time.time() - self._last_event_at > _STALE_THRESHOLD:
                logger.warning("No events for %.0fs — reconnecting", _STALE_THRESHOLD)
                raise RuntimeError("Stale connection detected")

            messages, oldest = self._backend.poll_messages(channel_id, oldest)
            if messages:
                self._last_event_at = time.time()

            for msg in messages:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if self._allow_users and user_id not in self._allow_users:
                    logger.debug("Ignoring message from non-allowed user: %s", user_id)
                    continue
                self._dispatch_message(channel_id, msg)

            time.sleep(self._poll_interval)

    def _dispatch_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Spawn a thread to handle one inbound message."""
        user_id = msg.get("user", "unknown")
        session_key = f"{channel_id}:{user_id}"

        if session_key not in self._sender_locks:
            self._sender_locks[session_key] = threading.Lock()
        lock = self._sender_locks[session_key]

        text = self._backend.strip_bot_mention(msg.get("text", ""))
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))

        def handle() -> None:
            if not lock.acquire(blocking=False):
                logger.debug("Skipping message — agent already busy for %s", session_key)
                return
            try:
                agent = StatefulSorcarAgent(self._agent_name)
                if session_key in self._sender_chat_ids:
                    agent._chat_id = self._sender_chat_ids[session_key]
                else:
                    agent.new_chat()
                    self._sender_chat_ids[session_key] = agent._chat_id

                tools = list(self._extra_tools)

                def reply(message: str) -> str:
                    """Send a reply to the current conversation.

                    Args:
                        message: Text to send as the bot's reply.

                    Returns:
                        JSON string with ok status.
                    """
                    try:
                        self._backend.send_message(channel_id, message, thread_ts)
                        return json.dumps({"ok": True})
                    except Exception as e:
                        return json.dumps({"ok": False, "error": str(e)})

                tools.append(reply)

                Path(self._work_dir).mkdir(parents=True, exist_ok=True)
                agent.run(
                    prompt_template=text,
                    model_name=self._model_name,
                    max_budget=self._max_budget,
                    work_dir=self._work_dir,
                    tools=tools,
                    headless=True,
                    verbose=False,
                )
            except Exception as e:
                logger.error("Agent error for %s: %s", session_key, e, exc_info=True)
                try:
                    self._backend.send_message(
                        channel_id,
                        f"Error processing your message: {e}",
                        thread_ts,
                    )
                except Exception:
                    pass
            finally:
                lock.release()

        thread = threading.Thread(target=handle, daemon=True)
        thread.start()
