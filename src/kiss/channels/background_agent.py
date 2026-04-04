"""Background channel daemon — polls a ChannelBackend and runs agents.

Pseudocode
----------

run():  (blocking main loop)
    Connect to backend with exponential-backoff retry.
    Join the named channel.
    Poll for new messages in a loop.
    Reconnect on staleness (no events for a while) or errors.

On each inbound message:
    Skip bot's own messages and non-allowed users.
    Route to a per-sender session (keyed by channel+user).
    Each session has a FIFO queue and a mutex ensuring only one
    worker thread processes that sender's messages at a time.
    Messages within a session share a persistent agent chat_id
    so the conversation is multi-turn.

Worker thread (one per active sender):
    Drain the session queue sequentially.
    For each message, create/resume a StatefulSorcarAgent and run it,
    passing a reply() tool the agent can call to post back to the channel.

stop():
    Signal all loops to exit, disconnect the backend, join worker threads.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels import ChannelBackend
from kiss.core.base import Base

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3.0
_RECONNECT_BASE = 2.0
_RECONNECT_MAX = 60.0
_RECONNECT_ATTEMPTS = 5
_STALE_THRESHOLD = 300.0
_HANDLER_JOIN_TIMEOUT = 5.0
_REPLY_WAIT_TIMEOUT = 300.0


@dataclass
class _SenderState:
    """Mutable per-sender state protected by the daemon's sender-state lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    chat_id: str = ""
    pending_messages: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)


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
        self._sender_states: dict[str, _SenderState] = {}
        self._sender_states_lock = threading.Lock()
        self._handler_threads: set[threading.Thread] = set()
        self._handler_threads_lock = threading.Lock()
        self._last_event_at: float = time.time()
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Start the daemon loop. Blocks until stop() is called or fatal error."""
        Base.reset_global_budget()
        reconnect_delay = _RECONNECT_BASE
        attempts = 0
        while not self._stop_event.is_set():
            try:
                self._connect_and_poll()
                reconnect_delay = _RECONNECT_BASE
                attempts = 0
            except Exception as e:
                attempts += 1
                if _RECONNECT_ATTEMPTS > 0 and attempts >= _RECONNECT_ATTEMPTS:  # pragma: no branch
                    logger.error("Max reconnect attempts reached: %s", e)
                    raise
                logger.warning(
                    "Channel error (attempt %d): %s. Retrying in %.1fs",
                    attempts,
                    e,
                    reconnect_delay,
                )
                if self._stop_event.wait(reconnect_delay):  # pragma: no branch
                    break
                reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX)

    def stop(self) -> None:
        """Signal the daemon to stop and wait briefly for handler cleanup."""
        self._stop_event.set()
        self._disconnect_backend()
        self._join_handler_threads()

    def _connect_and_poll(self) -> None:
        """Connect to channel and run the polling loop."""
        if not self._backend.connect():  # pragma: no branch
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)

        try:
            channel_id = ""
            if self._channel_name:  # pragma: no branch
                channel_id = self._backend.find_channel(self._channel_name) or ""
                if not channel_id:  # pragma: no branch
                    raise RuntimeError(f"Channel not found: {self._channel_name!r}")
                self._backend.join_channel(channel_id)
                logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

            oldest = str(time.time())
            self._last_event_at = time.time()

            while not self._stop_event.is_set():  # pragma: no branch
                if time.time() - self._last_event_at > _STALE_THRESHOLD:  # pragma: no branch
                    logger.warning("No events for %.0fs — reconnecting", _STALE_THRESHOLD)
                    raise RuntimeError("Stale connection detected")

                messages, oldest = self._backend.poll_messages(channel_id, oldest)
                if messages:  # pragma: no branch
                    self._last_event_at = time.time()

                for msg in messages:  # pragma: no branch
                    if self._backend.is_from_bot(msg):  # pragma: no branch
                        continue
                    user_id = msg.get("user", "")
                    if self._allow_users and user_id not in self._allow_users:  # pragma: no branch
                        logger.debug("Ignoring message from non-allowed user: %s", user_id)
                        continue
                    self._dispatch_message(channel_id, msg)

                if self._stop_event.wait(self._poll_interval):  # pragma: no branch
                    break
        finally:
            self._disconnect_backend()

    def _dispatch_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Queue an inbound message and ensure a handler thread is running."""
        user_id = msg.get("user", "unknown")
        session_key = f"{channel_id}:{user_id}"
        state = self._get_sender_state(session_key)
        state.pending_messages.put(msg)
        self._start_sender_worker(session_key, channel_id, state)

    def _get_sender_state(self, session_key: str) -> _SenderState:
        """Return the per-sender state object for *session_key*."""
        with self._sender_states_lock:
            state = self._sender_states.get(session_key)
            if state is None:
                state = _SenderState()
                self._sender_states[session_key] = state
            return state

    def _start_sender_worker(
        self, session_key: str, channel_id: str, state: _SenderState
    ) -> None:
        """Start a managed worker for queued messages from one sender."""
        if not state.lock.acquire(blocking=False):
            return

        def handle() -> None:
            try:
                self._process_sender_queue(session_key, channel_id, state)
            finally:
                state.lock.release()
                with self._handler_threads_lock:
                    self._handler_threads.discard(thread)

        thread = threading.Thread(target=handle, daemon=True)
        with self._handler_threads_lock:
            self._handler_threads.add(thread)
        thread.start()

    def _process_sender_queue(
        self, session_key: str, channel_id: str, state: _SenderState
    ) -> None:
        """Drain queued messages for one sender sequentially."""
        while not self._stop_event.is_set():  # pragma: no branch
            try:
                msg = state.pending_messages.get_nowait()
            except queue.Empty:
                return
            self._handle_message(session_key, channel_id, msg, state)

    def _handle_message(
        self,
        session_key: str,
        channel_id: str,
        msg: dict[str, Any],
        state: _SenderState,
    ) -> None:
        """Run one agent task for a queued inbound message."""
        text = self._backend.strip_bot_mention(msg.get("text", ""))
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))
        agent = StatefulSorcarAgent(self._agent_name)
        if state.chat_id:  # pragma: no branch
            agent.resume_chat_by_id(state.chat_id)
        else:
            agent.new_chat()
            state.chat_id = agent.chat_id

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

        replied = threading.Event()
        original_reply = reply

        def reply_with_tracking(message: str) -> str:
            """Send a reply to the current conversation.

            Args:
                message: Text to send as the bot's reply.

            Returns:
                JSON string with ok status.
            """
            replied.set()
            return original_reply(message)

        tools[-1] = reply_with_tracking

        Path(self._work_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = agent.run(
                prompt_template=text,
                model_name=self._model_name,
                max_budget=self._max_budget,
                work_dir=self._work_dir,
                tools=tools,
                headless=True,
                verbose=False,
            )
            state.chat_id = agent.chat_id
            if not replied.is_set():  # pragma: no branch
                result_yaml = yaml.safe_load(result)
                summary = (result_yaml.get("summary", "") if result_yaml else "") or result
                self._backend.send_message(channel_id, summary, thread_ts)
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

    def _disconnect_backend(self) -> None:
        """Best-effort backend cleanup hook for stop and reconnect paths."""
        disconnect = getattr(self._backend, "disconnect", None)
        if not callable(disconnect):  # pragma: no branch
            return
        try:
            disconnect()
        except Exception:
            logger.warning("Backend disconnect failed", exc_info=True)

    def _join_handler_threads(self) -> None:
        """Join in-flight handler threads with a bounded timeout."""
        with self._handler_threads_lock:
            threads = list(self._handler_threads)
        deadline = time.monotonic() + _HANDLER_JOIN_TIMEOUT
        for thread in threads:  # pragma: no branch
            remaining = deadline - time.monotonic()
            if remaining <= 0:  # pragma: no branch
                break
            thread.join(timeout=remaining)
