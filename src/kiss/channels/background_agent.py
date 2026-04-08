"""Background channel daemon — polls a ChannelBackend and runs agents.

Pseudocode
----------

run():  (blocking main loop)
    Connect to backend with exponential-backoff retry.
    Join the named channel.
    Poll for new messages in a loop.
    Reconnect on errors with exponential backoff (reset on success).

run_once():  (one-shot poll mode)
    Connect to backend.
    Join the named channel.
    Poll recent messages.
    Skip bot messages and non-allowed users.
    Skip messages the bot has already replied to.
    Process each pending message with a StatefulSorcarAgent.
    Return the number of messages processed.

On each inbound message:
    Skip bot's own messages and non-allowed users.
    Route to a per-thread session (keyed by channel + thread root ts).
    Each top-level message starts a new chat session.
    Thread replies reuse the chat session of the top-level message.
    Each session has a FIFO queue and a mutex ensuring only one
    worker thread processes that session's messages at a time.

Worker thread (one per active session):
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
        self._stop_event = threading.Event()
        # Thread reply polling: thread_ts → oldest timestamp for polling.
        # Only used when the backend supports poll_thread_messages().
        self._active_threads: dict[str, str] = {}
        self._active_threads_lock = threading.Lock()
        self._poll_thread_fn = getattr(backend, "poll_thread_messages", None)

    def run(self) -> None:
        """Start the daemon loop. Blocks until stop() is called or fatal error."""
        Base.reset_global_budget()
        self._reconnect_attempts = 0
        self._reconnect_delay = _RECONNECT_BASE
        while not self._stop_event.is_set():
            try:
                self._connect_and_poll()
            except Exception as e:
                self._reconnect_attempts += 1
                if (  # pragma: no branch
                    _RECONNECT_ATTEMPTS > 0 and self._reconnect_attempts >= _RECONNECT_ATTEMPTS
                ):
                    logger.error("Max reconnect attempts reached: %s", e)
                    raise
                logger.warning(
                    "Channel error (attempt %d): %s. Retrying in %.1fs",
                    self._reconnect_attempts,
                    e,
                    self._reconnect_delay,
                )
                if self._stop_event.wait(self._reconnect_delay):  # pragma: no branch
                    break
                self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX)

    def run_once(self) -> int:
        """One-shot poll: check for pending messages, process them, and exit.

        Connects to the backend, joins the configured channel, retrieves
        recent messages, filters to allowed users, skips messages the bot
        has already replied to, and runs a StatefulSorcarAgent for each
        pending message.  Each message is processed synchronously.

        Returns:
            Number of messages processed.

        Raises:
            RuntimeError: If connection or channel lookup fails.
        """
        Base.reset_global_budget()
        if not self._backend.connect():
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)
        try:
            channel_id = ""
            if self._channel_name:
                channel_id = self._backend.find_channel(self._channel_name) or ""
                if not channel_id:
                    raise RuntimeError(f"Channel not found: {self._channel_name!r}")
                self._backend.join_channel(channel_id)
                logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

            messages, _ = self._backend.poll_messages(channel_id, "0", limit=50)

            processed = 0
            for msg in messages:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if self._allow_users and user_id not in self._allow_users:
                    continue
                if self._has_bot_reply(channel_id, msg):
                    continue
                state = _SenderState()
                self._handle_message(
                    f"{channel_id}:{msg.get('ts', '')}",
                    channel_id,
                    msg,
                    state,
                )
                processed += 1

            return processed
        finally:
            self._disconnect_backend()

    def _has_bot_reply(self, channel_id: str, msg: dict[str, Any]) -> bool:
        """Check if the bot has already replied to a message's thread.

        Uses ``poll_thread_messages`` if the backend supports it.
        Returns ``False`` when thread polling is unavailable or the
        message has no replies.

        Args:
            channel_id: Channel ID containing the message.
            msg: Message dict from poll_messages.

        Returns:
            True if the bot has already replied in the thread.
        """
        if self._poll_thread_fn is None:
            return False
        if msg.get("reply_count", 0) == 0:
            return False
        msg_ts = msg.get("ts", "")
        if not msg_ts:
            return False
        try:
            replies, _ = self._poll_thread_fn(channel_id, msg_ts, "0", limit=100)
            return any(self._backend.is_from_bot(r) for r in replies)
        except Exception:
            logger.debug("Error checking thread replies for %s", msg_ts, exc_info=True)
            return False

    def stop(self) -> None:
        """Signal the daemon to stop and wait briefly for handler cleanup."""
        self._stop_event.set()
        self._disconnect_backend()
        self._join_handler_threads()

    def _connect_and_poll(self) -> None:
        """Connect to channel and run the polling loop.

        Resets the retry counter on successful connect so that transient
        poll errors don't accumulate across otherwise-healthy sessions.
        """
        if not self._backend.connect():  # pragma: no branch
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)
        # Reset retry state on successful connection.
        self._reconnect_attempts = 0
        self._reconnect_delay = _RECONNECT_BASE

        try:
            channel_id = ""
            if self._channel_name:  # pragma: no branch
                channel_id = self._backend.find_channel(self._channel_name) or ""
                if not channel_id:  # pragma: no branch
                    raise RuntimeError(f"Channel not found: {self._channel_name!r}")
                self._backend.join_channel(channel_id)
                logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

            oldest = str(time.time())

            while not self._stop_event.is_set():  # pragma: no branch
                messages, oldest = self._backend.poll_messages(channel_id, oldest)

                for msg in messages:  # pragma: no branch
                    if self._backend.is_from_bot(msg):  # pragma: no branch
                        continue
                    user_id = msg.get("user", "")
                    if self._allow_users and user_id not in self._allow_users:  # pragma: no branch
                        logger.debug("Ignoring message from non-allowed user: %s", user_id)
                        continue
                    self._dispatch_message(channel_id, msg)

                self._poll_active_threads(channel_id)

                if self._stop_event.wait(self._poll_interval):  # pragma: no branch
                    break
        finally:
            self._disconnect_backend()

    def _dispatch_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Queue an inbound message and ensure a handler thread is running.

        Sessions are keyed by thread root timestamp so that each top-level
        message gets a fresh chat session while thread replies reuse the
        session of their parent message.
        """
        thread_root = msg.get("thread_ts") or msg.get("ts", "")
        session_key = f"{channel_id}:{thread_root}"
        state = self._get_sender_state(session_key)
        state.pending_messages.put(msg)
        self._start_sender_worker(session_key, channel_id, state)
        # Register top-level messages for thread reply polling.
        is_top_level = "thread_ts" not in msg or msg.get("thread_ts") == msg.get("ts")
        msg_ts = msg.get("ts", "")
        if is_top_level and msg_ts and self._poll_thread_fn is not None:
            with self._active_threads_lock:
                if msg_ts not in self._active_threads:
                    self._active_threads[msg_ts] = f"{float(msg_ts) + 0.000001:.6f}"

    def _poll_active_threads(self, channel_id: str) -> None:
        """Poll tracked threads for new replies and dispatch them.

        Only runs when the backend provides a ``poll_thread_messages``
        method.  Skips bot messages and non-allowed users just like the
        top-level message loop.
        """
        if self._poll_thread_fn is None:
            return
        with self._active_threads_lock:
            threads = list(self._active_threads.items())
        for thread_ts, thread_oldest in threads:
            try:
                msgs, new_oldest = self._poll_thread_fn(channel_id, thread_ts, thread_oldest)
            except Exception:
                logger.debug("Error polling thread %s", thread_ts, exc_info=True)
                continue
            with self._active_threads_lock:
                self._active_threads[thread_ts] = new_oldest
            for msg in msgs:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if self._allow_users and user_id not in self._allow_users:
                    continue
                self._dispatch_message(channel_id, msg)

    def _get_sender_state(self, session_key: str) -> _SenderState:
        """Return the per-sender state object for *session_key*."""
        with self._sender_states_lock:
            state = self._sender_states.get(session_key)
            if state is None:
                state = _SenderState()
                self._sender_states[session_key] = state
            return state

    def _start_sender_worker(self, session_key: str, channel_id: str, state: _SenderState) -> None:
        """Start a managed worker for queued messages from one sender."""
        if not state.lock.acquire(blocking=False):
            return

        def handle() -> None:
            try:
                self._process_sender_queue(session_key, channel_id, state)
            finally:
                with self._handler_threads_lock:
                    self._handler_threads.discard(thread)

        thread = threading.Thread(target=handle, daemon=True)
        with self._handler_threads_lock:
            self._handler_threads.add(thread)
        thread.start()

    def _process_sender_queue(self, session_key: str, channel_id: str, state: _SenderState) -> None:
        """Drain queued messages for one sender sequentially.

        Re-checks the queue under the lock before releasing to prevent
        message orphaning when a producer enqueues while the worker is
        about to exit.
        """
        try:
            while not self._stop_event.is_set():  # pragma: no branch
                try:
                    msg = state.pending_messages.get_nowait()
                except queue.Empty:
                    return
                self._handle_message(session_key, channel_id, msg, state)
        finally:
            # Re-check under the lock: if new messages arrived between the
            # get_nowait() Empty and here, release and re-start a worker.
            has_more = not state.pending_messages.empty()
            state.lock.release()
            if has_more and not self._stop_event.is_set():
                self._start_sender_worker(session_key, channel_id, state)

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
        replied = threading.Event()

        def reply(message: str) -> str:
            """Send a reply to the current conversation.

            Args:
                message: Text to send as the bot's reply.

            Returns:
                JSON string with ok status.
            """
            replied.set()
            try:
                self._backend.send_message(channel_id, message, thread_ts)
                return json.dumps({"ok": True})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        tools.append(reply)

        Path(self._work_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = agent.run(
                prompt_template=text,
                model_name=self._model_name,
                max_budget=self._max_budget,
                work_dir=self._work_dir,
                tools=tools,
                verbose=False,
            )
            state.chat_id = agent.chat_id
            if not replied.is_set():  # pragma: no branch
                result_yaml = yaml.safe_load(result)
                summary = (result_yaml.get("summary", "") if result_yaml else "") or result
                self._send_reply(channel_id, summary, thread_ts)
        except Exception as e:
            logger.error("Agent error for %s: %s", session_key, e, exc_info=True)
            self._send_reply(channel_id, f"Error processing your message: {e}", thread_ts)

    def _send_reply(self, channel_id: str, text: str, thread_ts: str) -> None:
        """Send a reply message, retrying once on transient failure.

        Args:
            channel_id: Channel to post to.
            text: Message text.
            thread_ts: Thread timestamp for threading.
        """
        for attempt in range(2):
            try:
                self._backend.send_message(channel_id, text, thread_ts)
                return
            except Exception:
                if attempt == 0:
                    logger.warning("Reply failed, retrying...", exc_info=True)
                    time.sleep(1)
                else:
                    logger.error("Reply failed after retry", exc_info=True)

    def _disconnect_backend(self) -> None:
        """Best-effort backend cleanup hook for stop and reconnect paths."""
        try:
            self._backend.disconnect()
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
