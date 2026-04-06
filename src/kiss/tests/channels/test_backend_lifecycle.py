from __future__ import annotations

import inspect
import queue
import threading
import time
from typing import Any, cast

from kiss.channels.background_agent import ChannelDaemon, _SenderState
from kiss.channels.irc_agent import IRCChannelBackend
from kiss.channels.line_agent import LineChannelBackend
from kiss.channels.slack_agent import SlackChannelBackend
from kiss.channels.synology_chat_agent import SynologyChatChannelBackend
from kiss.channels.whatsapp_agent import WhatsAppChannelBackend
from kiss.channels.zalo_agent import ZaloChannelBackend


class _FakeSlackClient:
    def __init__(self) -> None:
        self.calls = 0

    def conversations_replies(self, *, channel: str, ts: str, limit: int) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"messages": [{"ts": "1", "user": "other", "text": "old"}]}
        if self.calls == 2:
            return {"messages": [{"ts": "1", "user": "other", "text": "old"}]}
        return {"messages": [{"ts": "2", "user": "u1", "text": "reply"}]}


class _FakeSocket:
    def __init__(self) -> None:
        self.timeout: float | None = None
        self.shutdown_called = False
        self.closed = False

    def settimeout(self, value: float | None) -> None:
        self.timeout = value

    def recv(self, size: int) -> bytes:
        raise OSError("closed")

    def shutdown(self, how: int) -> None:
        self.shutdown_called = True

    def close(self) -> None:
        self.closed = True


def test_slack_wait_for_reply_honors_timeout() -> None:
    backend = SlackChannelBackend()
    backend._client = cast(Any, _FakeSlackClient())
    assert backend.wait_for_reply("c", "t", "missing", timeout_seconds=0.01) is None


def test_slack_wait_for_reply_returns_new_matching_message() -> None:
    backend = SlackChannelBackend()
    backend._client = cast(Any, _FakeSlackClient())
    assert backend.wait_for_reply("c", "t", "u1", timeout_seconds=0.1) == "reply"


def test_whatsapp_disconnect_stops_server() -> None:
    backend = WhatsAppChannelBackend()
    assert backend._start_webhook_server(port=18080)
    assert backend._webhook_server is not None
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_webhook_connect_failure_is_reported() -> None:
    backend = LineChannelBackend()
    backend._message_queue = queue.Queue()
    assert backend._start_webhook_server(port=18083)
    conflict = LineChannelBackend()
    assert not conflict._start_webhook_server(port=18083)
    assert "bind failed" in conflict.connection_info.lower()
    backend.disconnect()


def test_synology_disconnect_stops_server() -> None:
    backend = SynologyChatChannelBackend()
    assert backend._start_webhook_server(port=18081)
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_zalo_disconnect_stops_server() -> None:
    backend = ZaloChannelBackend()
    assert backend._start_webhook_server(port=18082)
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_irc_disconnect_closes_socket_and_joins_thread() -> None:
    backend = IRCChannelBackend()
    fake_sock = _FakeSocket()
    backend._sock = cast(Any, fake_sock)
    thread = threading.Thread(target=lambda: time.sleep(0.01))
    thread.start()
    backend._reader_thread = thread
    backend.disconnect()
    assert fake_sock.shutdown_called
    assert fake_sock.closed
    assert backend._reader_thread is None


# ---------------------------------------------------------------------------
# Regression: message-loss race in _dispatch_message (§37 / review Bug 1)
#
# Old bug: _dispatch_message had a `state.lock.locked()` shortcut. When a
# worker was about to release its lock (queue drained, lock still held), the
# dispatcher saw locked()=True, assumed the worker would pick up the new
# message, and skipped spawning. The worker then released the lock → message
# orphaned until the next inbound message for that sender.
#
# Fix: always call _start_sender_worker(), which uses lock.acquire(blocking=
# False) as the sole gate. If the lock is free a new worker starts and drains
# the queue; if held the existing worker will drain it.
# ---------------------------------------------------------------------------


class _TrackingBackend:
    """Minimal backend that records sent messages for testing."""

    connection_info = "test"

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._lock = threading.Lock()

    def connect(self) -> bool:
        return True

    def find_channel(self, name: str) -> str:
        return "ch"

    def join_channel(self, channel_id: str) -> None:
        pass

    def poll_messages(self, channel_id: str, oldest: str) -> tuple:
        return [], oldest

    def is_from_bot(self, msg: dict) -> bool:
        return False

    def strip_bot_mention(self, text: str) -> str:
        return text

    def send_message(
        self, channel_id: str, text: str, thread_ts: str = ""
    ) -> None:
        with self._lock:
            self.sent.append(text)

    def disconnect(self) -> None:
        pass


def test_dispatch_no_message_loss_under_rapid_fire() -> None:
    """All messages dispatched rapidly to one thread are eventually processed.

    Regression test for the message-loss race where a worker finishing its
    queue could miss a newly enqueued message because the dispatcher relied
    on lock.locked() instead of always attempting to start a worker.
    """
    # We exercise the queue+worker machinery directly (not the full daemon
    # poll loop) to focus on the _dispatch_message → _start_sender_worker
    # → _process_sender_queue path with real threads.
    daemon = ChannelDaemon(
        backend=_TrackingBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )

    processed: list[str] = []
    processed_lock = threading.Lock()

    # Intercept _handle_message to record which messages get processed
    # without needing a real LLM agent.  We add a small delay to simulate
    # agent work, which widens the race window that the old code had.
    def tracking_handle(
        session_key: str,
        channel_id: str,
        msg: dict[str, Any],
        state: _SenderState,
    ) -> None:
        text = msg.get("text", "")
        time.sleep(0.005)  # simulate brief work
        with processed_lock:
            processed.append(text)

    daemon._handle_message = tracking_handle  # type: ignore[method-assign]

    # All messages share the same thread_ts so they route to one session.
    n_messages = 20
    for i in range(n_messages):
        daemon._dispatch_message(
            "ch",
            {"user": "u1", "text": f"msg-{i}", "thread_ts": "root", "ts": f"1.{i:06d}"},
        )
        # Small stagger so some dispatches happen while worker is mid-drain
        if i % 5 == 0:
            time.sleep(0.01)

    # Wait for all workers to finish (session key is now channel:thread_root)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = daemon._get_sender_state("ch:root")
        if state.pending_messages.empty() and not state.lock.locked():
            break
        time.sleep(0.01)

    daemon._join_handler_threads()

    with processed_lock:
        assert len(processed) == n_messages, (
            f"Expected {n_messages} messages processed, got {len(processed)}. "
            f"Missing: {set(f'msg-{i}' for i in range(n_messages)) - set(processed)}"
        )
        assert set(processed) == {f"msg-{i}" for i in range(n_messages)}


def test_dispatch_no_locked_shortcut() -> None:
    """_dispatch_message does not use state.lock.locked() as a gate.

    The old buggy code checked `if state.lock.locked(): return` to skip
    worker spawning.  This was the root cause of message loss.
    """
    source = inspect.getsource(ChannelDaemon._dispatch_message)
    assert "lock.locked()" not in source, (
        "_dispatch_message must not use lock.locked() — "
        "this was the root cause of the message-loss race"
    )


def test_dispatch_always_calls_start_sender_worker() -> None:
    """Every _dispatch_message call attempts to start a worker.

    After queuing the message, _dispatch_message must unconditionally call
    _start_sender_worker so the lock.acquire(blocking=False) gate is the
    only thing that decides whether a new worker starts.
    """
    source = inspect.getsource(ChannelDaemon._dispatch_message)
    assert "self._start_sender_worker(" in source


# ---------------------------------------------------------------------------
# Regression: concurrent-worker budget reset (§36 / review semantic issue)
#
# Old bug: reset_global_budget() was called per-task inside _handle_message.
# When two senders had concurrent workers, each worker's per-task reset
# zeroed the budget accumulated by the other worker mid-execution.
#
# Fix: reset_global_budget() is called once at daemon start (in run()),
# not per-task.  Workers accumulate cost independently without interference.
# ---------------------------------------------------------------------------


def test_budget_reset_at_daemon_start_not_per_task() -> None:
    """reset_global_budget is in run(), not in _handle_message or _process_sender_queue.

    Calling it per-task would zero budget accumulated by concurrent workers.
    """
    run_source = inspect.getsource(ChannelDaemon.run)
    assert "reset_global_budget" in run_source

    handle_source = inspect.getsource(ChannelDaemon._handle_message)
    assert "reset_global_budget" not in handle_source

    queue_source = inspect.getsource(ChannelDaemon._process_sender_queue)
    assert "reset_global_budget" not in queue_source


# ---------------------------------------------------------------------------
# Thread-session semantics: top-level → new session, thread reply → same
# ---------------------------------------------------------------------------


def test_top_level_messages_get_separate_sessions() -> None:
    """Each top-level message creates a distinct session (new chat)."""
    daemon = ChannelDaemon(
        backend=_TrackingBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )
    # Two top-level messages with different ts values.
    daemon._dispatch_message("ch", {"user": "u1", "text": "first", "ts": "100.0"})
    daemon._dispatch_message("ch", {"user": "u1", "text": "second", "ts": "200.0"})
    # They should create two distinct session states.
    state1 = daemon._sender_states.get("ch:100.0")
    state2 = daemon._sender_states.get("ch:200.0")
    assert state1 is not None
    assert state2 is not None
    assert state1 is not state2


def test_thread_reply_reuses_parent_session() -> None:
    """A thread reply routes to the same session as the parent top-level msg."""
    daemon = ChannelDaemon(
        backend=_TrackingBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )
    # Top-level message.
    daemon._dispatch_message("ch", {"user": "u1", "text": "hello", "ts": "100.0"})
    # Thread reply (thread_ts == parent ts, ts is reply's own timestamp).
    daemon._dispatch_message(
        "ch",
        {"user": "u1", "text": "reply", "thread_ts": "100.0", "ts": "100.1"},
    )
    # Both should be in the same session, keyed by thread root "100.0".
    state = daemon._sender_states.get("ch:100.0")
    assert state is not None
    assert state.pending_messages.qsize() >= 1  # at least the reply is queued
    # And there should be NO session for the reply's own ts.
    assert daemon._sender_states.get("ch:100.1") is None


def test_top_level_message_registers_active_thread() -> None:
    """Top-level message registers its ts in _active_threads when backend
    supports poll_thread_messages."""

    class _ThreadableBackend(_TrackingBackend):
        def poll_thread_messages(
            self, channel_id: str, thread_ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [], oldest

    daemon = ChannelDaemon(
        backend=_ThreadableBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )
    daemon._dispatch_message("ch", {"user": "u1", "text": "hi", "ts": "50.0"})
    assert "50.0" in daemon._active_threads


def test_thread_reply_does_not_register_active_thread() -> None:
    """Thread replies should NOT register a new active thread entry."""

    class _ThreadableBackend(_TrackingBackend):
        def poll_thread_messages(
            self, channel_id: str, thread_ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [], oldest

    daemon = ChannelDaemon(
        backend=_ThreadableBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )
    # Only a thread reply, no prior top-level message.
    daemon._dispatch_message(
        "ch",
        {"user": "u1", "text": "reply", "thread_ts": "50.0", "ts": "50.1"},
    )
    # "50.0" is the thread root but came from a reply, not a top-level msg —
    # "50.1" definitely should NOT be registered.
    assert "50.1" not in daemon._active_threads


def test_no_active_threads_without_poll_thread_support() -> None:
    """Backends without poll_thread_messages don't register active threads."""
    daemon = ChannelDaemon(
        backend=_TrackingBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )
    assert daemon._poll_thread_fn is None
    daemon._dispatch_message("ch", {"user": "u1", "text": "hi", "ts": "50.0"})
    assert len(daemon._active_threads) == 0


def test_poll_active_threads_dispatches_replies() -> None:
    """_poll_active_threads picks up thread replies and dispatches them."""
    replies_for: dict[str, list[dict[str, Any]]] = {
        "100.0": [
            {"user": "u1", "text": "user reply", "thread_ts": "100.0", "ts": "100.5"},
        ],
    }

    class _ThreadableBackend(_TrackingBackend):
        def poll_thread_messages(
            self, channel_id: str, thread_ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            msgs = replies_for.pop(thread_ts, [])
            new_oldest = oldest
            for m in msgs:
                ts = float(m.get("ts", "0"))
                if ts >= float(new_oldest):
                    new_oldest = f"{ts + 0.000001:.6f}"
            return msgs, new_oldest

    daemon = ChannelDaemon(
        backend=_ThreadableBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
    )

    dispatched: list[str] = []

    # Replace _dispatch_message to track what gets dispatched from thread poll.
    orig_dispatch = daemon._dispatch_message

    def tracking_dispatch(channel_id: str, msg: dict[str, Any]) -> None:
        dispatched.append(msg.get("text", ""))
        orig_dispatch(channel_id, msg)

    daemon._dispatch_message = tracking_dispatch  # type: ignore[method-assign]

    # Register a thread manually.
    with daemon._active_threads_lock:
        daemon._active_threads["100.0"] = "100.000001"

    daemon._poll_active_threads("ch")
    assert "user reply" in dispatched


def test_poll_active_threads_skips_bot_and_disallowed() -> None:
    """_poll_active_threads filters bot messages and non-allowed users."""

    class _ThreadableBackend(_TrackingBackend):
        def is_from_bot(self, msg: dict[str, Any]) -> bool:
            return msg.get("user") == "bot"

        def poll_thread_messages(
            self, channel_id: str, thread_ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [
                {"user": "bot", "text": "bot msg", "thread_ts": "1.0", "ts": "1.1"},
                {"user": "blocked", "text": "blocked msg", "thread_ts": "1.0", "ts": "1.2"},
                {"user": "allowed", "text": "ok msg", "thread_ts": "1.0", "ts": "1.3"},
            ], "1.300001"

    daemon = ChannelDaemon(
        backend=_ThreadableBackend(),  # type: ignore[arg-type]
        channel_name="",
        agent_name="test",
        allow_users=["allowed"],
    )

    dispatched: list[str] = []

    def tracking_handle(
        session_key: str,
        channel_id: str,
        msg: dict[str, Any],
        state: _SenderState,
    ) -> None:
        dispatched.append(msg.get("text", ""))

    daemon._handle_message = tracking_handle  # type: ignore[method-assign]

    with daemon._active_threads_lock:
        daemon._active_threads["1.0"] = "1.000001"

    daemon._poll_active_threads("ch")
    # Wait briefly for the worker to process.
    time.sleep(0.1)
    daemon._join_handler_threads()
    # Only the allowed user's message should have been dispatched.
    assert dispatched == ["ok msg"]


def test_session_key_uses_thread_root() -> None:
    """Verify _dispatch_message uses thread_root (not user) for session key."""
    source = inspect.getsource(ChannelDaemon._dispatch_message)
    assert "thread_root" in source or "thread_ts" in source
    # Must NOT key by user_id.
    assert 'f"{channel_id}:{user_id}"' not in source


