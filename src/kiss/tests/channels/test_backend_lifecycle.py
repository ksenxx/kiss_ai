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
    """All messages dispatched rapidly to one sender are eventually processed.

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

    n_messages = 20
    for i in range(n_messages):
        daemon._dispatch_message("ch", {"user": "u1", "text": f"msg-{i}"})
        # Small stagger so some dispatches happen while worker is mid-drain
        if i % 5 == 0:
            time.sleep(0.01)

    # Wait for all workers to finish
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = daemon._get_sender_state("ch:u1")
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


