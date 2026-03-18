"""Tests for background_agent single-instance lock and chunked messaging."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

from filelock import FileLock

from kiss.agents.claw.background_agent import (
    _LOCK_FILE,
    _MAX_CHUNK,
    _PID_FILE,
    _clear_stale_lock,
    _is_pid_alive,
    _read_pid,
    _send_chunked,
    run_background_agent,
    stop_background_agent,
)


class RecordingBackend:
    """Minimal ChannelBackend implementation that records sent messages."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []  # (channel, text, thread_ts)

    def connect(self) -> bool:
        return True

    @property
    def connection_info(self) -> str:
        return ""

    def find_channel(self, name: str) -> str | None:
        return None

    def find_user(self, username: str) -> str | None:
        return None

    def join_channel(self, channel_id: str) -> None:
        pass

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        self.sent.append((channel_id, text, thread_ts))

    def wait_for_reply(self, channel_id: str, thread_ts: str, user_id: str) -> str:
        return ""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        return False

    def strip_bot_mention(self, text: str) -> str:
        return text


def _run_in_child(result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Run run_background_agent in a child process, capture output."""
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        run_background_agent(work_dir="/tmp/test_claw")
    except SystemExit:
        pass
    except Exception as e:
        buf.write(f"EXCEPTION: {e}")
    finally:
        sys.stdout = old_stdout
    result_queue.put(buf.getvalue())


def test_single_instance_lock() -> None:
    """When one instance holds the lock, a second instance exits immediately."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Acquire the lock in this process to simulate a running instance
    lock = FileLock(_LOCK_FILE, timeout=0)
    lock.acquire()

    try:
        # Start a child process that tries to run the background agent
        q: multiprocessing.Queue[str] = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_in_child, args=(q,))
        p.start()
        p.join(timeout=15)

        output = q.get(timeout=5)
        assert "Another background agent instance is already running" in output
    finally:
        lock.release()
        _LOCK_FILE.unlink(missing_ok=True)


def test_send_chunked_short_message() -> None:
    """A message under the limit is sent as a single message."""
    b = RecordingBackend()
    _send_chunked(b, "C1", "t1", "hello world")
    assert len(b.sent) == 1
    assert b.sent[0] == ("C1", "hello world", "t1")


def test_send_chunked_exact_limit() -> None:
    """A message exactly at the limit is sent as a single message."""
    b = RecordingBackend()
    text = "x" * _MAX_CHUNK
    _send_chunked(b, "C1", "t1", text)
    assert len(b.sent) == 1
    assert b.sent[0][1] == text


def test_send_chunked_splits_at_newline() -> None:
    """A long message is split at the last newline before the limit."""
    b = RecordingBackend()
    # Create a message with a newline near the end of the first chunk
    line1 = "a" * (_MAX_CHUNK - 100)
    line2 = "b" * 50
    line3 = "c" * 200
    text = line1 + "\n" + line2 + "\n" + line3
    _send_chunked(b, "C1", "t1", text)
    assert len(b.sent) == 2
    assert b.sent[0][1] == line1 + "\n" + line2
    # Second chunk is the remainder (leading newline stripped)
    assert b.sent[1][1] == line3


def test_send_chunked_no_newline_splits_at_limit() -> None:
    """A long message without newlines splits at _MAX_CHUNK."""
    b = RecordingBackend()
    text = "x" * (_MAX_CHUNK + 500)
    _send_chunked(b, "C1", "t1", text)
    assert len(b.sent) == 2
    assert b.sent[0][1] == "x" * _MAX_CHUNK
    assert b.sent[1][1] == "x" * 500


def test_send_chunked_multiple_chunks() -> None:
    """A very long message is split into multiple chunks."""
    b = RecordingBackend()
    text = "x" * (_MAX_CHUNK * 3 + 100)
    _send_chunked(b, "C1", "t1", text)
    assert len(b.sent) == 4
    for i in range(3):
        assert len(b.sent[i][1]) == _MAX_CHUNK
    assert len(b.sent[3][1]) == 100


def test_send_chunked_empty_message() -> None:
    """An empty message sends nothing."""
    b = RecordingBackend()
    _send_chunked(b, "C1", "t1", "")
    assert len(b.sent) == 0


def test_lock_released_after_exit() -> None:
    """After the lock holder exits, a new instance can acquire the lock."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.unlink(missing_ok=True)

    # Acquire and release the lock
    lock = FileLock(_LOCK_FILE, timeout=0)
    lock.acquire()
    lock.release()

    # A new lock acquisition should succeed
    lock2 = FileLock(_LOCK_FILE, timeout=0)
    lock2.acquire()
    lock2.release()
    _LOCK_FILE.unlink(missing_ok=True)


def test_read_pid_missing_file() -> None:
    """_read_pid returns None when the PID file doesn't exist."""
    _PID_FILE.unlink(missing_ok=True)
    assert _read_pid() is None


def test_read_pid_valid() -> None:
    """_read_pid returns the stored PID."""
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text("12345")
    try:
        assert _read_pid() == 12345
    finally:
        _PID_FILE.unlink(missing_ok=True)


def test_read_pid_invalid_content() -> None:
    """_read_pid returns None for non-integer content."""
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text("not-a-pid")
    try:
        assert _read_pid() is None
    finally:
        _PID_FILE.unlink(missing_ok=True)


def test_is_pid_alive_current_process() -> None:
    """_is_pid_alive returns True for the current process."""
    assert _is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead_process() -> None:
    """_is_pid_alive returns False for a non-existent PID."""
    # Use a very high PID that's unlikely to exist
    assert _is_pid_alive(99999999) is False


def test_clear_stale_lock_removes_dead_pid() -> None:
    """_clear_stale_lock removes lock/PID files when the process is dead."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.write_text("")
    _PID_FILE.write_text("99999999")
    _clear_stale_lock()
    assert not _LOCK_FILE.exists()
    assert not _PID_FILE.exists()


def test_clear_stale_lock_keeps_live_pid() -> None:
    """_clear_stale_lock preserves files when the process is alive."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.write_text("")
    _PID_FILE.write_text(str(os.getpid()))
    try:
        _clear_stale_lock()
        assert _LOCK_FILE.exists()
        assert _PID_FILE.exists()
    finally:
        _LOCK_FILE.unlink(missing_ok=True)
        _PID_FILE.unlink(missing_ok=True)


def test_clear_stale_lock_no_pid_file() -> None:
    """_clear_stale_lock does nothing when there's no PID file."""
    _PID_FILE.unlink(missing_ok=True)
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.write_text("")
    _clear_stale_lock()
    # Lock file should still exist (no PID to check)
    assert _LOCK_FILE.exists()
    _LOCK_FILE.unlink(missing_ok=True)


def test_stop_background_agent_no_pid_file() -> None:
    """stop_background_agent returns False when no PID file exists."""
    _PID_FILE.unlink(missing_ok=True)
    assert stop_background_agent() is False


def test_stop_background_agent_dead_pid() -> None:
    """stop_background_agent cleans up stale files for dead PID."""
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text("99999999")
    _LOCK_FILE.write_text("")
    result = stop_background_agent()
    assert result is False
    assert not _PID_FILE.exists()
    assert not _LOCK_FILE.exists()


def test_single_instance_lock_shows_pid() -> None:
    """When lock is held and PID file exists, error message includes PID."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    lock = FileLock(_LOCK_FILE, timeout=0)
    lock.acquire()

    try:
        q: multiprocessing.Queue[str] = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_in_child, args=(q,))
        p.start()
        p.join(timeout=15)

        output = q.get(timeout=5)
        assert "already running" in output
        assert str(os.getpid()) in output
        assert "--stop" in output
    finally:
        lock.release()
        _LOCK_FILE.unlink(missing_ok=True)
        _PID_FILE.unlink(missing_ok=True)
