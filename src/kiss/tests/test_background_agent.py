"""Tests for background_agent single-instance lock."""

from __future__ import annotations

import multiprocessing

from filelock import FileLock

from kiss.agents.claw.background_agent import _LOCK_FILE, run_background_agent


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
