# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 5: Read/Write on a non-regular file must not hang.

Pre-fix, ``UsefulTools.Read`` called ``Path.read_text()`` on any existing
non-directory path.  Opening a FIFO for reading blocks until a writer
appears, so a model calling ``Read`` on a named pipe (or e.g. ``Write`` to
one) hung the agent FOREVER with no timeout.  Devices/sockets have the same
failure mode.  Both tools must instead return an error string immediately.
"""

import os
import threading

import pytest

from kiss.core.useful_tools import UsefulTools


def _run_with_timeout(fn, timeout: float = 5.0) -> str | None:
    """Run *fn* in a daemon thread; return its result or None on hang."""
    box: list[str] = []

    def _target() -> None:
        box.append(fn())

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None
    return box[0]


def _unblock_fifo_reader(fifo: str) -> None:
    """If a reader is blocked on *fifo*, satisfy it so the thread exits."""
    try:
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
    except OSError:
        pass


def test_read_on_fifo_returns_error_immediately(tmp_path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no mkfifo")
    fifo = str(tmp_path / "pipe")
    os.mkfifo(fifo)
    tools = UsefulTools(work_dir=str(tmp_path))
    try:
        result = _run_with_timeout(lambda: tools.Read(fifo))
    finally:
        _unblock_fifo_reader(fifo)
    assert result is not None, "Read on a FIFO hung the agent (blocked > 5s)"
    assert result.startswith("Error"), result
    assert "regular file" in result, result


def test_write_to_fifo_returns_error_immediately(tmp_path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no mkfifo")
    fifo = str(tmp_path / "pipe")
    os.mkfifo(fifo)
    tools = UsefulTools(work_dir=str(tmp_path))

    def _unblock_writer() -> None:
        try:
            fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError:
            pass

    try:
        result = _run_with_timeout(lambda: tools.Write(fifo, "hello"))
    finally:
        _unblock_writer()
    assert result is not None, "Write to a FIFO hung the agent (blocked > 5s)"
    assert result.startswith("Error"), result


def test_read_on_char_device_returns_error() -> None:
    """/dev/zero is an infinite stream; Read must refuse it, not OOM/hang."""
    if not os.path.exists("/dev/zero"):
        pytest.skip("no /dev/zero")
    tools = UsefulTools()
    result = _run_with_timeout(lambda: tools.Read("/dev/zero"))
    assert result is not None, "Read on /dev/zero hung"
    assert result.startswith("Error"), result


def test_read_symlink_to_regular_file_still_works(tmp_path) -> None:
    """Regression guard: symlinks to regular files must keep working."""
    target = tmp_path / "real.txt"
    target.write_text("hello symlink")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Read(str(link)) == "hello symlink"
