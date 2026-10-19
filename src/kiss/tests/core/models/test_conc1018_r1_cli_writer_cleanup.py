# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_CLIProcess`` must never leave its stdin writer running past ``close()``.

Review R1, issue I1: ``send_prompt`` started the writer thread *before*
storing it in ``self._writer``.  A Stop (``KeyboardInterrupt``, injected
asynchronously into task threads) landing between the two statements
left a running writer that ``close()`` never joined; ``close()`` then
closed stdin while the writer was between its cancel check and its
``os.write``, so the bytes went to a closed descriptor, or to whatever
file reused its number.

The first test replays that exact interleaving with real threads and a
real child process: a profile hook parks the writer right before its
first ``os.write``; a trace hook raises ``KeyboardInterrupt`` on the first
line of ``send_prompt`` executed after ``writer.start()`` returned.  The
writer is released only once ``close()`` is blocked joining it (the fix)
or has already returned (the bug), so the outcome does not depend on
scheduling.

The second test keeps the earlier B1 fix: when ``Thread.start()`` itself
raises (``RLIMIT_NPROC`` set to 1 in a fresh interpreter makes
``pthread_create`` fail with a real ``can't start new thread``),
``close()`` must not try to join the never-started thread and must still
release every pipe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any

import pytest

from kiss.core.models.model import _CLIProcess

_SLEEPING_CHILD = [sys.executable, "-c", "import time; time.sleep(30)"]


def _frames_of(thread_ident: int) -> list[Any]:
    """Return the live frame stack of the thread with *thread_ident*, innermost first."""
    frame = sys._current_frames().get(thread_ident)
    frames = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    return frames


def _is_joining(thread_ident: int, target: threading.Thread) -> bool:
    """Return whether the thread with *thread_ident* is inside ``target.join()``."""
    return any(
        frame.f_code is threading.Thread.join.__code__ and frame.f_locals.get("self") is target
        for frame in _frames_of(thread_ident)
    )


def test_stop_right_after_writer_start_joins_writer_before_stdin_closes() -> None:
    """A Stop raised right after ``writer.start()`` still gets the writer joined by ``close()``."""
    proc = _CLIProcess(_SLEEPING_CHILD, "r1", 60)
    stdin = proc._proc.stdin
    assert stdin is not None
    ready = threading.Event()  # the writer is parked before its first os.write
    release = threading.Event()  # the writer may go on
    closed = threading.Event()  # close() has returned
    main_ident = threading.get_ident()
    writers: list[threading.Thread] = []
    stdin_closed_at_write: list[bool] = []

    def writer_profile(frame: Any, event: str, arg: Any) -> None:
        if event == "c_call" and arg is os.write and not ready.is_set():
            ready.set()
            assert release.wait(30)
            stdin_closed_at_write.append(stdin.closed)

    def caller_trace(frame: Any, event: str, arg: Any) -> Any:
        if event == "line" and frame.f_code is _CLIProcess.send_prompt.__code__:
            writer = frame.f_locals.get("writer")
            if writer is not None and writer.ident is not None and not writers:
                writers.append(writer)
                assert ready.wait(10)
                raise KeyboardInterrupt("Stop right after writer.start()")
        return caller_trace

    def release_writer_when_close_cannot_race() -> None:
        assert ready.wait(30)
        while not closed.wait(0.005):
            if _is_joining(main_ident, writers[0]):
                break
        release.set()

    releaser = threading.Thread(target=release_writer_when_close_cannot_race)
    releaser.start()
    threading.setprofile(writer_profile)
    sys.settrace(caller_trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            with proc:
                proc.send_prompt("R1_PROMPT_BYTES")
        writer_alive_after_close = writers[0].is_alive()
    finally:
        sys.settrace(None)
        threading.setprofile(None)
        closed.set()
        release.set()
        releaser.join(30)
        for writer in writers:
            writer.join(30)
    assert not writer_alive_after_close, "close() returned with the stdin writer still running"
    assert stdin_closed_at_write == [False], "the writer wrote after close() closed stdin"
    assert stdin.closed
    assert proc._proc.returncode is not None


# Prints how close() fared after Thread.start() failed inside send_prompt.
_START_FAILURE_PROBE = """
import json, resource, sys
from kiss.core.models.model import _CLIProcess
proc = _CLIProcess([sys.executable, "-c", "import time; time.sleep(30)"], "r1", 60)
soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
error = None
try:
    with proc:
        proc.send_prompt("never written")
except RuntimeError as e:
    error = str(e)
finally:
    resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
p = proc._proc
print(json.dumps({
    "error": error,
    "returncode": p.returncode,
    "pipes_closed": [p.stdin.closed, p.stdout.closed, p.stderr.closed],
    "writer_alive": proc._writer is not None and proc._writer.is_alive(),
}))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_NPROC fault injection is Linux-only")
def test_writer_start_failure_still_releases_every_pipe() -> None:
    """``close()`` after a failed ``Thread.start()`` closes all pipes and raises nothing itself."""
    if os.geteuid() == 0:
        pytest.skip("RLIMIT_NPROC is not enforced for root")
    result = subprocess.run(
        [sys.executable, "-c", _START_FAILURE_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["error"] == "can't start new thread", report
    assert report["pipes_closed"] == [True, True, True], report
    assert report["returncode"] is not None, report
    assert report["writer_alive"] is False, report
