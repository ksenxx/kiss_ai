# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R2 finding 3: the bounded file reads stay bounded while the file grows.

``_read_end`` (job-log tails and ``until=`` scans) and ``edit_guards._file_ends``
(the file a ``cat``/``tail`` showed) measured the file, seeked to a bounded
window and then read to EOF, so bytes a writer appended in between (a job
log does grow) were all read.  A real writer appends 2 MB right before the
read: a ``sys.setprofile`` hook runs it when the helper calls the file's
``read`` (instrumentation of the real path, nothing replaced).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.agents.sorcar import edit_guards
from kiss.agents.sorcar.useful_tools import _read_end, _tail_lines

GROWTH = b"x" * 2_000_000 + b"\n"


class _GrowBeforeRead:
    """Profile hook: append :data:`GROWTH` to *path* before its *nth* ``read`` call."""

    def __init__(self, path: Path, nth: int) -> None:
        self.path = path
        self.nth = nth
        self.reads = 0

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        owner = getattr(arg, "__self__", None)
        if event != "c_call" or getattr(arg, "__name__", "") != "read":
            return
        if os.fspath(getattr(owner, "name", "")) != str(self.path):
            return
        self.reads += 1
        if self.reads == self.nth:
            with open(self.path, "ab") as writer:
                writer.write(GROWTH)


def _call_while_growing(path: Path, nth: int, fn: Any, *args: Any) -> Any:
    hook = _GrowBeforeRead(path, nth)
    sys.setprofile(hook)
    try:
        return fn(*args)
    finally:
        sys.setprofile(None)
        assert hook.reads >= nth, "the file was never read"
        assert path.stat().st_size > len(GROWTH), "the writer never appended"


def _log(path: Path) -> Path:
    """Write 1,000 lines of 12 bytes to *path*; return it."""
    path.write_bytes(b"".join(b"line %06d\n" % i for i in range(1000)))
    return path


def test_read_end_reads_no_more_than_its_window(tmp_path: Path) -> None:
    log = _log(tmp_path / "job.log")
    # The last 1,200 bytes measured are the last 100 whole lines.
    text = _call_while_growing(log, 1, _read_end, log, 1200)
    assert text == "".join(f"line {i:06d}\n" for i in range(900, 1000))
    log = _log(tmp_path / "other.log")
    assert _call_while_growing(log, 1, _tail_lines, log, 2, 1200) == (
        "line 000998\nline 000999\n")


def test_file_ends_reads_no_more_than_its_two_ends(tmp_path: Path) -> None:
    shown = edit_guards._SHOWN_FILE_BYTES
    viewed = tmp_path / "big.txt"
    viewed.write_bytes(b"a" * shown + b"tail\n")
    # The head is read first, the tail second: the file grows before the tail read.
    text = _call_while_growing(viewed, 2, edit_guards._file_ends, viewed)
    assert len(text) <= 2 * shown + 1
    assert text == "a" * shown + "tail\n"
