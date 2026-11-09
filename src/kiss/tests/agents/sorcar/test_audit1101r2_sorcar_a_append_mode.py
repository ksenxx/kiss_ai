# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 round 2 (sorcar_a F2): a redirected append lands in append mode.

The ``Write`` a ``cat >> f`` here-document is redirected to used to
reread the file and overwrite it with that text followed by the body:
a writer appending between the reread and the overwrite was still
erased.  The Write now opens the file for appending (``O_APPEND``) and
writes the body only, as the shell's ``>>`` does, so no interleaving
loses a line.

The concurrent writer is a real shell append; a trace of the test
thread (:class:`AppendWhen`) chooses when it lands, right before the
Write opens the file for writing, and replaces nothing.  The container
variant lives in ``test_audit1101_s5_append_redirect.py``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, _write_file, _write_regular
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash here-documents")


class AppendWhen:
    """A trace of the current thread making *append* once, when *entered* first holds.

    The trace sees every function call of the thread (``sys.settrace``)
    and runs *append* (a real shell append) on the first call whose frame
    satisfies *entered*: it chooses the interleaving and replaces nothing.

    Args:
        entered: Whether the frame just called is the one to append before.
        append: Makes the concurrent append.
    """

    def __init__(self, entered: Callable[[FrameType], bool], append: Callable[[], Any]) -> None:
        self.entered = entered
        self.append = append
        self.made = 0
        self._previous: Any = None

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and not self.made and self.entered(frame):
            self.made += 1
            self.append()
        return None

    def __enter__(self) -> AppendWhen:
        self._previous = sys.gettrace()
        sys.settrace(self)
        return self

    def __exit__(self, *exc: object) -> None:
        sys.settrace(self._previous)


def opens_for_writing(frame: FrameType) -> bool:
    """Whether *frame* is the Write about to write the file.

    An append opens the file itself (:func:`_write_regular`, for
    appending); an overwrite writes a sibling temporary file and renames
    it over the file (:func:`_write_file`), so its frame's entry is the
    moment before the file changes.
    """
    return frame.f_code in (_write_regular.__code__, _write_file.__code__)


def shell_append(cwd: Path, name: str, line: str) -> None:
    """Append *line* to the file *name* under *cwd* through a real shell ``>>``."""
    subprocess.run(["bash", "-c", f"echo {line} >> {name}"], cwd=cwd, check=True)


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UsefulTools:
    """A redirecting tool set on a work dir with no detach grace."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.redirect = True
    return tools


def _redirected_append(tools: UsefulTools, command: str) -> ToolRedirect:
    with pytest.raises(ToolRedirect) as raised:
        tools.Bash(command, "append", timeout_seconds=5)
    return raised.value


def test_an_append_landing_as_the_write_opens_the_file_is_kept(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """The body goes after what the file holds when the Write opens it, whatever landed before."""
    log = tmp_path / "log.txt"
    log.write_text("initial\n")
    tools.Read(str(log))
    redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nrequested\nEOF")
    assert redirect.arguments == {"file_path": str(log), "content": "initial\nrequested\n"}

    def append_concurrently() -> None:
        shell_append(tmp_path, "log.txt", "concurrent")

    with AppendWhen(opens_for_writing, append_concurrently) as landed:
        out = tools.Write(**redirect.arguments)
    assert landed.made == 1, "the concurrent append did not land before the Write opened the file"
    assert out == f"Successfully wrote 18 characters to {log}"
    assert log.read_text() == "initial\nconcurrent\nrequested\n"
    # The pending append was consumed: the same Write again is the plain overwrite it says,
    # whatever lands as it opens the file.
    with AppendWhen(opens_for_writing, append_concurrently) as landed:
        out = tools.Write(**redirect.arguments)
    assert landed.made == 1
    assert out == f"Successfully wrote 18 characters to {log}"
    assert log.read_text() == "initial\nrequested\n"


def test_an_append_into_a_file_removed_meanwhile_recreates_it_with_the_body(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """As the shell's ``>>`` would: the file holds the body alone, in the mode a new file gets."""
    log = tmp_path / "log.txt"
    log.write_text("initial\n")
    tools.Read(str(log))
    redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nrequested\nEOF")
    log.unlink()
    assert tools.Write(**redirect.arguments) == f"Successfully wrote 18 characters to {log}"
    assert log.read_text() == "requested\n"


def enters_write_instead(frame: FrameType) -> bool:
    """Whether *frame* is ``UsefulTools._write_instead``: the file was checked, not yet reread."""
    return frame.f_code is UsefulTools._write_instead.__code__


def test_an_append_into_a_file_gone_since_the_check_keeps_what_another_writer_creates(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """A file removed between the command's check and its reread is still appended to.

    The redirect then carries the body alone, and the Write it names is
    still an append: a writer creating the file while the Write is being
    decided keeps its bytes, as under the shell's ``>>``.
    """
    log = tmp_path / "log.txt"
    log.write_text("initial\n")
    tools.Read(str(log))

    def remove_now() -> None:
        log.unlink()

    with AppendWhen(enters_write_instead, remove_now) as removed:
        redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nrequested\nEOF")
    assert removed.made == 1, "the file was not removed between the check and the reread"
    assert redirect.arguments == {"file_path": str(log), "content": "requested\n"}
    # Another writer creates the file while the Write is being decided.
    subprocess.run(["bash", "-c", "echo theirs > log.txt"], cwd=tmp_path, check=True)
    assert tools.Write(**redirect.arguments) == f"Successfully wrote 10 characters to {log}"
    assert log.read_text() == "theirs\nrequested\n"
