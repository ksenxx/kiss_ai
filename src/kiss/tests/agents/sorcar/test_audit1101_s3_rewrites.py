# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01, report S3 (useful_tools): the shell rewrites ``Bash`` applies as tool calls.

Through a real :class:`UsefulTools` on files in ``tmp_path``, nothing
mocked:

* F1: a here-document append into a FIFO is refused at once instead of
  blocking forever on reading the FIFO before the ``Write``;
* F2: the ``Write`` an append was redirected to appends to what the file
  holds by then, so what another writer appended in between is kept;
  the pending append is consumed once and forgotten when any other
  call touches the file;
* F6: a long ``sleep`` spelled inside quoted data is left as it is and
  the ``sleep`` the shell runs is the one taken out;
* F7: a ``sed -i`` on a symlink or a hard link is not applied as an
  ``Edit`` of the target (sed replaces the link with a new file); a
  python rewrite, which follows the link as python does, still is.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, _without_sleep
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.conftest import host_sed_is_gnu, posix_only

pytestmark = posix_only("the host Bash tool runs bash here-documents, sleeps and sed")


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UsefulTools:
    """A tool set on a work dir with no detach grace."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    return UsefulTools(work_dir=str(tmp_path))


def _job_id(started: str) -> str:
    assert started.startswith("Started background job "), started
    return started.split()[3]


# --- F1: an append into a FIFO ------------------------------------------------


def test_heredoc_append_into_a_fifo_is_refused_without_reading_it(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """Reading a FIFO blocks until a writer opens it: the append must not read it."""
    (tmp_path / "tmp").mkdir()
    fifo = tmp_path / "tmp" / "fifo"  # a scratch path: no Read is required first
    os.mkfifo(fifo)
    outcome: list[str] = []
    thread = threading.Thread(
        target=lambda: outcome.append(
            tools.Bash("cat <<'EOF' >> tmp/fifo\npayload\nEOF", "append", timeout_seconds=1)
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=5)
    try:
        assert not thread.is_alive(), "Bash is still blocked after 5 s"
    finally:
        if thread.is_alive():
            # Unblock the reader stuck in open(): a writer end that closes at once.
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            thread.join(timeout=5)
    assert outcome == [
        f"Error: {fifo} exists and is not a regular file (directory/FIFO/device/socket); "
        "refusing to append to it."
    ]
    # the same for tee -a, and a directory target
    out = tools.Bash("tee -a tmp/fifo <<'EOF'\npayload\nEOF", "append", timeout_seconds=1)
    assert out.startswith(f"Error: {fifo} exists and is not a regular file")
    (tmp_path / "tmp" / "dir").mkdir()
    out = tools.Bash("cat >> tmp/dir <<'EOF'\npayload\nEOF", "append", timeout_seconds=1)
    assert out.startswith(f"Error: {tmp_path / 'tmp' / 'dir'} exists and is not a regular file")
    # a regular scratch file still takes the append, read or not
    (tmp_path / "tmp" / "notes.md").write_text("old\n")
    out = tools.Bash("cat >> tmp/notes.md <<'EOF'\nmore\nEOF", "append", timeout_seconds=1)
    assert out.startswith("Appended through Write(")
    assert (tmp_path / "tmp" / "notes.md").read_text() == "old\nmore\n"


# --- F2: an append redirected to a Write ------------------------------------


def _redirected_append(tools: UsefulTools, command: str) -> ToolRedirect:
    with pytest.raises(ToolRedirect) as raised:
        tools.Bash(command, "append", timeout_seconds=5)
    return raised.value


def test_redirected_append_keeps_what_another_writer_appended_meanwhile(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    log = tmp_path / "log.txt"
    log.write_text("initial\n")
    tools.redirect = True
    tools.Read(str(log))
    redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nrequested-append\nEOF")
    assert redirect.name == "Write"
    assert redirect.arguments == {"file_path": str(log), "content": "initial\nrequested-append\n"}
    assert redirect.note.startswith("Appended through Write(")
    # Another process appends before the agent makes the Write the redirect names.
    subprocess.run(["bash", "-c", "echo external-writer >> log.txt"], cwd=tmp_path, check=True)
    out = tools.Write(**redirect.arguments)
    assert out == f"Successfully wrote 25 characters to {log}"
    assert log.read_text() == "initial\nexternal-writer\nrequested-append\n"
    # The pending append was consumed: the same Write again is the plain overwrite it says.
    assert tools.Write(**redirect.arguments).startswith("Successfully wrote 25 characters")
    assert log.read_text() == "initial\nrequested-append\n"
    # tee -a goes the same way, and a file that did not change is written as named.
    redirect = _redirected_append(tools, "tee -a log.txt <<'EOF'\nvia tee\nEOF")
    assert redirect.arguments["content"] == "initial\nrequested-append\nvia tee\n"
    assert tools.Write(**redirect.arguments).startswith("Successfully wrote 33 characters")
    assert log.read_text() == "initial\nrequested-append\nvia tee\n"


def test_pending_append_is_forgotten_when_another_call_touches_the_file(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    log = tmp_path / "log.txt"
    log.write_text("initial\n")
    tools.redirect = True
    tools.Read(str(log))
    redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nmore\nEOF")
    # A Read of the file in between: the Write is then the overwrite it names.
    tools.Read(str(log), force=True)
    log.write_text("initial\nexternal\n")
    assert tools.Write(**redirect.arguments).startswith("Successfully wrote")
    assert log.read_text() == "initial\nmore\n"
    # A Write of other content is a plain Write, and clears the pending append.
    redirect = _redirected_append(tools, "cat >> log.txt <<'EOF'\nmore\nEOF")
    assert tools.Write(str(log), "replaced\n").startswith("Successfully wrote")
    assert log.read_text() == "replaced\n"
    log.write_text("replaced\nexternal\n")
    assert tools.Write(**redirect.arguments).startswith("Successfully wrote")
    assert log.read_text() == "initial\nmore\nmore\n"
    # A later redirect for another file replaces the pending append.
    other = tmp_path / "other.txt"
    other.write_text("o\n")
    tools.Read(str(other))
    first = _redirected_append(tools, "cat >> log.txt <<'EOF'\nagain\nEOF")
    second = _redirected_append(tools, "cat >> other.txt <<'EOF'\nmore\nEOF")
    log.write_text("initial\nmore\nmore\nexternal\n")
    other.write_text("o\nexternal\n")
    assert tools.Write(**second.arguments).startswith("Successfully wrote")
    assert other.read_text() == "o\nexternal\nmore\n"
    assert tools.Write(**first.arguments).startswith("Successfully wrote")
    assert log.read_text() == "initial\nmore\nmore\nagain\n"
    # Without an agent the append is made here and the file reads as appended to.
    tools.redirect = False
    out = tools.Bash("cat >> log.txt <<'EOF'\nlast\nEOF", "append", timeout_seconds=5)
    assert out.startswith("Appended through Write(") and "Successfully wrote" in out
    assert log.read_text() == "initial\nmore\nmore\nagain\nlast\n"


# --- F6: a sleep spelled in quoted data ---------------------------------------


def test_sleep_in_quoted_data_is_kept_and_the_real_sleep_is_waited_on(
    tools: UsefulTools,
) -> None:
    job = _job_id(tools.Bash("sleep 0.5", "quick", background=True))
    before = time.monotonic()
    command = "printf '%s\\n' 'sleep 60'; timeout 1 sleep 60; echo ran"
    out = tools.Bash(command, "poll", timeout_seconds=100)
    waited = f"Waited on job {job} for up to 60 s instead of sleeping (exited with code 0).\n"
    assert out == "sleep 60\nran\n" + waited
    assert time.monotonic() - before < 10
    # the text search skips the words of a quoted argument and a here-document body
    assert _without_sleep("printf '%s\\n' 'sleep 60'; sleep 60", ["60"]) == (
        "printf '%s\\n' 'sleep 60'; true"
    )
    assert _without_sleep('echo "sleep 60"; sleep 60', ["60"]) == 'echo "sleep 60"; true'
    assert _without_sleep("cat <<'EOF'\nsleep 60\nEOF\nsleep 60", ["60"]) == (
        "cat <<'EOF'\nsleep 60\nEOF\ntrue"
    )
    assert _without_sleep("sleep 60; sleep 60", ["60"]) == "true; sleep 60"
    assert _without_sleep("echo $(sleep 60); sleep 60", ["60"]) == "echo $(true); sleep 60"
    # a sleep the text search finds nowhere but in data is not taken out at all
    assert _without_sleep("echo 'sleep 60'; sleep 6'0'", ["60"]) is None


# --- F7: a sed -i on a link -----------------------------------------------------


def test_sed_on_a_symlink_runs_as_sed_instead_of_editing_the_target(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old value\n")
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")
    tools.Read(str(link))
    out = tools.Bash("sed -i 's/old/new/' link.txt", "edit link", timeout_seconds=10)
    assert not out.startswith("Applied as"), out
    if not host_sed_is_gnu():
        pytest.skip("needs GNU sed as the oracle")
    # sed replaced the link with a regular file and left the target alone
    assert target.read_text() == "old value\n"
    assert not link.is_symlink() and link.read_text() == "new value\n"
    # a multi-line sed on a link stays refused, naming the Edit
    link.unlink()
    link.symlink_to("target.txt")
    out = tools.Bash("sed -i 's/old/new\\nnext/' link.txt", "edit link", timeout_seconds=10)
    assert out.startswith("REFUSED: the `sed -i` call did NOT run: it writes a multi-line")
    assert target.read_text() == "old value\n" and link.is_symlink()
    # a sed on a plain file is still applied as the Edit
    plain = tmp_path / "plain.txt"
    plain.write_text("old value\n")
    tools.Read(str(plain))
    out = tools.Bash("sed -i 's/old/new/' plain.txt", "edit", timeout_seconds=10)
    assert out.startswith("Applied as Edit(") and plain.read_text() == "new value\n"


def test_sed_on_a_hard_link_runs_as_sed_which_unlinks_it(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old value\n")
    hard = tmp_path / "hard.txt"
    os.link(target, hard)
    tools.Read(str(hard))
    out = tools.Bash("sed -i 's/old/new/' hard.txt", "edit link", timeout_seconds=10)
    assert not out.startswith("Applied as"), out
    if not host_sed_is_gnu():
        pytest.skip("needs GNU sed as the oracle")
    # sed wrote a new file under the link's name: the other name keeps the old text
    assert target.read_text() == "old value\n" and hard.read_text() == "new value\n"
    assert os.stat(target).st_nlink == 1


def test_python_rewrite_of_a_symlink_is_still_applied_as_edit(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """``open('link')`` follows the link as ``Edit`` does: the conversion is exact."""
    target = tmp_path / "target.txt"
    target.write_text("old value\n")
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")
    tools.Read(str(link))
    command = (
        "python3 - <<'EOF'\ns = open('link.txt').read()\ns = s.replace('old', 'new')\n"
        "open('link.txt', 'w').write(s)\nEOF"
    )
    out = tools.Bash(command, "rewrite", timeout_seconds=10)
    assert out.startswith("Applied as Edit("), out
    assert target.read_text() == "new value\n" and link.is_symlink()
