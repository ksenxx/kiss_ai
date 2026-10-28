# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-26 (sorcar-tools F1/F2): Docker tools after a Stop and the rewrite guard.

F1: ``DockerTools.Edit(edits=...)`` only stopped the batch on an inner
result starting with ``Error``.  A task Stop makes ``DockerManager.Bash``
return ``Killed: the task was stopped.`` (with the exit-code line after
it), which the loop recorded as ``edit 1: ...`` and then went on with the
next edit: every remaining edit still stored its payload through an exec
that was cancelled at once, each costing a worker thread and a bounded
reaper wait, and the report listed them all as results.  The batch now
returns the Stop result as is after the first stopped edit.

F2: ``DockerManager.Bash`` and ``run_commands_parallel`` applied only the
destructive/install guards; the host ``Bash`` also refuses the shell
rewrites ``Edit`` should make (``sed -i`` with a multi-line script, a
``python3 -`` heredoc that reads, replaces and writes a file back).  The
container tools now apply the same ``rewrite_guard``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]

_SED_REWRITE = "sed -i 's/a/b\\nc/' /tmp/some_file.txt"
_HEREDOC_REWRITE = (
    "python3 - <<'EOF'\n"
    "p = '/tmp/some_file.txt'\n"
    "s = open(p).read()\n"
    "s = s.replace('a', 'b')\n"
    "open(p, 'w').write(s)\n"
    "EOF"
)


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_stopped_edit_batch_ends_at_the_first_stopped_edit(
    env: DockerManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The batch returns the Stop result as is; the later edits never run."""
    tools = DockerTools(env.Bash)
    env.stop_event = threading.Event()
    try:
        env.Bash("printf 'one\\ntwo\\nthree\\n' > /tmp/batch_stop.txt", "seed")
        capsys.readouterr()
        env.stop_event.set()
        result = tools.Edit(
            "/tmp/batch_stop.txt",
            edits=json.dumps([
                {"old_string": "one", "new_string": "uno"},
                {"old_string": "two", "new_string": "dos"},
                {"old_string": "three", "new_string": "tres"},
            ]),
        )
    finally:
        env.stop_event = None
    assert STOPPED_MESSAGE in result, result
    assert not result.startswith("edit 1:"), result
    assert "edit 2" not in result and "edit 3" not in result, result
    # Only the first edit's payload store reached the container.
    assert capsys.readouterr().out.count("Edit /tmp/batch_stop.txt") == 1
    assert env.Bash("cat /tmp/batch_stop.txt", "cat").startswith("one\ntwo\nthree")


def test_failed_edit_batch_still_reports_the_failing_edit(env: DockerManager) -> None:
    """An ordinary failure keeps its ``Error: edit k`` report (unchanged contract)."""
    tools = DockerTools(env.Bash)
    env.Bash("printf 'one\\ntwo\\n' > /tmp/batch_fail.txt", "seed")
    result = tools.Edit(
        "/tmp/batch_fail.txt",
        edits=json.dumps([
            {"old_string": "one", "new_string": "uno"},
            {"old_string": "missing", "new_string": "x"},
        ]),
    )
    assert result.startswith("Error: edit 2: edits 1-1 applied; "), result
    assert env.Bash("cat /tmp/batch_fail.txt", "cat").startswith("uno\ntwo")


@pytest.mark.parametrize("command", [_SED_REWRITE, _HEREDOC_REWRITE])
def test_docker_bash_refuses_shell_rewrites(env: DockerManager, command: str) -> None:
    """The container ``Bash`` refuses the same rewrites as the host ``Bash``."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    result = env.Bash(command, "rewrite")
    assert result.startswith("REFUSED:"), result
    assert "Edit(" in result
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")


def test_docker_run_commands_parallel_refuses_shell_rewrites(env: DockerManager) -> None:
    """The fan-out refuses a rewrite per command and still runs the others."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    report = env.run_commands_parallel(json.dumps([_SED_REWRITE, "echo fine"]))
    assert "REFUSED:" in report, report
    assert "fine" in report, report
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")
