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
container tools now apply the same ``rewrite_guard``.  Since 2026-10-31
a rewrite that spells out one exact ``Write``/``Edit`` (a literal
here-document copy, a python ``.replace``, a one-substitution ``sed -i``)
is raised by ``Bash`` as that call instead
(:class:`~kiss.core.tool_verdict.ToolRedirect`, which the agent makes
through its own tool path: hook, guard and events see it as if the
model had issued it; ``Bash`` itself touches nothing), while
``run_commands_parallel``, which cannot redirect, still refuses it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.core.tool_verdict import ALLOW, ToolRedirect, refuse
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import _Harness

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
# A cat here-document over the existing file after a ``cd`` (``/`` is the
# default workdir): the guard asks the container whether ``/`` is a directory.
_CAT_HEREDOC_REWRITE = "cd / && cat > tmp/some_file.txt <<'EOF'\nb\nEOF"
# The exact call each rewrite of the seeded ``a\n`` stands for.
_REWRITE_CALLS = {
    _SED_REWRITE: (
        "Edit",
        {"file_path": "/tmp/some_file.txt", "old_string": "a", "new_string": "b\nc",
         "replace_all": False},
    ),
    _HEREDOC_REWRITE: (
        "Edit",
        {"file_path": "/tmp/some_file.txt", "old_string": "a", "new_string": "b",
         "replace_all": True},
    ),
    _CAT_HEREDOC_REWRITE: ("Write", {"file_path": "/tmp/some_file.txt", "content": "b\n"}),
}


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


@pytest.mark.parametrize("command", [_SED_REWRITE, _HEREDOC_REWRITE, _CAT_HEREDOC_REWRITE])
def test_docker_bash_makes_shell_rewrites_as_the_edit_or_write(
    env: DockerManager, command: str,
) -> None:
    """The container ``Bash`` raises the exact ``Edit``/``Write`` a rewrite stands for."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    with pytest.raises(ToolRedirect) as info:
        env.Bash(command, "rewrite")
    name, arguments = _REWRITE_CALLS[command]
    assert (info.value.name, info.value.arguments) == (name, arguments)
    assert info.value.followup is None
    assert info.value.note.startswith(f"Applied as {name}("), info.value.note
    # The call is the agent's to make: Bash itself left the file alone.
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")


@pytest.mark.parametrize(
    ("command", "name", "args", "content"),
    [
        (
            _SED_REWRITE, "Edit",
            {"old_string": "a", "new_string": "b\nc", "replace_all": False}, b"b\nc\n",
        ),
        (
            _HEREDOC_REWRITE, "Edit",
            {"old_string": "a", "new_string": "b", "replace_all": True}, b"b\n",
        ),
        (_CAT_HEREDOC_REWRITE, "Write", {"content": "b\n"}, b"b\n"),
    ],
)
def test_docker_bash_makes_shell_rewrites_as_edit_or_write(
    env: DockerManager, command: str, name: str, args: dict[str, Any], content: bytes,
) -> None:
    """The container ``Bash`` makes the Edit/Write a rewrite stands for through the agent."""
    h = _Harness(env)
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    expected = (name, {"file_path": "/tmp/some_file.txt", **args})
    # The hook sees the redirected call and may refuse it: nothing changes then.
    h.verdict = refuse("not yet")
    result = h.call("Bash", command=command, description="rewrite")
    assert result.startswith(f"Applied as {name}(") and result.endswith(
        f"REFUSED: the {name} call did NOT run and had no effect. not yet"
    ), result
    assert h.hook_calls[-1] == expected and h.tool_calls() == ["Bash", name]
    assert env.read_bytes("/tmp/some_file.txt") == b"a\n"
    # Allowed, the call lands and the rewrite's result opens with what it was applied as.
    h.verdict = ALLOW
    result = h.call("Bash", command=command, description="rewrite")
    assert result.startswith(f"Applied as {name}(") and "Successfully" in result, result
    assert h.hook_calls[-1] == expected and h.tool_calls() == ["Bash", name, "Bash", name]
    assert env.read_bytes("/tmp/some_file.txt") == content


def test_docker_run_commands_parallel_refuses_shell_rewrites(env: DockerManager) -> None:
    """The fan-out refuses a rewrite per command and still runs the others."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    report = env.run_commands_parallel(
        json.dumps([_SED_REWRITE, _CAT_HEREDOC_REWRITE, "echo fine"]))
    assert report.count("REFUSED:") == 2, report
    assert "fine" in report, report
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")


def test_docker_bash_heredoc_into_a_new_file_is_a_write_and_an_append_reads_it(
    env: DockerManager,
) -> None:
    """A here-document into a file the container lacks is its ``Write``; an append adds to it."""
    env.Bash("rm -f /tmp/heredoc_new.txt", "clean")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat > /tmp/heredoc_new.txt <<'EOF'\nnew\nEOF", "create")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/heredoc_new.txt", "content": "new\n"}
    assert env.Bash("test -e /tmp/heredoc_new.txt; echo $?", "check").startswith("1")
    env.Bash("printf 'new\\n' > /tmp/heredoc_new.txt", "seed")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat >> /tmp/heredoc_new.txt <<'EOF'\nmore\nEOF", "append")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/heredoc_new.txt", "content": "new\nmore\n"}
    assert info.value.note.startswith("Appended through Write("), info.value.note
    content = env.Bash("cat /tmp/heredoc_new.txt", "cat")
    assert content.startswith("new\n") and "more" not in content, content


def test_docker_bash_heredoc_may_create_a_new_file(env: DockerManager) -> None:
    """A here-document into a file the container lacks creates it; an append goes after it."""
    h = _Harness(env)
    env.Bash("rm -f /tmp/heredoc_new.txt", "clean")
    create = "cat > /tmp/heredoc_new.txt <<'EOF'\nnew\nEOF"
    created = h.call("Bash", command=create, description="create")
    assert created == (
        'Applied as Write(file_path="/tmp/heredoc_new.txt", content="new\\n"), whose result '
        "follows:\nSuccessfully wrote 4 characters to /tmp/heredoc_new.txt"
    ), created
    append = "cat >> /tmp/heredoc_new.txt <<'EOF'\nmore\nEOF"
    again = h.call("Bash", command=append, description="append")
    assert again.startswith(
        'Appended through Write(file_path="/tmp/heredoc_new.txt", content="new\\nmore\\n")'
    ), again
    assert h.hook_calls[-1] == (
        "Write", {"file_path": "/tmp/heredoc_new.txt", "content": "new\nmore\n"},
    )
    assert env.read_bytes("/tmp/heredoc_new.txt") == b"new\nmore\n"
