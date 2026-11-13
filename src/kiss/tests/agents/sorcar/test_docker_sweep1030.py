# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): the container Edit batch and the here-document probe, on real Docker.

* ``DockerTools.Edit`` with ``edits=`` judged a failed edit by an
  ``Error`` prefix, but a failure can print a Python traceback (a
  non-UTF-8 or unwritable file) or a failed payload store, neither of
  which starts with ``Error``.  Such an edit was reported as
  ``edit 1: Traceback ...`` and the batch went on.  Only the
  ``Successfully replaced`` line counts as success now, so the batch
  stops at the failing edit with ``Error: edit k: ...``.
* The here-document guard of ``DockerManager.Bash`` and
  ``run_commands_parallel`` asked the container whether the target
  exists with ``container.exec_run`` on the calling thread, outside the
  ``_exec`` machinery: neither the command's deadline nor a Stop bounded
  it, and a docker-py read timeout escaped ``Bash`` as an exception.  The
  probe is now an ``_exec`` like any command.  The original hang needs
  a daemon that stalls on one request, which cannot be arranged against
  a real daemon without a fault-injecting proxy (a test double), so the
  tests below pin the probe's outcomes instead.  Since 2026-10-31 a
  literal here-document copy is not refused but raised as the ``Write``
  it stands for (:class:`~kiss.core.tool_verdict.ToolRedirect`, made by
  the agent through its hook and events; ``Bash`` itself touches
  nothing), whether the target is absent or present: a copy into an
  absent file creates it, one over an existing file is the hook's to
  refuse, and a stopped task's redirected ``Write`` returns the stopped
  result and writes nothing.  The probe now serves an append (``>>``),
  whose current text is read from the container, and
  ``run_commands_parallel``, which cannot redirect.  A stopped probe
  counts as "present" and the stopped read makes the append an error:
  nothing runs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

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


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def _listed_execs(env: DockerManager) -> set[str]:
    """The execs the daemon lists for the container: created or running, dropped at their exit."""
    assert env.container is not None and env.container.id is not None
    return set(env.client.api.inspect_container(env.container.id)["ExecIDs"] or ())


def test_a_traceback_failure_stops_the_edit_batch(env: DockerManager) -> None:
    """A non-UTF-8 file makes edit 1 raise; the batch reports it and edit 2 never runs."""
    env.Bash("printf 'one\\377\\ntwo\\n' > /tmp/latin1.txt", "seed")
    result = DockerTools(env.Bash).Edit(
        "/tmp/latin1.txt",
        edits=json.dumps([
            {"old_string": "one", "new_string": "uno"},
            {"old_string": "two", "new_string": "dos"},
        ]),
    )
    assert result.startswith("Error: edit 1: "), result
    assert "UnicodeDecodeError" in result, result
    assert "edit 2" not in result, result
    assert env.Bash("cat -v /tmp/latin1.txt", "dump").startswith("oneM-^?\ntwo\n")


def test_a_heredoc_into_a_new_file_is_made_as_a_write(env: DockerManager) -> None:
    """A copy into an absent target is the ``Write`` of its body; ``Bash`` creates nothing."""
    env.Bash("rm -f /tmp/probe_new.txt", "clean")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat > /tmp/probe_new.txt <<'EOF'\nfresh\nEOF", "create")
    redirect = info.value
    assert redirect.name == "Write"
    assert redirect.arguments == {"file_path": "/tmp/probe_new.txt", "content": "fresh\n"}
    assert redirect.followup is None
    assert redirect.note.startswith('Applied as Write(file_path="/tmp/probe_new.txt"'), (
        redirect.note
    )
    assert env.Bash("test -e /tmp/probe_new.txt; echo $?", "check").startswith("1")


def test_a_heredoc_over_an_existing_file_is_made_as_a_write(env: DockerManager) -> None:
    """A copy over a present target is the ``Write`` of its body; an append reads the text."""
    env.Bash("printf 'kept\\n' > /tmp/probe_old.txt", "seed")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat > /tmp/probe_old.txt <<'EOF'\nclobbered\nEOF", "rewrite")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/probe_old.txt", "content": "clobbered\n"}
    assert env.Bash("cat /tmp/probe_old.txt", "cat").startswith("kept\n")
    # The probe answers "present" and the current text is fetched from the container.
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat >> /tmp/probe_old.txt <<'EOF'\nmore\nEOF", "append")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/probe_old.txt", "content": "kept\nmore\n"}
    assert info.value.note.startswith("Appended through Write("), info.value.note
    assert env.Bash("cat /tmp/probe_old.txt", "cat").startswith("kept\n")


def test_a_stopped_probe_refuses_and_runs_nothing(env: DockerManager) -> None:
    """A Stop ends the probe unanswered: the target counts as present, the append is an error."""
    env.Bash("rm -f /tmp/probe_stopped.txt", "clean")
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        result = env.Bash("cat >> /tmp/probe_stopped.txt <<'EOF'\nx\nEOF", "append")
        # A plain copy needs no probe: it is the Write the (stopped) agent would make.
        with pytest.raises(ToolRedirect) as info:
            env.Bash("cat > /tmp/probe_stopped.txt <<'EOF'\nx\nEOF", "create")
    finally:
        env.stop_event = None
    assert result.startswith("Error: cannot append to /tmp/probe_stopped.txt: "), result
    assert info.value.arguments == {"file_path": "/tmp/probe_stopped.txt", "content": "x\n"}
    assert env.Bash("test -e /tmp/probe_stopped.txt; echo $?", "check").startswith("1")


def test_a_heredoc_creates_a_new_file(env: DockerManager) -> None:
    """A copy into an absent file is made as the Write it stands for, through the agent."""
    h = _Harness(env)
    env.Bash("rm -f /tmp/probe_new.txt", "clean")
    command = "cat > /tmp/probe_new.txt <<'EOF'\nfresh\nEOF"
    result = h.call("Bash", command=command, description="create")
    assert result == (
        'Applied as Write(file_path="/tmp/probe_new.txt", content="fresh\\n"), whose result '
        "follows:\nSuccessfully wrote 6 characters to /tmp/probe_new.txt"
    ), result
    assert h.hook_calls == [
        ("Bash", {"command": command, "description": "create"}),
        ("Write", {"file_path": "/tmp/probe_new.txt", "content": "fresh\n"}),
    ]
    assert h.tool_calls() == ["Bash", "Write"]
    assert env.read_bytes("/tmp/probe_new.txt") == b"fresh\n"


def test_a_heredoc_over_an_existing_file_is_the_hooks_to_refuse(env: DockerManager) -> None:
    """A copy over an existing file reaches the hook as the Write; its refusal keeps the content."""
    h = _Harness(env)
    env.Bash("printf 'kept\\n' > /tmp/probe_old.txt", "seed")
    command = "cat > /tmp/probe_old.txt <<'EOF'\nclobbered\nEOF"
    h.verdict = refuse("keep it")
    result = h.call("Bash", command=command, description="rewrite")
    assert result == (
        'Applied as Write(file_path="/tmp/probe_old.txt", content="clobbered\\n"), whose result '
        "follows:\nREFUSED: the Write call did NOT run and had no effect. keep it"
    ), result
    assert h.hook_calls[-1] == (
        "Write", {"file_path": "/tmp/probe_old.txt", "content": "clobbered\n"},
    )
    assert env.read_bytes("/tmp/probe_old.txt") == b"kept\n"
    # Allowed, the same copy replaces the content as the shell would have.
    h.verdict = ALLOW
    result = h.call("Bash", command=command, description="rewrite")
    assert result.endswith("Successfully wrote 10 characters to /tmp/probe_old.txt"), result
    assert env.read_bytes("/tmp/probe_old.txt") == b"clobbered\n"


def test_a_stopped_task_writes_nothing(env: DockerManager) -> None:
    """With the Stop set, the redirected Write starts nothing: stopped result, no exec, no file."""
    h = _Harness(env)
    env.Bash("rm -f /tmp/probe_stopped.txt", "clean")
    execs = _listed_execs(env)
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        result = h.call(
            "Bash", command="cat > /tmp/probe_stopped.txt <<'EOF'\nx\nEOF", description="create",
        )
    finally:
        env.stop_event = None
    assert result == (
        'Applied as Write(file_path="/tmp/probe_stopped.txt", content="x\\n"), whose result '
        f"follows:\n{STOPPED_MESSAGE}\n[exit code: -1]"
    ), result
    assert h.tool_calls() == ["Bash", "Write"]
    assert env.Bash("test -e /tmp/probe_stopped.txt; echo $?", "check").startswith("1")
    # An exec made for the stopped Write would still be listed: whether it then
    # started was a race between its worker and the cancellation.
    assert not _listed_execs(env) - execs


def test_run_commands_parallel_probes_each_heredoc(env: DockerManager) -> None:
    """The fan-out's probe runs per command: the new file is created, the old one kept."""
    env.Bash("rm -f /tmp/fan_new.txt; printf 'kept\\n' > /tmp/fan_old.txt", "seed")
    report = env.run_commands_parallel(json.dumps([
        "cat > /tmp/fan_new.txt <<'EOF'\nfresh\nEOF",
        "cat > /tmp/fan_old.txt <<'EOF'\nclobbered\nEOF",
    ]))
    assert report.count("REFUSED:") == 1, report
    assert env.Bash("cat /tmp/fan_new.txt", "cat").startswith("fresh")
    assert env.Bash("cat /tmp/fan_old.txt", "cat").startswith("kept")
