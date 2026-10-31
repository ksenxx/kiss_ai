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
  tests below pin the probe's three outcomes instead: the target is
  absent (the copy runs), present (refused), or the probe was stopped
  (refused, nothing runs).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available

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


def test_a_heredoc_creates_a_new_file(env: DockerManager) -> None:
    """The probe answers "absent": the copy runs."""
    env.Bash("rm -f /tmp/probe_new.txt", "clean")
    result = env.Bash("cat > /tmp/probe_new.txt <<'EOF'\nfresh\nEOF", "create")
    assert not result.startswith("REFUSED:"), result
    assert env.Bash("cat /tmp/probe_new.txt", "cat").startswith("fresh")


def test_a_heredoc_over_an_existing_file_is_refused(env: DockerManager) -> None:
    """The probe answers "present": refused, the file keeps its content."""
    env.Bash("printf 'kept\\n' > /tmp/probe_old.txt", "seed")
    result = env.Bash("cat > /tmp/probe_old.txt <<'EOF'\nclobbered\nEOF", "rewrite")
    assert result.startswith("REFUSED:") and "/tmp/probe_old.txt" in result, result
    assert env.Bash("cat /tmp/probe_old.txt", "cat").startswith("kept")


def test_a_stopped_probe_refuses_and_runs_nothing(env: DockerManager) -> None:
    """A Stop ends the probe unanswered: the target counts as existing, nothing runs."""
    env.Bash("rm -f /tmp/probe_stopped.txt", "clean")
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        result = env.Bash("cat > /tmp/probe_stopped.txt <<'EOF'\nx\nEOF", "create")
    finally:
        env.stop_event = None
    assert result.startswith("REFUSED:"), result
    assert env.Bash("test -e /tmp/probe_stopped.txt; echo $?", "check").startswith("1")


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
