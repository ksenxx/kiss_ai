# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20 (B5/F2): ``DockerTools._store`` after a Stop.

A ``Write``/``Edit`` piece that is killed by a task Stop returns the
``Killed: the task was stopped.`` result.  ``_store`` then ran one more
``docker exec`` to discard its staging file, but ``DockerManager._exec``
returns before its first slice while the stop event is set, so the
``rm`` never ran in the container; the discard only cost an extra exec
worker and a bounded wait on its cancellation before the Stop could
propagate.  The discard is now skipped when the piece reports the Stop
and still runs when a piece fails for any other reason.
"""

from __future__ import annotations

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


@pytest.fixture
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_stopped_store_skips_the_discard(
    env: DockerManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A piece killed by the Stop is reported as is; no discard exec follows."""
    tools = DockerTools(env.Bash)
    env.stop_event = threading.Event()
    env.stop_event.set()
    result = tools.Write("/tmp/stopped/target.txt", "never lands\n")
    assert STOPPED_MESSAGE in result, result
    assert "(discard)" not in capsys.readouterr().out


def test_failed_store_still_discards(
    env: DockerManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A piece that fails for another reason removes the staging file."""
    tools = DockerTools(env.Bash)
    env.Bash("mkdir -p /tmp/onto/adir", "mkdir")
    result = tools.Write("/tmp/onto/adir", "C" * 200_000)
    assert "directory" in result, result
    assert "(discard)" in capsys.readouterr().out
    assert env.Bash("ls -A /tmp/onto", "ls").split("\n")[0] == "adir"
