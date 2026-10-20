# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``DockerTools`` stores a payload in one piece, whatever happens around it.

A payload above the argv limit travels in several ``docker exec``
commands.  Landing them straight on the destination let a second
writer truncate the file between two pieces: both writes reported
success and the file held ten bytes of one followed by the tail of
the other.  A piece that failed (no space left, the destination is a
directory) left the partial file behind; for an ``Edit`` that was its
staging payload under ``/tmp``.

The payload is now staged beside the destination under a unique name
and published with one ``mv`` once every piece landed, so a reader
sees either the previous file or the whole new one, and the staging
file is removed on every exit path that did not publish it.  All
tests run against real containers; the race is pinned with the
settrace ``Pause`` on ``DockerManager.Bash``.
"""

import hashlib
import threading
from collections.abc import Iterator
from typing import Any

import docker
import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.core.kiss_error import KISSError
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.server.pause import Pause

IMAGE = "python:3.11-slim"

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


@pytest.fixture
def small_tmp_env() -> Iterator[DockerManager]:
    """A manager attached to a container whose ``/tmp`` is a 128 KiB tmpfs."""
    client = docker.from_env()
    container = client.containers.run(
        IMAGE, command="sleep infinity", detach=True, tmpfs={"/tmp": "size=131072"},
    )
    try:
        manager = DockerManager(f"container:{container.id}")
        manager.open()
        try:
            yield manager
        finally:
            manager.close()
    finally:
        container.remove(force=True)
        client.close()


def _sha256_in(env: DockerManager, path: str) -> str:
    """The SHA-256 the container computes for *path*."""
    return env.Bash(f"sha256sum {path}", "hash").split()[0]


def _listing(env: DockerManager, directory: str) -> list[str]:
    """The names in *directory*, dotfiles included, without the exit-code line."""
    output = env.Bash(f"ls -A {directory}", "ls")
    return [line for line in output.splitlines() if not line.startswith("[exit code:")]


def _is_append_piece(frame_locals: dict[str, Any]) -> bool:
    """Whether a ``DockerManager.Bash`` call carries a second or later payload piece."""
    return "base64 -d >>" in frame_locals["command"]


def test_concurrent_writes_never_mix_and_readers_never_see_a_partial_file(
    env: DockerManager,
) -> None:
    """Two writers of one path leave the file one of the two contents, never a mixture."""
    tools = DockerTools(env.Bash)
    path = "/tmp/race/target.txt"
    assert "Successfully wrote" in tools.Write(path, "previous\n")
    big = "A" * 80_000
    results: list[str] = []

    def write_big() -> None:
        results.append(tools.Write(path, big))

    writer = threading.Thread(target=write_big)
    with Pause(DockerManager.Bash.__code__, _is_append_piece) as pause:
        writer.start()
        assert pause.paused.wait(60), "the big write never reached its second piece"
        # The first piece has landed somewhere, but a reader still sees
        # the previous file whole.
        assert env.Bash(f"cat {path}", "cat").split("\n")[0] == "previous"
        results.append(tools.Write(path, "B" * 10))
        pause.release()
    writer.join(60)
    assert not writer.is_alive()
    assert all("Successfully wrote" in result for result in results), results
    assert _sha256_in(env, path) == hashlib.sha256(big.encode()).hexdigest()
    assert _listing(env, "/tmp/race") == ["target.txt"]


def test_failed_edit_leaves_no_staging_file(small_tmp_env: DockerManager) -> None:
    """An Edit whose payload does not fit the disk reports it and leaves nothing behind."""
    tools = DockerTools(small_tmp_env.Bash)
    assert "Successfully wrote" in tools.Write("/tmp/target.txt", "old\n")
    result = tools.Edit("/tmp/target.txt", "old", "X" * 200_000)
    assert "No space left on device" in result, result
    assert small_tmp_env.Bash("cat /tmp/target.txt", "cat").split("\n")[0] == "old"
    assert _listing(small_tmp_env, "/tmp") == ["target.txt"]


def test_write_onto_a_directory_fails_and_leaves_nothing(env: DockerManager) -> None:
    """A multi-piece Write whose destination is a directory fails without a leftover."""
    tools = DockerTools(env.Bash)
    env.Bash("mkdir -p /tmp/onto/adir", "mkdir")
    result = tools.Write("/tmp/onto/adir", "C" * 200_000)
    assert "Successfully wrote" not in result
    assert "directory" in result, result
    assert _listing(env, "/tmp/onto") == ["adir"]


def test_write_keeps_the_mode_of_the_replaced_file(env: DockerManager) -> None:
    """Overwriting an executable keeps it executable."""
    tools = DockerTools(env.Bash)
    tools.Write("/tmp/mode/run.sh", "#!/bin/sh\necho one\n")
    env.Bash("chmod 755 /tmp/mode/run.sh", "chmod")
    assert "Successfully wrote" in tools.Write("/tmp/mode/run.sh", "#!/bin/sh\necho two\n")
    assert env.Bash("stat -c %a /tmp/mode/run.sh", "stat").split("\n")[0] == "755"
    assert env.Bash("/tmp/mode/run.sh", "run").split("\n")[0] == "two"


def test_manager_closed_between_pieces_raises_and_publishes_nothing(
    env: DockerManager,
) -> None:
    """A store cut short by the container going away raises instead of publishing."""
    tools = DockerTools(env.Bash)
    errors: list[BaseException] = []

    def write_big() -> None:
        try:
            tools.Write("/tmp/cut/target.txt", "D" * 80_000)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer = threading.Thread(target=write_big)
    with Pause(DockerManager.Bash.__code__, _is_append_piece) as pause:
        writer.start()
        assert pause.paused.wait(60)
        env.close()
        pause.release()
    writer.join(60)
    assert not writer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KISSError), errors
