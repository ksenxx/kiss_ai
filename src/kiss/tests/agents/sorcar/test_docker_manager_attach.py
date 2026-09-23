# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``DockerManager("container:<id>")`` attaches to a caller-owned container.

Against a real Docker daemon (skipped when none runs): ``open()`` looks the
container up instead of starting one, commands run in the container's own
working directory (``/`` when the image declares none), no shared volume is
created, and ``close()`` drops the handle without stopping or removing the
container.  Attaching by name works like attaching by id.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import docker
import docker.errors
import docker.models.containers
import pytest

from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX, DockerManager

IMAGE = "python:3.11-slim"


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not running"),
]


@pytest.fixture
def containers() -> Iterator[
    tuple[docker.models.containers.Container, docker.models.containers.Container]
]:
    """A running container (no declared working directory) and one with ``/opt``."""
    client = docker.from_env()
    name = f"kiss-attach-{uuid.uuid4().hex[:8]}"
    plain = client.containers.run(IMAGE, command="sleep infinity", detach=True, name=name)
    with_workdir = client.containers.run(
        IMAGE, command="sleep infinity", detach=True, working_dir="/opt",
    )
    try:
        yield plain, with_workdir
    finally:
        for container in (plain, with_workdir):
            container.remove(force=True)


def test_attach_by_id_uses_root_when_image_declares_no_workdir(containers) -> None:
    plain, _ = containers
    manager = DockerManager(ATTACH_PREFIX + plain.id)
    assert manager.attached_container == plain.id
    with manager as mgr:
        assert mgr.container is not None and mgr.container.id == plain.id
        assert mgr.workdir == "/"
        assert mgr.host_shared_path is None
        assert "hello" in mgr.Bash("echo hello", "greet")
        assert mgr.Bash("pwd", "cwd").splitlines()[0] == "/"
    assert manager.container is None
    plain.reload()
    assert plain.status == "running"


def test_attach_by_name_uses_the_containers_workdir(containers) -> None:
    plain, with_workdir = containers
    with DockerManager(f"{ATTACH_PREFIX}{plain.name}") as by_name:
        assert by_name.container is not None and by_name.container.id == plain.id
    with DockerManager(ATTACH_PREFIX + with_workdir.id) as mgr:
        assert mgr.workdir == "/opt"
        assert mgr.Bash("pwd", "cwd").splitlines()[0] == "/opt"
    with_workdir.reload()
    assert with_workdir.status == "running"


def test_attach_to_unknown_container_fails_loudly() -> None:
    manager = DockerManager(ATTACH_PREFIX + "no-such-container-" + uuid.uuid4().hex)
    with pytest.raises(docker.errors.NotFound):
        manager.open()
    assert manager.container is None
    manager.close()  # nothing to close, nothing to remove


def test_streaming_bash_timeout_returns_output_so_far(containers) -> None:
    """A timed-out streamed command reports the output it produced before the deadline."""
    plain, _ = containers
    streamed: list[str] = []
    with DockerManager(ATTACH_PREFIX + plain.id) as mgr:
        mgr.stream_callback = streamed.append
        result = mgr.Bash("echo started; sleep 5; echo never", "slow", timeout_seconds=1)
        silent = mgr.Bash("sleep 5", "silent", timeout_seconds=1)
    assert result.startswith(
        "Error: command timed out after 1s and was killed. Output before the timeout:"
    )
    assert "started" in result and "never" not in result
    assert silent == "Error: command timed out after 1s"
    assert "started\n" in "".join(streamed)
