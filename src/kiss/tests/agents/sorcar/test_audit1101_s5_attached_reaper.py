# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F3): closing an attached manager does not disarm its reaper.

A timed-out command whose ``exec_start`` the daemon delayed past the
deadline is handed to a reaper that kills it once it starts.  The
reaper watched ``manager.container`` to know whether the container
still lived, and ``_kill_exec`` swept ``manager.container`` too; but
``close()`` of a manager attached to a container (``container:<id>``)
nulls that attribute while the container, owned by someone else, lives
on.  The delayed command then started after the close and ran to its
end in the surviving container.  The reaper and its sweeps now work on
the container the exec was created in, and stop when that exec or
container is gone or the daemon refused the start.

The test runs two real managers on one real container; the only
injection is the latency of one ``exec_start`` acknowledgement on the
docker SDK client (the real call is made once the gate opens), which
a real daemon cannot be asked to delay.  ``DockerManager`` itself is
not patched.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import docker
import pytest

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)
from kiss.tests.conftest import wait_for

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]


@pytest.fixture
def cleanup() -> Iterator[None]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield
    finally:
        remove_new_image_containers(client, before)


def _no_reaper_thread() -> bool:
    """Whether no exec reaper thread is alive (one ends once the exec it watches is dead)."""
    return not any(t.name.startswith("docker-exec-reaper-") for t in threading.enumerate())


def test_a_closed_attached_manager_still_reaps_its_delayed_exec(cleanup: None) -> None:
    """The exec a timed-out ``Bash`` left starting is killed after the attached manager closed."""
    owner = DockerManager(IMAGE, workdir="/tmp", mount_shared_volume=False)
    owner.open()
    assert owner.container is not None
    attached = DockerManager(f"container:{owner.container.id}")
    attached.open()
    api = attached.client.api
    real_start = api.exec_start
    entered, release = threading.Event(), threading.Event()
    exec_ids: list[str] = []

    def delayed_start(exec_id: str, *args: Any, **kwargs: Any) -> Any:
        """The daemon acknowledges the command's own start only once *release* is set."""
        if not entered.is_set():
            exec_ids.append(exec_id)
            entered.set()
            assert release.wait(30)
        return real_start(exec_id, *args, **kwargs)

    api.exec_start = delayed_start
    try:
        began = time.monotonic()
        out = attached.Bash("sleep 60", "delayed start", timeout_seconds=1)
        assert "timed out" in out.lower(), out
        assert time.monotonic() - began < 1 + _KILL_WAIT_S + 2, "the deadline was not kept"
        assert entered.is_set() and len(exec_ids) == 1
        attached.close()
        assert attached.container is None
        # The daemon starts the command now, in the container that survived the close.
        release.set()

        def killed() -> bool:
            info = owner.client.api.exec_inspect(exec_ids[0])
            return bool(info.get("Pid")) and not info.get("Running")

        assert wait_for(killed, timeout=15), "the delayed command ran on after the close"
        assert wait_for(_no_reaper_thread, timeout=10), "the reaper did not end with its exec"
    finally:
        release.set()
        api.exec_start = real_start
        attached.close()
        owner.close()
