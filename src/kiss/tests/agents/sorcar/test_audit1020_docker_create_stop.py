# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""R4 review, issue 2: a Stop inside ``containers.create()`` must not orphan the container.

docker-py's ``create()`` is two daemon calls: ``api.create_container``
(the daemon makes the container) and a ``get`` of the new id that
returns the handle.  A Stop (``KeyboardInterrupt`` injected into the
task thread) landing between the two left ``DockerManager.container``
None while a container of this process sat in ``Created`` state;
``close()`` printed "No container to close." and nothing ever removed
it.  ``open()`` now names the container before creating it, so the
failure path (and a later ``close()``) removes it by that name when no
handle came back.

The tests run against the real daemon (skipped when none answers).
The interrupt is injected with ``sys.settrace`` on the thread running
``open()``: the trace function raises ``KeyboardInterrupt`` when
docker-py's ``ContainerCollection.get`` is entered from ``create``,
exactly the point a Stop would hit after the daemon made the container.
"""

from __future__ import annotations

import sys
import threading
from types import FrameType
from typing import Any

import docker
import docker.errors
import pytest
from docker.models.containers import ContainerCollection

from kiss.agents.sorcar.docker_manager import OWNER_LABEL, DockerManager, owner_label_value
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]

_GET_CODE = ContainerCollection.get.__code__
_CREATE_CODE = ContainerCollection.create.__code__


def _interrupt_before_get(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function: raise ``KeyboardInterrupt`` on entering ``get`` from ``create``."""
    if event == "call" and frame.f_code is _GET_CODE:
        caller = frame.f_back
        if caller is not None and caller.f_code is _CREATE_CODE:
            raise KeyboardInterrupt("stop injected after the daemon created the container")
    return None


def _open_with_stop_after_create(manager: DockerManager, errors: list[BaseException]) -> None:
    """Thread body: run ``open()`` with the interrupt armed, record what it raised."""
    sys.settrace(_interrupt_before_get)
    try:
        manager.open()
    except BaseException as exc:  # noqa: BLE001 — the thread records any outcome
        errors.append(exc)
    finally:
        sys.settrace(None)


def test_stop_between_create_and_handle_removes_the_created_container() -> None:
    """A Stop landing after the daemon made the container leaves no orphan behind."""
    client = docker.from_env()
    before = image_container_ids(client)
    manager = DockerManager(IMAGE, mount_shared_volume=True)
    errors: list[BaseException] = []
    try:
        worker = threading.Thread(
            target=_open_with_stop_after_create, args=(manager, errors), daemon=True,
        )
        worker.start()
        worker.join(120)
        assert not worker.is_alive(), "open() did not return"
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt), errors
        assert manager.container is None
        assert manager.container_name is None, "the removed container's name was kept"
        assert manager.host_shared_path is None, "the shared volume dir leaked"
        leaked = image_container_ids(client) - before
        assert not leaked, f"containers orphaned by the interrupted open(): {leaked}"
        manager.close()  # nothing to close, must not raise
    finally:
        manager.close()
        remove_new_image_containers(client, before)


def test_close_removes_a_named_container_whose_handle_never_arrived() -> None:
    """``close()`` with no handle still removes the container ``open()`` named.

    The case where the failure path of ``open()`` was itself interrupted
    (the daemon's stop watchdog injects a second interrupt into a
    cleanup that takes too long): the container exists under the name
    the manager chose, ``container`` is None, and ``close()`` is the
    last chance to remove it.
    """
    client = docker.from_env()
    before = image_container_ids(client)
    manager = DockerManager(IMAGE, mount_shared_volume=False)
    name = "kiss-test-" + owner_label_value().replace(":", "-")
    created = client.containers.create(
        IMAGE, command="/bin/bash", name=name, labels={OWNER_LABEL: owner_label_value()},
    )
    try:
        manager.container_name = name
        assert manager.container is None
        manager.close()
        assert manager.container_name is None
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(created.id or name)
        assert image_container_ids(client) - before == set()
        manager.close()  # idempotent: no name, no handle, nothing to do
    finally:
        manager.close()
        remove_new_image_containers(client, before)
