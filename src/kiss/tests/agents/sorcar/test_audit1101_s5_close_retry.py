# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F6): a close whose removal failed does not forget a running container.

``DockerManager._close_locked`` caught a failed ``stop()``, then a
failed plain ``remove()`` (the daemon refuses to remove a running
container), and still cleared the handle and the name, deleted the
shared directory and printed ``Container closed successfully``: the
container ran on, owned by nobody, and a second ``close()`` had nothing
to act on.  Ownership is now cleared only once the container is gone:
the removal is the force remove the failure path of ``open()`` uses
(``_discard_container``), which keeps the name on the manager when it
fails, so a later ``close()`` retries by name.

Audit 2026-11-07 (A08 claude-1, claude-3, claude-9): ``close()`` first
asked for a graceful ``stop()``, which the container's PID 1 (an
interactive bash) ignores, so every close waited the daemon's 10 s
grace; and a Stop injected into the closing thread (the runner's
watchdog, a ``KeyboardInterrupt``) between the daemon calls left the
container and the shared directory behind for good.  The close is one
force remove now, bounded well under the grace, and an interrupt that
lands inside a daemon call still removes the container and the
directory before it propagates.  What a command made under the shared
volume is root-owned on the host, where ``rmtree`` cannot delete it:
the container empties the volume before it goes.

The container and the manager are real; the one injection is a
``remove()`` of the docker SDK handle raising once (the interrupt is
raised by a trace function on entering ``remove``), since a real daemon
cannot be asked to fail one RPC.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from types import FrameType
from typing import Any

import docker
import docker.errors
import pytest
from docker.models.containers import Container

from kiss.agents.sorcar.docker_manager import DockerManager
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


@pytest.fixture
def cleanup() -> Iterator[None]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield
    finally:
        remove_new_image_containers(client, before)


def test_a_close_whose_removal_failed_keeps_the_container_for_the_next_close(
    cleanup: None, capsys: pytest.CaptureFixture[str],
) -> None:
    """The name and the shared directory stay tracked; the next close removes both."""
    manager = DockerManager(IMAGE)
    manager.open()
    owned = manager.container
    assert owned is not None and manager.host_shared_path is not None
    shared = manager.host_shared_path
    name = manager.container_name
    real_remove = owned.remove

    def failed_remove(*args: Any, **kwargs: Any) -> None:
        raise OSError("one transient remove RPC failure")

    owned.remove = failed_remove  # type: ignore[method-assign]
    try:
        capsys.readouterr()
        manager.close()
    finally:
        owned.remove = real_remove  # type: ignore[method-assign]
    out = capsys.readouterr().out
    assert "Failed to remove container" in out and "closed successfully" not in out, out
    owned.reload()  # still there, still owned by name
    assert manager.container is None and manager.container_name == name
    assert manager.host_shared_path == shared and os.path.isdir(shared)
    manager.close()
    out = capsys.readouterr().out
    with pytest.raises(docker.errors.NotFound):
        owned.reload()
    assert manager.container is None and manager.container_name is None
    assert manager.host_shared_path is None and not os.path.exists(shared)
    assert "No container to close." in out, out


def test_close_is_one_force_remove_and_empties_the_root_owned_volume(
    cleanup: None, capsys: pytest.CaptureFixture[str],
) -> None:
    """No 10 s stop grace, and what the container made under the volume is gone too."""
    manager = DockerManager(IMAGE)
    manager.open()
    owned = manager.container
    assert owned is not None and manager.host_shared_path is not None
    shared = manager.host_shared_path
    result = manager.Bash("id -u && mkdir -p /testbed/sub && echo x > /testbed/sub/f", "root")
    assert result.strip() == "0", result
    assert os.path.exists(os.path.join(shared, "sub", "f"))
    capsys.readouterr()
    started = time.monotonic()
    manager.close()
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    assert elapsed < 5.0, f"close() took {elapsed:.1f}s (the stop grace is 10 s)"
    with pytest.raises(docker.errors.NotFound):
        owned.reload()
    assert manager.container is None and manager.container_name is None
    assert manager.host_shared_path is None and not os.path.exists(shared), out
    assert "Failed to clean up" not in out, out
    assert out.rstrip().endswith("Container closed successfully"), out


_REMOVE_CODE = Container.remove.__code__


def _interrupt_once_in_remove(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function: raise ``KeyboardInterrupt`` on the first entry into ``Container.remove``."""
    if event == "call" and frame.f_code is _REMOVE_CODE:
        sys.settrace(None)
        raise KeyboardInterrupt("stop injected inside the daemon call")
    return None


def _close_with_stop_in_remove(manager: DockerManager, errors: list[BaseException]) -> None:
    """Thread body: run ``close()`` with the interrupt armed, record what it raised."""
    sys.settrace(_interrupt_once_in_remove)
    try:
        manager.close()
    except BaseException as exc:  # noqa: BLE001 — the thread records any outcome
        errors.append(exc)
    finally:
        sys.settrace(None)


def test_a_stop_landing_inside_the_removal_still_removes_everything(cleanup: None) -> None:
    """The interrupt propagates, and neither the container nor the directory is left behind."""
    manager = DockerManager(IMAGE)
    manager.open()
    owned = manager.container
    assert owned is not None and manager.host_shared_path is not None
    shared = manager.host_shared_path
    errors: list[BaseException] = []
    closer = threading.Thread(
        target=_close_with_stop_in_remove, args=(manager, errors), daemon=True,
    )
    closer.start()
    closer.join(60)
    assert not closer.is_alive(), "close() did not return"
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt), errors
    with pytest.raises(docker.errors.NotFound):
        owned.reload()
    assert manager.container is None and manager.container_name is None
    assert manager.host_shared_path is None and not os.path.exists(shared)
