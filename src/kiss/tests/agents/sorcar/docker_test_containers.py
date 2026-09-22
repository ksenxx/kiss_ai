# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Container bookkeeping for Docker tests that share the daemon with other work.

The Docker daemon on a developer or CI machine is not exclusive to the test
run: benchmarks and other tasks start and remove their own containers at
the same time.  A test that snapshots ``client.containers.list(all=True)``
before and after its own ``DockerManager.open()`` therefore sees foreign
containers in the difference, and a teardown that force-removes "anything
new" kills work it does not own.  ``containers.list()`` also inspects each
listed container, so a foreign container removed between the two calls
raises ``docker.errors.NotFound`` in the middle of the test's teardown.

Every helper here restricts itself to containers created from
:data:`IMAGE`, the image all Sorcar Docker tests use, and lists them
sparsely (no per-container inspect).  The image is the only ownership
marker ``DockerManager`` leaves on a container, so the Docker tests still
share it among themselves: run them in ONE pytest process (never in
concurrent splits), or one test's teardown can remove another's container.
"""

from __future__ import annotations

import docker

IMAGE = "python:3.11-slim"


def image_container_ids(client: docker.DockerClient) -> set[str]:
    """Return the ids of every container, running or not, created from :data:`IMAGE`.

    Args:
        client: The Docker client to query.

    Returns:
        The container ids, filtered by ``ancestor`` so containers of other
        workloads on the same daemon are never counted.
    """
    containers = client.containers.list(all=True, sparse=True, filters={"ancestor": IMAGE})
    return {container.id for container in containers if container.id}


def remove_new_image_containers(client: docker.DockerClient, before: set[str]) -> None:
    """Force-remove every :data:`IMAGE` container that is not in ``before``.

    Args:
        client: The Docker client to act through.
        before: The ids returned by :func:`image_container_ids` before the
            test started; those containers are left alone.
    """
    for container_id in image_container_ids(client) - before:
        try:
            client.api.remove_container(container_id, force=True)
        except Exception:
            pass
