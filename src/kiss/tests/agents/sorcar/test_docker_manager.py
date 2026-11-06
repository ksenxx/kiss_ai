# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test suite for DockerManager without mocking."""

import time
import unittest
from collections.abc import Iterator

import docker
import docker.errors
import pytest
import requests

from kiss.agents.sorcar.docker_manager import OWNER_LABEL, DockerManager, owner_label_value
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManagerStreaming(unittest.TestCase):

    def test_streaming_error_exit_code(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            result = env.Bash("echo before_fail && false", "Error stream test")
        assert "[exit code:" in result

    def test_streaming_stderr(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            env.Bash("echo stdout_msg && echo stderr_msg >&2", "Stderr stream")
        joined = "".join(streamed)
        assert "stdout_msg" in joined
        assert "stderr_msg" in joined


@pytest.fixture
def registry_repository() -> Iterator[str]:
    """A repository of :data:`IMAGE` in a private registry on a random host port.

    The registry runs in a container of its own; the image is pushed to
    it and then untagged locally, so every ``open()`` of the yielded
    ``localhost:<port>/kiss-registry-test`` has to pull it.
    """
    client = docker.from_env()
    before = image_container_ids(client)
    registry = client.containers.run(
        "registry:2", detach=True, ports={"5000/tcp": None},
        labels={OWNER_LABEL: owner_label_value()},
    )
    repository = ""
    try:
        registry.reload()
        port = registry.attrs["NetworkSettings"]["Ports"]["5000/tcp"][0]["HostPort"]
        deadline = time.monotonic() + 60
        while True:
            try:
                if requests.get(f"http://127.0.0.1:{port}/v2/", timeout=5).status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                assert time.monotonic() < deadline, "the registry did not come up"
                time.sleep(0.2)
        repository = f"localhost:{port}/kiss-registry-test"
        client.images.get(IMAGE).tag(repository, "latest")
        pushed = client.images.push(repository, tag="latest")
        assert '"error"' not in pushed, pushed
        yield repository
    finally:
        registry.remove(force=True)
        for reference in client.images.list(name=repository) if repository else []:
            for tag in reference.attrs.get("RepoTags") or []:
                if tag.startswith(repository):
                    client.images.remove(tag)
            for digest in reference.attrs.get("RepoDigests") or []:
                if digest.startswith(repository):
                    client.images.remove(digest)
        remove_new_image_containers(client, before)


@pytest.mark.slow
@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_open_pulls_a_port_qualified_reference_and_a_digest(registry_repository: str) -> None:
    """``localhost:<port>/name`` and ``name@sha256:...`` pull as docker reads them.

    Split at their last colon they asked the daemon for image
    ``localhost`` with tag ``<port>/name`` ("invalid tag format") and
    for ``name@sha256`` with the digest's hex as its tag.
    """
    client = docker.from_env()
    digests = client.images.get(f"{registry_repository}:latest").attrs["RepoDigests"]
    digest = next(d for d in digests if d.startswith(f"{registry_repository}@"))
    for reference in (registry_repository, digest):
        client.images.remove(f"{registry_repository}:latest")
        with pytest.raises(docker.errors.ImageNotFound):
            client.images.get(reference)
        with DockerManager(reference) as env:
            assert env.Bash("echo pulled", "probe").split("\n")[0] == "pulled"
        client.images.get(reference)


if __name__ == "__main__":
    unittest.main()
