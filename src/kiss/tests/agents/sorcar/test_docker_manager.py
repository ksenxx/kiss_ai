# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test suite for DockerManager without mocking."""

import contextlib
import os
import re
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path

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
    ``localhost:<port>/kiss-registry-test`` has to pull it. The port is
    published as ``docker run -P`` does: on Docker Desktop an explicit
    binding with no host port left the daemon's push to ``localhost``
    dialing ``[::1]:<port>`` until it timed out.
    """
    client = docker.from_env()
    before = image_container_ids(client)
    registry = client.containers.run(
        "registry:2", detach=True, publish_all_ports=True,
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
            attrs = reference.attrs
            for name in [*(attrs.get("RepoTags") or []), *(attrs.get("RepoDigests") or [])]:
                # A containerd image store drops the digest with the tag.
                with contextlib.suppress(docker.errors.ImageNotFound):
                    if name.startswith(repository):
                        client.images.remove(name)
        remove_new_image_containers(client, before)


@pytest.mark.slow
@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_open_pulls_a_port_qualified_reference_and_a_digest(registry_repository: str) -> None:
    """``localhost:<port>/name`` and ``name@sha256:...`` pull as docker reads them.

    Split at their last colon they asked the daemon for image
    ``localhost`` with tag ``<port>/name`` ("invalid tag format") and
    for ``name@sha256`` with the digest's hex as its tag.

    The digest is the registry's own: the local ``RepoDigests`` entry of a
    containerd image store is the multi-platform index, which a push of
    one platform never uploads.
    """
    client = docker.from_env()
    host, _, name = registry_repository.partition("/")
    accept = ", ".join((
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ))
    manifest = requests.head(
        f"http://{host}/v2/{name}/manifests/latest", headers={"Accept": accept}, timeout=10,
    )
    manifest.raise_for_status()
    digest = f"{registry_repository}@{manifest.headers['Docker-Content-Digest']}"
    for reference in (registry_repository, digest):
        client.images.remove(f"{registry_repository}:latest")
        with pytest.raises(docker.errors.ImageNotFound):
            client.images.get(reference)
        with DockerManager(reference) as env:
            assert env.Bash("echo pulled", "probe").split("\n")[0] == "pulled"
        client.images.get(reference)


@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_close_empties_the_shared_volume_but_not_a_bind_mounted_below_it(tmp_path: Path) -> None:
    """``close()`` deletes what the container made under ``/testbed``, not the caller's mounts.

    S3/F1.  The container empties the shared volume itself (what its commands
    made there is root-owned on the host, out of ``rmtree``'s reach);
    the sweep used to descend into a caller's bind mount below
    ``/testbed`` and delete the caller's files.  All three mounts keep
    theirs, the glob character in one's path and the backslash in
    another's (an escape to ``find -path``, which ``glob.escape`` leaves
    as it is) notwithstanding, while the host side of the shared volume
    is gone.
    """
    project, data, back = tmp_path / "project", tmp_path / "data", tmp_path / "back"
    for directory in (project, data, back):
        directory.mkdir()
        (directory / "kept.txt").write_text(f"{directory.name} kept\n", encoding="utf-8")
    env = DockerManager(
        IMAGE,
        volumes={
            str(project): "/testbed/project/",
            str(data): "/testbed/da[t]a",
            str(back): "/testbed/da\\ta",
        },
    )
    env.open()
    shared = env.host_shared_path
    assert shared is not None and os.path.isdir(shared)
    filled = env.Bash(
        "mkdir -p /testbed/owned/deep && echo made > /testbed/owned/deep/file"
        " && cat /testbed/project/kept.txt '/testbed/da[t]a/kept.txt' '/testbed/da\\ta/kept.txt'",
        "fill",
    )
    assert filled.split("\n")[:3] == ["project kept", "data kept", "back kept"], filled
    env.close()
    assert env.container_name is None
    assert (project / "kept.txt").read_text(encoding="utf-8") == "project kept\n"
    assert (data / "kept.txt").read_text(encoding="utf-8") == "data kept\n"
    assert (back / "kept.txt").read_text(encoding="utf-8") == "back kept\n"
    assert not os.path.exists(shared), os.listdir(shared)


@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_bash_until_re_runs_the_command_until_its_output_matches() -> None:
    """``Bash(until=)`` polls inside one call: a later match, the deadline, the refusals."""
    with DockerManager(IMAGE, workdir="/tmp") as env:
        # The flag appears two seconds in: the first run misses it, a later one finds it.
        env.Bash("(sleep 2; echo READY > /tmp/flag) > /dev/null 2>&1 &", "raise a flag later")
        t0 = time.monotonic()
        result = env.Bash("cat /tmp/flag 2>/dev/null; echo checked", "wait for the flag",
                          timeout_seconds=30, until="^READY$", poll_seconds=1)
        assert time.monotonic() - t0 < 20, result
        assert result.startswith("READY\nchecked\n")
        verdict = re.search(r"\[until '\^READY\$': matched on run (\d+) after (\d+\.\d) s\]$",
                            result)
        assert verdict is not None, result
        assert int(verdict.group(1)) >= 2 and 1.0 <= float(verdict.group(2)) < 20
        # No match within the deadline: the call lasts the whole timeout and returns
        # the last run's output, its exit code and the tally.
        t0 = time.monotonic()
        result = env.Bash("echo still nothing; exit 3", "wait in vain", timeout_seconds=3,
                          until="NEVER", poll_seconds=0)
        elapsed = time.monotonic() - t0
        assert 2.9 <= elapsed < 10, result
        assert result.startswith("still nothing\n") and "[exit code: 3]" in result
        assert re.search(r"\[until 'NEVER': not matched after [23] runs and 3\.\d s of the 3 s "
                         r"timeout_seconds\]$", result), result
        # An interval longer than the timeout: one run, and still the whole timeout.
        t0 = time.monotonic()
        result = env.Bash("echo once", "wait once", timeout_seconds=3, until="NEVER",
                          poll_seconds=5)
        assert 2.9 <= time.monotonic() - t0 < 10, result
        assert re.search(r"not matched after 1 runs and 3\.\d s of the 3 s", result), result
        # A run the deadline kills ends the loop with the timeout result: a first run
        # that hangs, and a later run that gets only what is left of the deadline.
        result = env.Bash("echo early; tail -f /dev/null", "hang", timeout_seconds=2, until="late")
        assert result.startswith("Error: command timed out after 2") and "early" in result
        assert re.search(r"\[until 'late': not matched after 1 runs and [2-9]\.\d s of the 2 s "
                         r"timeout_seconds\]$", result), result
        t0 = time.monotonic()
        result = env.Bash("sleep 1.5; echo slow", "slow poll", timeout_seconds=3, until="NEVER",
                          poll_seconds=1)
        assert time.monotonic() - t0 < 6, result
        assert result.startswith("Error: command timed out after 3"), result
        assert re.search(r"not matched after 2 runs and [3-4]\.\d s of the 3 s", result), result
        # The refusals: a bad regex, a poll interval that is not a number; the
        # guards still apply first, and a plain call is untouched.
        assert env.Bash("true", "bad regex", until="[").startswith(
            "Error: until is not a valid regular expression: ")
        assert env.Bash("true", "bad poll", until="x", poll_seconds="soon").startswith(  # type: ignore[arg-type]
            "Error: poll_seconds must be a number of seconds, got 'soon'")
        # An integer no float can hold raised OverflowError out of the tool.
        assert env.Bash("true", "huge poll", until="x", poll_seconds=10**400).startswith(
            "Error: poll_seconds must be a number of seconds, got 1000")
        assert env.Bash("rm -rf /", "guarded", until="x").startswith("REFUSED: ")
        assert env.Bash("echo plain", "plain") == "plain\n"
        # A pattern the search child cannot be given as an argument (a NUL, a
        # character with no bytes) is refused too: accepted, nothing was searched
        # and the call ran to its deadline although its output matched.
        for unsearchable in ("a\0b", "\ud800"):
            t0 = time.monotonic()
            result = env.Bash("printf 'a\\0b'", "unsearchable", timeout_seconds=8,
                              until=unsearchable)
            assert result.startswith("Error: until is not a valid regular expression: "), result
            assert time.monotonic() - t0 < 4, result
        # An output too long to keep whole is not searched (the match may be in the
        # middle that was dropped): the poll ends on that run and says so, as on the host.
        flood = ("head -c 1500000 /dev/zero | tr '\\0' a; echo; echo MIDDLE; "
                 "head -c 1700000 /dev/zero | tr '\\0' b")
        t0 = time.monotonic()
        result = env.Bash(flood, "flood", timeout_seconds=20, until="MIDDLE", poll_seconds=1)
        assert time.monotonic() - t0 < 12, result[-300:]
        assert re.search(
            r"\[until 'MIDDLE': not searched on run 1: the output is too long to keep whole "
            r"\([\d,]+ characters dropped\); poll with a command that prints less "
            r"\(grep, tail\)\]$", result), result[-300:]
        # A task Stop ends the wait after the current run.
        env.stop_event = threading.Event()
        threading.Timer(1.5, env.stop_event.set).start()
        t0 = time.monotonic()
        result = env.Bash("echo polled", "stopped wait", timeout_seconds=60, until="NEVER",
                          poll_seconds=3)
        assert time.monotonic() - t0 < 10, result
        assert "[until 'NEVER': not matched after 1 runs and" in result


if __name__ == "__main__":
    unittest.main()
