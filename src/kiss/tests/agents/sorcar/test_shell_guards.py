# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the shell guards shared by every Sorcar shell tool.

The destructive-command guard and the install-timeout lift came from the
HarnessTax container harness (``coding_sea.py``); these tests exercise them
through the host ``UsefulTools.Bash`` / ``run_commands_parallel`` tools and,
when a Docker daemon is running, through ``DockerManager``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import docker
import pytest

from kiss.agents.seas.coding import coding_sea
from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.shell_guards import (
    DESTRUCTIVE_VERDICT,
    INSTALL_TIMEOUT_SECONDS,
    destructive_command_guard,
    lift_install_timeout,
)
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    remove_new_image_containers,
)

BLOCKED = [
    "kill -9 -1",
    "kill -1",
    "echo cleanup; kill -TERM -1",
    "pkill -f .",
    "pkill -9 -f '.*'",
    "sudo rm -rf /",
    "rm -rf {wd}",
    "rm -rf {wd}/",
    "rm -rf '{wd}/*'",
    "cd / && rm -r {wd}",
    "echo cleanup\nkill -9 -1",
    "echo cleanup\nrm -rf {wd}",
    "rm -rf /tmp/one {wd}",
    "rm -rf /*",
    "rm -rf {wd}*",
]

ALLOWED = [
    "kill -1 1234",
    "kill -9 $PID",
    "kill -15 %1",
    "pkill -f myserver",
    "pkill -f 'python app.py'",
    "rm -rf {wd}/build",
    "rm -rf ./tmp/x",
    "rm -rf /tmp/kiss-scratch",
    "grep -R 'kill -1' .",
    "echo 'rm -rf /'",
    "rm -rf build/ {wd}/sub",
    "rm -rf {wd}er",
    "git rm -r --cached {wd}/sub",
    "rm -rf ./build && cd {wd}",
    "rm file.txt; ls {wd}",
    "rm file.txt\nprintf '%s' {wd}",
    "rm -f out.log > {wd}/log.txt",
    "rm -rf ./build | tee {wd}/log.txt",
]


def test_destructive_guard_protects_root_and_work_dir(tmp_path: Path) -> None:
    """Root and the work dir (any spelling) are refused; everything else runs."""
    wd = str(tmp_path)
    for command in BLOCKED:
        assert destructive_command_guard(command.format(wd=wd), wd) == DESTRUCTIVE_VERDICT, command
    for command in ALLOWED:
        assert destructive_command_guard(command.format(wd=wd), wd) is None, command
    # The trailing-slash spelling and the resolved path of a symlinked work dir
    # are protected too, and ``None`` protects the root only.
    link = tmp_path.parent / (tmp_path.name + "-link")
    link.symlink_to(tmp_path)
    assert destructive_command_guard(f"rm -rf {tmp_path}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {link}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {tmp_path}", f"{tmp_path}/") == DESTRUCTIVE_VERDICT
    assert destructive_command_guard("rm -rf /", None) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {tmp_path}", None) is None


def test_install_timeout_lift() -> None:
    """Installs and builds get at least 900 s; other commands keep their timeout."""
    for command in ["pip install requests", "python3 -m pip install -e .", "uv sync",
                    "uv pip install x", "sudo apt-get install -y gcc", "npm install",
                    "pnpm add lodash", "cargo build --release", "cd src && make -j8",
                    "cmake --build .", "go build ./...", "npm ci", "npm run build",
                    "echo prepare\nmake"]:
        assert lift_install_timeout(command, 30) == INSTALL_TIMEOUT_SECONDS, command
        assert lift_install_timeout(command, 1800) == 1800, command
    for command in ["uv run pytest", "npm get registry", "grep -R make .", "go test ./...",
                    "pip list", "echo make"]:
        assert lift_install_timeout(command, 30) == 30, command


def test_bash_refuses_destructive_commands_without_running_them(tmp_path: Path) -> None:
    """``Bash`` returns the verdict and touches nothing; allowed deletes still run."""
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "build").mkdir()
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Bash(f"rm -rf {tmp_path}", "wipe") == DESTRUCTIVE_VERDICT
    assert tools.Bash("kill -9 -1", "kill all") == DESTRUCTIVE_VERDICT
    assert tools.Bash(f"rm -rf {tmp_path}", "wipe in background", background=True) == (
        DESTRUCTIVE_VERDICT)
    assert keep.read_text() == "keep"
    assert tools.Bash(f"rm -rf {tmp_path}/build && echo gone", "allowed").strip() == "gone"
    assert not (tmp_path / "build").exists()


def test_bash_lifts_install_timeout(tmp_path: Path) -> None:
    """A build asked to run with a 1 s timeout is not killed after 1 s."""
    (tmp_path / "Makefile").write_text("all:\n\tsleep 2; echo built\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    started = time.monotonic()
    out = tools.Bash("make", "slow build", timeout_seconds=1)
    assert time.monotonic() - started >= 2
    assert "built" in out and "timeout" not in out.lower()
    # A non-build command with the same timeout is still killed at the deadline,
    # and the model sees what it printed before the kill.
    out = tools.Bash("echo phase-1; sleep 30", "slow probe", timeout_seconds=1)
    assert out.startswith("Error: Command execution timeout after 1s. Output before the timeout:")
    assert "phase-1" in out
    assert tools.Bash("sleep 30", "silent probe", timeout_seconds=1) == (
        "Error: Command execution timeout")


def test_run_commands_parallel_guards_each_command(tmp_path: Path) -> None:
    """A destructive command in a batch is refused; its siblings still run."""
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "Makefile").write_text("all:\n\tsleep 2; echo built\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    report = tools.run_commands_parallel(
        f'["rm -rf {tmp_path}", "echo sibling", "make"]', timeout_seconds=1)
    assert DESTRUCTIVE_VERDICT in report
    assert "sibling" in report
    assert "built" in report
    assert keep.read_text() == "keep"


def test_coding_sea_shares_the_guards() -> None:
    """The HarnessTax harness uses the shared patterns, not private copies."""
    from kiss.agents.sorcar import shell_guards

    assert coding_sea.INSTALL_COMMANDS is shell_guards.INSTALL_COMMANDS
    assert coding_sea.INSTALL_TIMEOUT_SECONDS == shell_guards.INSTALL_TIMEOUT_SECONDS
    assert shell_guards.destructive_pattern("/testbed/").search("rm -rf /testbed")


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture
def cleanup() -> Iterator[set[str]]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield before
    finally:
        remove_new_image_containers(client, before)


@pytest.mark.slow
@pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not running")
def test_docker_bash_applies_the_guards(cleanup: set[str]) -> None:
    """Docker ``Bash`` and ``run_commands_parallel`` refuse destructive commands
    against the container work dir and keep a build alive past a short timeout."""
    mgr = DockerManager(IMAGE, workdir="/work")
    mgr.open()
    try:
        assert mgr.Bash("mkdir -p /work && echo keep > /work/keep.txt", "seed").strip() == ""
        assert mgr.Bash("rm -rf /work", "wipe") == DESTRUCTIVE_VERDICT
        assert mgr.Bash("kill -9 -1", "kill all") == DESTRUCTIVE_VERDICT
        assert mgr.Bash("cat /work/keep.txt", "check").strip() == "keep"
        report = mgr.run_commands_parallel(
            '["rm -rf /work/", "echo sibling", "cat /work/keep.txt"]', timeout_seconds=30)
        assert DESTRUCTIVE_VERDICT in report and "sibling" in report and "keep" in report
        # ``make`` is a build: the 1 s timeout is lifted, so the 2 s sleep completes.
        # Non-streaming Docker paths keep the output printed before the kill.
        out = mgr.Bash("echo phase-1; sleep 2; echo built", "slow probe", timeout_seconds=1)
        assert "timed out" in out and "phase-1" in out and "built" not in out, out
        report = mgr.run_commands_parallel('["echo phase-2; sleep 5"]', timeout_seconds=1)
        assert "phase-2" in report, report
        # The slim image has no ``make``; a stand-in build tool of that name
        # (``_CMD`` accepts a directory prefix) shows the lift end to end.
        mgr.Bash("printf '#!/bin/sh\\nsleep 2; echo built\\n' > /work/make; chmod +x /work/make",
                 "fake make")
        out = mgr.Bash("/work/make -j2", "slow build", timeout_seconds=1)
        assert "built" in out and "timed out" not in out, out
    finally:
        mgr.close()
    assert not os.path.exists(mgr.host_shared_path or "/nonexistent-kiss-shared")
