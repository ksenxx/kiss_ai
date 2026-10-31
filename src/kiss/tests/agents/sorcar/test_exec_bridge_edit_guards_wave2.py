# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The exec bridge refuses the shell rewrites a Docker task refuses.

``DockerManager`` runs every ``Bash`` and ``run_commands_parallel``
command through ``_refuse_or_lift``: the destructive and install-timeout
guards plus the refusals of rewrites that belong in ``Edit`` (a
multi-line ``sed -i``, a ``cat``/``tee`` here-document into an existing
file).  ``ExecBridge``, its drop-in for a remote sandbox, applied only
the first two, so a bridged task could overwrite a file with a
here-document that a Docker task is refused.  The bridge now applies the
same guards, resolving relative targets against the environment's own
working directory, which it asks the environment for on entry.

A real bridge server (``ExecBridgeServer``) serves a local environment
whose commands run with ``sh`` in a temporary directory; no mocks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_UNRESOLVED_VERDICT, HEREDOC_VERDICT
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import (
    LocalEnvironment,
    _Loop,
    _no_reaper_thread,
    _wait_until,
)
from kiss.tests.agents.sorcar.test_exec_bridge_stop_and_guards import _bridge
from kiss.tests.conftest import posix_only

local_shell = posix_only("the bridged local environment runs sh commands")


@pytest.fixture
def bridged(tmp_path: Path) -> Iterator[tuple[Path, ExecBridge]]:
    """An entered bridge to a local environment working in a directory holding ``data.txt``."""
    workdir = tmp_path / "app"
    workdir.mkdir()
    (workdir / "data.txt").write_text("v1\n")
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(workdir)), loop.loop)
    try:
        # ``workdir`` is the host-side directory the agent passes; the
        # environment's own working directory is another one.
        with ExecBridge(f"exec:{server.url}", workdir="/host/side", volumes={}) as bridge:
            yield workdir, bridge
    finally:
        server.environment.kill_all()
        assert _wait_until(_no_reaper_thread)
        server.close()
        loop.close()


@local_shell
def test_bash_refuses_the_rewrites_a_docker_task_refuses(
    bridged: tuple[Path, ExecBridge],
) -> None:
    workdir, bridge = bridged
    data = workdir / "data.txt"
    assert bridge.workdir == str(workdir)

    overwrite = bridge.Bash("cat > data.txt <<'EOF'\nv2\nEOF", "overwrite")
    assert overwrite == HEREDOC_VERDICT.format(path=str(data))
    sed = bridge.Bash("sed -i 's/v1/v1\\nv2/' data.txt", "multi-line sed")
    assert sed.startswith("REFUSED: the `sed -i` call did NOT run"), sed
    assert data.read_text() == "v1\n"
    # A here-document may still create a new file.
    bridge.Bash("cat > fresh.txt <<'EOF'\nnew\nEOF", "create")
    assert (workdir / "fresh.txt").read_text() == "new\n"


@local_shell
def test_run_commands_parallel_refuses_them_too(bridged: tuple[Path, ExecBridge]) -> None:
    workdir, bridge = bridged
    report = bridge.run_commands_parallel(
        json.dumps(["cat >> data.txt <<'EOF'\nv2\nEOF", "echo ok"]),
    )
    assert report.startswith("2 commands: 1 succeeded, 1 failed, 0 timed out."), report
    assert HEREDOC_VERDICT.format(path=str(workdir / "data.txt")) in report
    assert (workdir / "data.txt").read_text() == "v1\n"


def test_an_environment_without_a_usable_working_directory_refuses_relative_targets() -> None:
    # The stand-in bridge answers every command with ``RAN``: no absolute
    # ``pwd``, so a relative target cannot be checked and never runs.
    with _bridge() as (server, bridge), bridge:
        assert bridge.workdir is None
        refusal = bridge.Bash("cat > notes.txt <<'EOF'\nv\nEOF", "relative")
        assert refusal == HEREDOC_UNRESOLVED_VERDICT.format(path="notes.txt")
        assert server.commands == ["pwd"]
