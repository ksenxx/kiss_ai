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
working directory, which it asks the environment for on entry.  A
literal here-document copy or mechanical edit is, as in a Docker task,
redirected to the ``Write``/``Edit`` it stands for (``ToolRedirect``)
rather than refused; ``run_commands_parallel`` still refuses it.

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
from kiss.agents.sorcar.useful_tools import CD_VERDICT
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import (
    LocalEnvironment,
    _Loop,
    _no_reaper_thread,
    _wait_until,
)
from kiss.tests.agents.sorcar.test_exec_bridge_stop_and_guards import _bridge
from kiss.tests.conftest import host_sed_is_gnu, posix_only

local_shell = posix_only("the bridged local environment runs sh commands")
needs_gnu_sed = pytest.mark.skipif(not host_sed_is_gnu(), reason="needs GNU sed as the oracle")


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

    # A copy into an existing file, an append and a copy creating a file
    # are the Write they stand for, made by the agent; the bridge runs nothing.
    with pytest.raises(ToolRedirect) as overwrite:
        bridge.Bash("cat > data.txt <<'EOF'\nv2\nEOF", "overwrite")
    assert (overwrite.value.name, overwrite.value.arguments) == (
        "Write", {"file_path": str(data), "content": "v2\n"},
    )
    with pytest.raises(ToolRedirect) as append:
        bridge.Bash("cat >> data.txt <<'EOF'\nv2\nEOF", "append")
    assert append.value.arguments == {"file_path": str(data), "content": "v1\nv2\n"}
    assert append.value.note.startswith("Appended through Write(")
    with pytest.raises(ToolRedirect) as create:
        bridge.Bash("cat > fresh.txt <<'EOF'\nnew\nEOF", "create")
    assert create.value.arguments == {"file_path": str(workdir / "fresh.txt"), "content": "new\n"}
    assert not (workdir / "fresh.txt").exists()
    with pytest.raises(ToolRedirect) as sed:
        bridge.Bash("sed -i 's/v1/v1\\nv2/' data.txt", "multi-line sed")
    assert (sed.value.name, sed.value.arguments) == ("Edit", {
        "file_path": str(data), "old_string": "v1", "new_string": "v1\nv2", "replace_all": False,
    })
    assert data.read_text() == "v1\n"
    # Other rewrites stay refused: a sed the Edit cannot spell, a target only the shell knows.
    refused = bridge.Bash("sed -i 's/v1/&\\n/' data.txt", "multi-line sed")
    assert refused.startswith("REFUSED: the `sed -i` call did NOT run"), refused
    refused = bridge.Bash('F=data.txt; cat > "$F" <<\'EOF\'\nv2\nEOF', "variable target")
    assert refused.startswith("REFUSED: the command did not run: it writes ")
    assert "only the shell knows which file that is" in refused
    assert data.read_text() == "v1\n"
    # The bytes the append read come through the bridge's fetch.
    assert bridge.read_bytes(str(data)) == b"v1\n"
    with pytest.raises(OSError, match="over the 1 byte limit"):
        bridge.read_bytes(str(data), max_bytes=1)
    with pytest.raises(OSError, match="cannot read "):
        bridge.read_bytes(str(workdir / "missing.txt"))


@local_shell
def test_the_rest_of_a_compound_rewrite_is_the_redirects_follow_up(
    bridged: tuple[Path, ExecBridge],
) -> None:
    """The redirect names the rest of ``cd DIR && <rewrite> && <rest>`` as the Bash to make next."""
    workdir, bridge = bridged
    data = workdir / "data.txt"
    (workdir / "app").mkdir()
    with pytest.raises(ToolRedirect) as copy:
        bridge.Bash("cd app && cat > f.txt <<'EOF' && cat f.txt\nv\nEOF", "copy, show", 45, 900)
    assert (copy.value.name, copy.value.arguments) == (
        "Write", {"file_path": str(workdir / "app" / "f.txt"), "content": "v\n"},
    )
    assert copy.value.followup == ("Bash", {
        "command": f"cd {workdir / 'app'} && cat f.txt", "description": "copy, show",
        "timeout_seconds": 45, "max_output_chars": 900,
    })
    assert not (workdir / "app" / "f.txt").exists()
    # A per-line sed gets the pairs exact for the file's text, fetched through the bridge.
    data.write_text("v1\nv1\n")
    with pytest.raises(ToolRedirect) as sed:
        bridge.Bash("sed -i 's/v1/v2/' data.txt; wc -l data.txt", "sed")
    assert (sed.value.name, sed.value.arguments) == ("Edit", {
        "file_path": str(data), "old_string": "v1", "new_string": "v2", "replace_all": True,
    })
    assert sed.value.followup == ("Bash", {
        "command": "wc -l data.txt", "description": "sed",
        "timeout_seconds": 30, "max_output_chars": 8000,
    })
    # A rewrite that is the whole command has no follow-up.
    with pytest.raises(ToolRedirect) as whole:
        bridge.Bash("sed -i 's/v1/v2/' data.txt", "sed")
    assert whole.value.followup is None
    # A line holding the pattern twice has no exact Edit: a multi-line sed stays refused (a
    # plain one runs in the environment: the next test).
    data.write_text("v1 v1\n")
    refused = bridge.Bash("sed -i 's/v1/v2\\n/' data.txt && cat data.txt", "sed")
    assert refused.startswith("REFUSED: the `sed -i` call did NOT run"), refused
    assert data.read_text() == "v1 v1\n"
    # A cd to a missing directory, or into a file, is refused before anything is written:
    # bash would run nothing after it (the probe asks ``test -d``, not ``test -e``).
    refusal = bridge.Bash("cd nowhere && cat > f.txt <<'EOF' && cat f.txt\nv\nEOF", "cd")
    assert refusal == CD_VERDICT.format(cwd=workdir / "nowhere")
    assert not (workdir / "nowhere").exists()
    refusal = bridge.Bash("cd data.txt && cat > f.txt <<'EOF' && cat f.txt\nv\nEOF", "cd")
    assert refusal == CD_VERDICT.format(cwd=data) and not (workdir / "f.txt").exists()


@local_shell
@needs_gnu_sed
def test_a_plain_sed_without_an_exact_edit_runs_in_the_environment(
    bridged: tuple[Path, ExecBridge],
) -> None:
    """A line holding the pattern twice has no exact Edit: the sed runs as the command it is."""
    workdir, bridge = bridged
    (workdir / "data.txt").write_text("v1 v1\n")
    assert bridge.Bash("sed -i 's/v1/v2/' data.txt && cat data.txt", "sed") == "v2 v1\n"


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
