# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F4): an unanswered directory probe authorizes no rewrite.

The container probe of the shell guards answers ``test -e`` and ``test
-d`` questions with ``exit_code != 1``, so a probe that gave no answer
(timed out, stopped) counted as passing.  That is the conservative
answer to the here-document guard's *existence* question (an unchecked
target is refused rather than overwritten), but the redirect of ``cd
DIR && <rewrite>`` asks the *directory* question: there the same answer
authorized the rewrite, and the redirected Write, with a fresh deadline
of its own, wrote the file bash would never have reached after the
failed ``cd``.  Only an answered ``test -d`` (exit 0) passes now; an
unanswered one refuses the rewrite like a missing directory, and the
existence question keeps its conservative answer.

Every test runs against a real container (or the real exec bridge
server over a local environment) through the real ``SorcarAgent`` tool
set and a real ``KISSAgent`` dispatch.  Nothing is mocked.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.agents.sorcar.useful_tools import CD_VERDICT
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import LocalEnvironment, _close, _Loop
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import _Harness

pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")

MISSING_DIR = "/s5-directory-that-does-not-exist"
REWRITE = f"cd {MISSING_DIR} && cat > {{path}} <<'EOF'\nwrong\nEOF"


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """One container of :data:`IMAGE` working in ``/tmp`` for the whole module."""
    manager = DockerManager(IMAGE, workdir="/tmp", mount_shared_volume=False)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_a_timed_out_or_stopped_directory_probe_refuses_the_rewrite(env: DockerManager) -> None:
    """``cd <missing> && cat > f``: no answer to ``test -d`` means no Write, as a ``cd`` failure."""
    h = _Harness(env)
    assert env.container is not None
    # The shell itself: the cd fails and nothing after it runs.
    direct = env.container.exec_run(["bash", "-c", REWRITE.format(path="/tmp/cd-direct.txt")])
    assert direct.exit_code != 0
    assert env.container.exec_run(["test", "!", "-e", "/tmp/cd-direct.txt"]).exit_code == 0
    # A probe past its deadline (a 0 s one) gives no answer: refused, nothing written,
    # no Write reaches the hook.
    command = REWRITE.format(path="/tmp/cd-timeout.txt")
    out = h.call("Bash", command=command, description="cd, rewrite", timeout_seconds=0)
    assert out == CD_VERDICT.format(cwd=MISSING_DIR), out
    assert h.hook_calls[-1][0] == "Bash" and h.tool_calls() == ["Bash"]
    assert env.Bash("test -e /tmp/cd-timeout.txt", "check", 30, 500) == "[exit code: 1]"
    # A stopped probe: the same.
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        out = h.call("Bash", command=command, description="cd, rewrite")
    finally:
        env.stop_event = None
    assert out == CD_VERDICT.format(cwd=MISSING_DIR), out
    assert h.tool_calls() == ["Bash", "Bash"]
    assert env.Bash("test -e /tmp/cd-timeout.txt", "check", 30, 500) == "[exit code: 1]"
    # An answered probe still lets the rewrite through, into the directory the cd named.
    env.Bash("rm -rf /tmp/cd-ok && mkdir -p /tmp/cd-ok", "clean", 30, 500)
    ok = "cd /tmp/cd-ok && cat > f.txt <<'EOF'\nright\nEOF"
    out = h.call("Bash", command=ok, description="ok")
    assert out.startswith('Applied as Write(file_path="/tmp/cd-ok/f.txt", content="right\\n")'), out
    assert env.read_bytes("/tmp/cd-ok/f.txt") == b"right\n"


def test_the_existence_probe_keeps_its_conservative_answer(env: DockerManager) -> None:
    """``run_commands_parallel`` (no redirect) still refuses a copy whose target nobody checked."""
    env.Bash("rm -f /tmp/fan-unchecked.txt", "clean", 30, 500)
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        report = env.run_commands_parallel(
            json.dumps(["cat > /tmp/fan-unchecked.txt <<'EOF'\nx\nEOF"]), timeout_seconds=5,
        )
    finally:
        env.stop_event = None
    assert HEREDOC_VERDICT.format(path="/tmp/fan-unchecked.txt") in report, report
    assert env.Bash("test -e /tmp/fan-unchecked.txt", "check", 30, 500) == "[exit code: 1]"


def test_the_bridge_refuses_a_rewrite_after_a_stopped_directory_probe(tmp_path: Path) -> None:
    """``ExecBridge`` answers its ``test -d`` the same way: a stopped probe is no directory."""
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(tmp_path)), loop.loop)
    try:
        with ExecBridge(f"exec:{server.url}") as bridge:
            h = _Harness(cast(Any, bridge))
            target = tmp_path / "cd-stopped.txt"
            bridge.stop_event = threading.Event()
            bridge.stop_event.set()
            try:
                out = h.call("Bash", command=REWRITE.format(path=target), description="cd")
            finally:
                bridge.stop_event = None
            assert out == CD_VERDICT.format(cwd=MISSING_DIR), out
            assert h.tool_calls() == ["Bash"] and not target.exists()
    finally:
        _close(server, loop)
