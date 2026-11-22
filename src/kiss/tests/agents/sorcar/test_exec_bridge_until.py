# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``ExecBridge.Bash(until=...)`` polls the remote environment inside one call.

The bridge's ``Bash`` re-runs a command every ``poll_seconds`` until its
output matches the regex or the deadline passes, as the Docker tool does,
so a bridged trial waits for a file or a log line in one turn.  A real
bridge server (``ExecBridgeServer``) serves a local environment whose
commands run with ``sh`` in a temporary directory; no mocks.

The status of a failed or killed command is whole in the result under
any ``max_output_chars``: it was cut with the output, and a failed
command then read as one that passed (audit 2026-11-12, B4).

The agent's ``Bash`` over a bridge spends a long ``sleep`` on a running
job only where the environment's shell would run the system's sleep
(``ExecBridge.sleep_probe``): it took every leading ``sleep N`` for the
system's (S16/F3, as it still was over a bridge).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.seas.coding.gate_readers import bash_exit_code
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.agents.sorcar.useful_tools import SLEEP_PROBE
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import LocalEnvironment, _close, _Loop
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import FAKE_SLEEP, _Harness
from kiss.tests.agents.sorcar.test_exec_bridge_edit_guards_wave2 import (  # noqa: F401
    bridged,
    local_shell,
)


class _Sandbox(LocalEnvironment):
    """A local environment whose shells run :attr:`setup` first, as a sandbox's image may.

    Asked which ``sleep`` its shell runs, it leaves the file ``asked`` in its directory first.
    """

    setup = ""

    async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> Any:
        if SLEEP_PROBE in command:
            Path(self.cwd, "asked").touch()
        return await super().exec(self.setup + command, timeout_sec)


@local_shell
def test_bash_until_polls_the_environment_in_one_call(
    bridged: tuple[Path, ExecBridge],  # noqa: F811 — pytest fixture
) -> None:
    """A match on a later run returns at once; no match returns at the deadline; bad input is
    refused."""
    workdir, bridge = bridged
    threading.Timer(1.5, (workdir / "flag").write_text, args=("READY\n",)).start()
    t0 = time.monotonic()
    result = bridge.Bash("cat flag 2>/dev/null; echo checked", "wait for the flag",
                         timeout_seconds=30, until="^READY$", poll_seconds=1)
    assert time.monotonic() - t0 < 20, result
    assert result.startswith("READY\nchecked\n")
    verdict = re.search(r"\[until '\^READY\$': matched on run (\d+) after \d+\.\d s\]$", result)
    assert verdict is not None and int(verdict.group(1)) >= 2, result
    t0 = time.monotonic()
    result = bridge.Bash("echo nothing; exit 2", "wait in vain", timeout_seconds=3,
                         until="NEVER", poll_seconds=1)
    assert 2 <= time.monotonic() - t0 < 10, result
    assert result.startswith("nothing\n") and "[exit code: 2]" in result
    assert re.search(r"\[until 'NEVER': not matched after [23] runs and \d+\.\d s of the 3 s "
                     r"timeout_seconds\]$", result), result
    assert bridge.Bash("true", "bad regex", until="(").startswith(
        "Error: until is not a valid regular expression: ")
    assert bridge.Bash("echo plain", "plain") == "plain\n"


@local_shell
def test_the_status_of_a_command_survives_any_output_budget(
    bridged: tuple[Path, ExecBridge],  # noqa: F811 — pytest fixture
) -> None:
    """A small ``max_output_chars`` cuts the output, never the status that follows it.

    Before: the ``[exit code: N]`` line and the timeout sentence were cut with the output,
    so the coding SEA's gate read the failed command as exit 0 and nothing showed the cut.
    """
    _, bridge = bridged
    for budget in (1, 20, 59, 200):
        result = bridge.Bash("seq 1 200; exit 7", "fails", max_output_chars=budget)
        assert result.endswith("\n[exit code: 7]"), (budget, result)
        assert bash_exit_code(result, container=True) == "7", (budget, result)
    result = bridge.Bash("seq 1 200; exit 7", "fails", max_output_chars=5)
    assert result == "1\n2\n3\n[exit code: 7]"
    result = bridge.Bash("seq 1 200; exit 7", "polled", timeout_seconds=2, max_output_chars=5,
                         until="NEVER", poll_seconds=1)
    assert result.startswith("1\n2\n3\n[exit code: 7]\n[until 'NEVER': not matched after "), result
    # The bridge returns no output of a command it killed: the sentence is the result.
    result = bridge.Bash("seq 1 200; sleep 30", "hangs", timeout_seconds=1, max_output_chars=5)
    assert result == "Error: command timed out after 1s"
    assert bash_exit_code(result, container=True) == "timeout"


@local_shell
def test_bash_until_search_ends_at_the_deadline_and_on_a_stop(
    bridged: tuple[Path, ExecBridge],  # noqa: F811 — pytest fixture
) -> None:
    """A pattern that backtracks for minutes holds neither the call nor this process.

    Audit 2026-11-10 (C13-F1): the loop of the container shells searched the output in
    the daemon's process, on the task thread; ``(a+)+$`` on thirty ``a`` and a ``!`` kept
    the GIL for over a minute, past the call's deadline and a Stop.
    """
    _, bridge = bridged
    command = f"echo {'a' * 30}!"
    verdict = r"\[until '\(a\+\)\+\$': not matched after 1 runs and \d+\.\d s of the %d s "
    t0 = time.monotonic()
    result = bridge.Bash(command, "poll", timeout_seconds=2, until="(a+)+$", poll_seconds=1)
    assert time.monotonic() - t0 < 20, result
    assert re.search(verdict % 2 + r"timeout_seconds\]$", result), result
    bridge.stop_event = threading.Event()
    threading.Timer(0.5, bridge.stop_event.set).start()
    t0 = time.monotonic()
    result = bridge.Bash(command, "stopped poll", timeout_seconds=60, until="(a+)+$")
    assert time.monotonic() - t0 < 20, result
    assert re.search(verdict % 60 + r"timeout_seconds\]$", result), result


@local_shell
@pytest.mark.skipif(
    not (shutil.which("setsid") and shutil.which("bash")),
    reason="the job scripts need setsid, the startup file bash",
)
def test_a_long_sleep_is_spent_on_a_job_only_where_the_environment_runs_the_systems_sleep(
    tmp_path: Path,
) -> None:
    """The agent's bridged ``Bash`` asks the environment which ``sleep`` its shell would run.

    It took every leading ``sleep N`` for the system's: with another
    ``sleep`` first on the environment's ``PATH`` the wait on the job
    stood for a command that fails at once, and the rest of it ran.
    The question is a command like any other (``ExecBridge.sleep_probe``),
    so a function of the startup file its shell reads is seen as well;
    a question that fails, or that no bridge takes, is no yes, and its
    answer is the first line of the output, whatever the shell wrote
    to stderr.
    """
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "sleep").write_text(FAKE_SLEEP)
    (fake / "sleep").chmod(0o755)
    (fake / "rc").write_text('sleep() { [ "$1" != 31 ] || return 7; command sleep "$@"; }\n')
    # The job files stay in the test's directory, whatever this process's environment names.
    plain = f"export KISS_SUBAGENT_DIR={tmp_path}/private; "
    # It runs until the environment is asked: the tool has found it running by then,
    # however long that took (or until this process is gone).
    until_asked = f"until [ -e asked ] || ! kill -0 {os.getpid()} 2>/dev/null; do sleep 0.1; done"
    loop = _Loop()
    sandbox = _Sandbox(str(tmp_path))
    sandbox.setup = plain
    server = ExecBridgeServer(sandbox, loop.loop)
    try:
        with ExecBridge(f"exec:{server.url}") as bridge:
            h = _Harness(cast(Any, bridge))
            bash = h.tools["Bash"]
            try:
                job = bash("sleep 300", "job", background=True).split()[3]
                for setup in (
                    f"export PATH={fake}:$PATH; ",  # another sleep first on the PATH
                    "export PATH=fake:$PATH; ",  # by a relative entry: the question fails
                    f"export BASH_ENV={fake}/rc; ",  # a function of the shell's startup file
                ):
                    sandbox.setup = plain + setup
                    out = bash("sleep 31 && echo RAN", "poll", timeout_seconds=35)
                    assert out == "[exit code: 7]", (setup, out)
                    (tmp_path / "asked").unlink()  # the environment was asked, each time
                sandbox.setup = plain
                waited = bash(until_asked, "job", background=True).split()[3]
                sandbox.setup = plain + "echo 'no such locale' >&2; "
                out = bash("sleep 31 && echo RAN", "poll", timeout_seconds=35)
                assert out == (
                    f"RAN\nno such locale\n\nWaited on job {waited} for up to 31 s instead of "
                    "sleeping (exited 0).\n"
                ), out
                sandbox.setup = plain
                assert h.jobs.last_running() == job, "a sleep above had no job to be spent on"
            finally:
                sandbox.setup = plain
                for started in list(h.jobs.commands):
                    h.jobs.bash_job(started, "kill")
    finally:
        _close(server, loop)
    assert ExecBridge("exec:http://127.0.0.1:notaport").sleep_probe("sleep") == ""
