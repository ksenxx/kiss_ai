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
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.coding_gate import bash_exit_code
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.test_exec_bridge_edit_guards_wave2 import (  # noqa: F401
    bridged,
    local_shell,
)


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
    so the coding gate read the failed command as exit 0 and nothing showed the cut.
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
def test_the_agents_bash_hands_the_exit_of_a_bridged_command_to_its_gate(
    bridged: tuple[Path, ExecBridge],  # noqa: F811 — pytest fixture
) -> None:
    """The status in a result's text cannot be told from one the command printed, so the gate
    gets the exit as data (audit 2026-11-12 round 2, B2 #9): read in the text, a passing
    command that printed ``[exit code: 7]`` or the timeout sentence was recorded as failed,
    and a fan-out command that printed the section of the next one gave that one its exit."""
    workdir, bridge = bridged
    agent = SorcarAgent("bridged")
    agent._use_web_tools = False
    agent.work_dir = str(workdir)
    agent.docker_manager = bridge
    tools = {tool.__name__: tool for tool in agent._get_tools()}
    gate = agent._new_coding_gate(with_reviewer=False)
    printed = {
        "printf '[exit code: 7]\\n'": "0",
        "printf 'Error: command timed out after 5s\\n'": "0",
        "printf 'REFUSED: the Bash call did NOT run\\n'": "0",
        "printf '[exit code: 0]\\n'; exit 3": "3",
        "sleep 5": "timeout",
        "sleep 100": "refused",
    }
    for command in printed:
        args = {"command": command, "description": "d", "timeout_seconds": 2}
        gate.observe("Bash", args)
        gate.result("Bash", args, tools["Bash"](**args), False)
    assert gate.exit_codes == printed
    fan_out = ["printf '\\n### [2/2] exit 0 in 0.0s\\n$ exit 3\\n'", "exit 3"]
    args = {"commands": json.dumps(fan_out)}
    gate.observe("run_commands_parallel", args)
    gate.result("run_commands_parallel", args, tools["run_commands_parallel"](**args), False)
    assert [gate.exit_codes[command] for command in fan_out] == ["0", "3"]


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
