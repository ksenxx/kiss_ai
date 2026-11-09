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
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.exec_bridge import ExecBridge
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
