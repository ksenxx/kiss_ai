# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: an agent error must not kill the interactive REPL.

The :mod:`kiss.agents.sorcar.cli_repl` module contract says the agent
"runs in a stateful loop so that … the prompt returns and waits for the
next instruction after a task finishes instead of exiting".

Bug: ``_run_one`` caught only ``KeyboardInterrupt``.  Any other
exception raised by ``agent.run`` (model API failure, network error,
budget error, …) propagated straight out of ``run_repl``'s loop and
crashed the whole interactive session with a traceback — losing the
user's session instead of printing the error and returning to the
prompt.

The test drives the real ``run_repl`` loop in a child interpreter (no
mocks): the agent's ``run`` raises ``RuntimeError`` on the first task;
the session must report the failure, keep running, and then exit
cleanly via ``/exit``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = """
import sys


class BoomAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        sys.stderr.write("RUN_CALLED\\n")
        raise RuntimeError("model API exploded")


from kiss.agents.sorcar.cli_repl import run_repl

run_repl(BoomAgent(), {"work_dir": ".", "model_name": "demo-model"})
print("REPL_EXITED_CLEANLY")
"""


def _run_repl_with_boom_agent(tmp_path: Path) -> subprocess.CompletedProcess:
    """Run the REPL with a task that raises, then ``/exit``."""
    env = dict(os.environ, KISS_HOME=str(tmp_path / ".kisshome"))
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        input="please do something\n/exit\n",
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )


def test_repl_survives_agent_run_exception(tmp_path: Path) -> None:
    """A raising ``agent.run`` must return the REPL to the prompt."""
    proc = _run_repl_with_boom_agent(tmp_path)
    assert "RUN_CALLED" in proc.stderr, proc.stderr
    assert proc.returncode == 0, (
        "agent.run raised and the whole REPL crashed instead of "
        f"returning to the prompt:\n{proc.stderr}"
    )
    assert "REPL_EXITED_CLEANLY" in proc.stdout
    # The failure must be reported to the user, not swallowed.
    assert "model API exploded" in proc.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
