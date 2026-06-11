# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: a failing slash command must not kill the REPL.

Bug: ``cli_repl.run_repl`` guards task execution (``_run_one`` catches
every ``Exception`` so "a failing task must not kill the interactive
session"), but the ``_handle_slash`` call sits bare inside the loop.
Any exception raised while handling a ``/`` command — e.g. ``/resume``
listing recent chats from a corrupt ``sorcar.db`` raises
``sqlite3.DatabaseError: file is not a database`` — escaped the loop
and terminated the whole interactive session with a traceback, instead
of reporting the error and returning to the prompt.

The test runs the real ``run_repl`` in a child interpreter whose
``KISS_HOME`` contains a corrupted ``sorcar.db``, sends ``/resume``
followed by ``/help`` and ``exit``, and asserts the REPL survived the
command failure (the ``/help`` output appears and the process exits
cleanly).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_slash_command_error_returns_to_prompt(tmp_path: Path) -> None:
    """``/resume`` over a corrupt DB must not terminate the session."""
    kiss_home = tmp_path / ".kisshome"
    kiss_home.mkdir()
    # A file that is definitely not a SQLite database: the first query
    # against it raises sqlite3.DatabaseError("file is not a database").
    (kiss_home / "sorcar.db").write_bytes(b"this is not a sqlite db\n" * 64)
    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    child_code = f"""
import os
os.environ["KISS_HOME"] = {str(kiss_home)!r}
import sys
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_repl import run_repl

agent = ChatSorcarAgent("Stateful Sorcar Agent")
run_repl(agent, {{"work_dir": {str(work_dir)!r},
                  "model_name": "demo-model", "verbose": True}})
print("REPL_EXITED_CLEANLY")
"""

    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        input="/resume\n/help\nexit\n",
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert "Commands:" in proc.stdout, (
        "the REPL died on the failing /resume command instead of "
        "reporting the error and returning to the prompt — /help was "
        f"never processed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "REPL_EXITED_CLEANLY" in proc.stdout, (
        f"run_repl did not return cleanly.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        f"child exited with {proc.returncode}.\nstderr:\n{proc.stderr}"
    )
