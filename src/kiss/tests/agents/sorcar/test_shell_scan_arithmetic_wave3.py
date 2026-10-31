# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R2 finding 2: the command substitutions of an arithmetic expression are scanned.

The scanner reads ``$((...))`` as one word and ``((...))`` as a group
with no commands, so an arithmetic ``<<`` is a shift and opens no
here-document.  But it skipped the whole expression, while bash runs
the ``$(...)`` and backtick substitutions in it: a here-document
rewrite or a long sleep inside one passed every guard.  Their commands
are now scanned; what is not a substitution stays arithmetic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT, heredoc_guard, heredoc_targets
from kiss.agents.sorcar.shell_scan import bash_path, host_path, simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard
from kiss.tests.conftest import posix_only

HEREDOCS = [
    # The review's command, and the same in an arithmetic command.
    "echo $(( $(cat <<'EOF' > notes.md\nhi\nEOF\necho 1) + 1 ))",
    "(( $(cat <<'EOF' > notes.md\nhi\nEOF\necho 1) + 1 ))",
    # A backtick substitution, one in double quotes, one in a nested expression.
    "x=$(( `cat <<'EOF' > notes.md\nhi\nEOF\necho 1` + 1 ))",
    "echo $(( \"$(cat <<'EOF' > notes.md\nhi\nEOF\necho 1)\" + 1 ))",
    "echo $(( $(( $(cat <<'EOF' > notes.md\nhi\nEOF\necho 1) )) << 2 ))",
    "for (( i = $(cat <<'EOF' > notes.md\nhi\nEOF\necho 0); i < 1; i++ )); do :; done",
]

SLEEPS = [
    "echo $(( $(sleep 60; echo 1) + 1 ))",
    "(( $(sleep 60; echo 1) ))",
    "x=$(( `sleep 60` + 1 ))",
    "echo $(( 1 << $(sleep 60; echo 2) ))",
    "echo $(( $(( $(sleep 60) + 1 )) * 2 ))",
    # A parenthesis in quotes is not the end of the expression (bash skips quotes too).
    "echo $(( $(echo '))'; sleep 60) + 1 ))",
]


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A work dir holding an existing ``notes.md``."""
    (tmp_path / "notes.md").write_text("old\n")
    return tmp_path


def _host_lexists(path: str) -> bool:
    """Whether the bash-spelt *path* exists on the host (as the Bash tool asks)."""
    return os.path.lexists(host_path(path))


@pytest.mark.parametrize("command", HEREDOCS)
def test_a_here_document_rewrite_in_an_arithmetic_substitution_is_refused(
    work: Path, command: str,
) -> None:
    # The guard takes and names paths the way bash spells them (``/c/...``
    # on Windows), as the Bash tool hands them over.
    assert heredoc_guard(command, bash_path(str(work)), _host_lexists) == HEREDOC_VERDICT.format(
        path=bash_path(str(work / "notes.md")))
    assert heredoc_targets(command) == ["notes.md"]


@pytest.mark.parametrize("command", SLEEPS)
def test_a_long_sleep_in_an_arithmetic_substitution_is_refused(command: str) -> None:
    refusal = sleep_guard(command)
    assert refusal is not None, command
    assert refusal.startswith("REFUSED: the Bash call did NOT run: a foreground sleep of 60s ")


def test_arithmetic_without_substitutions_holds_no_commands() -> None:
    assert [c.program() for c in simple_commands("echo $(( 1 << 2 )); (( x <<= 1 ))", None)] == [
        "echo"]
    # ``$((`` closing as ``) )`` is a substitution starting with a subshell:
    # its commands are scanned once, not again for the expression it is not.
    commands = simple_commands("x=$(( $(sleep 60) ) )", None)
    assert [c.program() for c in commands].count("sleep") == 1


def test_nested_expressions_that_do_not_close_are_each_tried_once() -> None:
    """Retrying the inner ``$((`` for every enclosing one doubled the work per level."""
    command = "sleep 60"
    for _ in range(22):
        command = f"$(( {command} ) )"
    started = time.monotonic()
    assert sleep_guard("x=" + command) is not None
    assert time.monotonic() - started < 5


@posix_only("the host Bash tool runs bash")
def test_the_bash_tool_refuses_the_rewrite_and_runs_plain_arithmetic(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    assert tools.Bash(HEREDOCS[0], "rewrite") == HEREDOC_VERDICT.format(path=work / "notes.md")
    assert (work / "notes.md").read_text() == "old\n"
    assert tools.Bash("echo $(( $(echo 3) << 1 )) $(( `echo 2` * 5 ))", "arithmetic") == "6 10\n"
