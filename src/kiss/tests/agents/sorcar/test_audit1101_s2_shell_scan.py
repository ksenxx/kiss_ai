# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F10: the scanner sees the commands inside a parameter expansion.

``${x:-$(...)}`` runs its command substitution when ``x`` is unset, so a
here-document rewrite, a ``sleep`` or any other command hidden there must
reach every scanner-based guard.  The scanner skipped ``${...}`` to the
first ``}``, missing the commands and, with a ``}`` inside quotes or a
nested expansion, the end of the word too.  Real bash settles what each
construct expands to.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT, heredoc_guard, heredoc_targets
from kiss.agents.sorcar.shell_scan import simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.conftest import posix_only

pytestmark = posix_only("bash is the oracle")


def _programs(command: str) -> list[str]:
    """The programs of the simple commands of *command*, in order."""
    return [simple.program() for simple in simple_commands(command, "/w")]


def _bash(command: str) -> str:
    """What bash prints for *command*, with ``S2_UNSET`` unset."""
    env = {key: value for key, value in os.environ.items() if key != "S2_UNSET"}
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True,
                          timeout=10, env=env).stdout


def _bash_version() -> tuple[int, int]:
    """The major and minor version of that bash: 3.2 on macOS, 5.2 on Linux."""
    major, minor = _bash('echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"').split()
    return int(major), int(minor)


def test_substitutions_inside_parameter_expansions_are_scanned() -> None:
    assert _programs('echo "${S2_UNSET:-$(date)}"') == ["echo", "date"]
    assert _programs("echo ${S2_UNSET:=`date`}") == ["echo", "date"]
    assert _programs('echo "${x:+$(a) $(b)}" ${y#$(c)} ${z/$(d)/$(e)}') == [
        "echo", "a", "b", "c", "d", "e"]
    assert _programs("echo ${x:-${y:-$(date)}}") == ["echo", "date"]
    command = "unset S2_UNSET; : \"${S2_UNSET:-$(cat > f <<'EOF'\nREPLACEMENT\nEOF\n)}\""
    assert _programs(command) == ["unset", ":", "cat"]
    assert heredoc_targets(command) == ["f"]
    assert heredoc_guard(command, "/w", lambda path: path == "/w/f") == HEREDOC_VERDICT.format(
        path="/w/f")
    assert heredoc_guard(command, "/w", lambda path: False) is None
    assert sleep_guard('echo "${S2_UNSET:-$(sleep 60)}"') is not None
    assert sleep_guard('echo "${S2_UNSET:-$(sleep 1)}"') is None


def test_the_end_of_a_parameter_expansion_is_found_as_bash_finds_it() -> None:
    # A ``}`` in quotes or escaped does not end it; a nested ``${`` closes with its own brace;
    # a bare ``{`` does not nest.  The words after the expansion are read as words.  Bash 4.2
    # started removing the backslash of the escaped brace; the macOS system bash (3.2) keeps it.
    escaped_brace = "}" if _bash_version() >= (4, 2) else "\\}"
    for command, words in (
        ("echo ${S2_UNSET:-'}'} tail", ["}", "tail"]),
        ('echo "${S2_UNSET:-\'}\'}" tail', ["'}'", "tail"]),
        ('echo ${S2_UNSET:-"}"} tail', ["}", "tail"]),
        ('echo "${S2_UNSET:-\\}}" tail', [escaped_brace, "tail"]),
        ("echo ${S2_UNSET:-{a}} tail", ["{a}", "tail"]),
        ("echo ${S2_UNSET:-${S2_UNSET:-z}} tail", ["z", "tail"]),
        ("echo ${#S2_UNSET} tail", ["0", "tail"]),
        ("echo ${S2_UNSET:-$((1+2))} tail", ["3", "tail"]),
    ):
        assert _bash(command).split() == words, command
        [simple] = simple_commands(command, "/w")
        assert [word.text for word in simple.words][-1] == "tail", command
        assert len(simple.words) == 3 and not simple.words[1].literal, command
    # An unterminated expansion ends with the text; the commands in it are still scanned.
    assert _programs("echo ${x:-$(date") == ["echo", "date"]
    assert _programs("echo ${x:-'unterminated") == ["echo"]


def test_a_heredoc_rewrite_hidden_in_a_parameter_expansion_is_refused_by_the_bash_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    target = tmp_path / "f"
    target.write_text("ORIGINAL\n")
    command = "unset S2_UNSET; : \"${S2_UNSET:-$(cat > f <<'EOF'\nREPLACEMENT\nEOF\n)}\""
    result = UsefulTools(work_dir=str(tmp_path)).Bash(command, "hidden rewrite", timeout_seconds=5)
    assert result == HEREDOC_VERDICT.format(path=target)
    assert target.read_text() == "ORIGINAL\n"
