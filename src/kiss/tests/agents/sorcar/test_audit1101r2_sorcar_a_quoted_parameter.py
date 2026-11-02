# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 rounds 2 and 3 (sorcar_a F3, N1): quotes in the operands of ``${...}``.

Bash reads the word of ``${x:-word}`` (``:=``, ``:+``, and without the
colon) by the quoting of the expansion: in a double-quoted string (and
an unquoted here-document body) its single quotes are characters and
the substitutions between them run; outside double quotes, and in the
message of ``${x:?word}``, in a pattern or a replacement (``${x#pat}``,
``${x/pat/rep}``) and in their nested expansions, they quote.  The
offset and length of ``${x:off:len}`` are read double-quoted even
outside double quotes.  An ANSI-C ``$'...'`` in a double-quoted
``${...}`` is expanded by the parser into a double-quoted string (a
single-quoted one in a pattern), so its substitutions run too.  The
round-1 scanner skipped every ``'...'`` in a ``${...}``, so a command
in the word escaped every scanner-based guard (the here-document
rewrite guard, the sleep guard); the round-2 scanner read the single
quotes of every operand as characters, so the guards refused harmless
commands.  Real bash is the oracle: a marker file tells whether it
ran the substitution, and the scanner must list the command exactly
then.

The bash versions the product runs do not agree everywhere: the macOS
system bash is 3.2, Linux and Git Bash have 5.x.  Where they differ
(:data:`BASH_32_RUNS`) a substitution the scanner misses is a guard
bypass on one of them, so the scanner must list it whenever one of the
versions runs it, and the oracle is checked against the version at
hand.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT, heredoc_guard
from kiss.agents.sorcar.shell_scan import simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.conftest import posix_only

pytestmark = posix_only("bash is the oracle")

#: ``{m}`` stands for the marker-touching command substitution.
CASES = (
    'unset U; echo "${U:-{m}}"',
    "unset U; echo \"${U:-'{m}'}\"",
    "unset U; echo \"${U:='{m}'}\"",
    "U=set; echo \"${U:+'{m}'}\"",
    "unset U; echo \"${U:-\"'{m}'\"}\"",
    "unset U; echo \"${U:-${V:-'{m}'}}\"",
    "unset U; echo \"${U:-a/b'{m}'}\"",
    "unset U; echo \"${U:-'\\{m}'}\"",
    "unset U; echo ${U:-'{m}'}",
    "U=abc; echo \"${U#'{m}'}\"",
    "U=abc; echo \"${U/b/'{m}'}\"",
    "echo \"$'{m}'\"",
    "unset U; cat <<EOF\n${U:-'{m}'}\nEOF",
    "unset U; cat <<EOF\n${U:-{m}}\nEOF",
    "unset U; cat <<'EOF'\n${U:-'{m}'}\nEOF",
    # Round 3 (N1): the message of ``?``, the nested expansions of a pattern
    # or a replacement, the offset of a substring, and ``$'...'``.
    "unset U; echo \"${U:?'{m}'}\"",
    "unset U; echo \"${U?'{m}'}\"",
    "unset U; echo \"${U:?\"'{m}'\"}\"",
    "unset U; echo \"${U:?${V:-'{m}'}}\"",
    "unset U; echo \"${U:?$'{m}'}\"",
    "unset U; echo ${U:?'{m}'}",
    "unset U; echo ${U:?$'{m}'}",
    "unset U; echo \"${U-'{m}'}\"",
    "unset U; echo \"${U='{m}'}\"",
    "U=set; echo \"${U+'{m}'}\"",
    "unset U; echo \"${U-?'{m}'}\"",
    "unset U; echo \"${U:-$'{m}'}\"",
    "unset U; echo ${U:-$'{m}'}",
    "unset U; V=aaa; echo \"${U:-${V/a/'{m}'}}\"",
    "unset U; echo \"${U:-${V:?'{m}'}}\"",
    "U=aaa; echo \"${U/a/${V:-'{m}'}}\"",
    "U=aaa; echo \"${U#${V:-'{m}'}}\"",
    "U=aaa; echo \"${U%%${V:-'{m}'}}\"",
    "U=aaa; echo \"${U^${V:-'{m}'}}\"",
    "U=aaa; echo \"${U,,${V:-'{m}'}}\"",
    "U=aaa; echo \"${U/${V:-'{m}'}}\"",
    "U=aaa; echo \"${U/a/${V:-${W:-'{m}'}}}\"",
    "U=aaa; echo \"${U/a/\"${V:-'{m}'}\"}\"",
    "U=aaa; echo \"${U#$'{m}'}\"",
    "U=aaa; echo \"${U/a/$'{m}'}\"",
    "U=aaa; echo \"${U#${V:-$'{m}'}}\"",
    "U=aaa; echo \"${U#${V:?$'{m}'}}\"",
    "U=aaa; echo ${U#${V:-'{m}'}}",
    "U=aaa; echo \"${U:'{m}'}\"",
    "U=aaa; echo ${U:'{m}'}",
    "U=aaa; echo ${U:0:'{m}'}",
    "U=aaa; echo ${U: -1:'{m}'}",
    "U=aaa; echo ${U:${V:-'{m}'}}",
    "U=aaa; echo ${U:0:$'{m}'}",
    "U=aaa; echo \"${U:$'{m}'}\"",
    # Round 3 review: a special parameter (``?``, ``#``, ``-``, ``$``, ``!``,
    # ``@``, ``*``, a position) or an indirection opens the expansion; its
    # character is not the operator.
    "set -- a b; : \"${?:+'{m}'}\"",
    ": \"${?+'{m}'}\"",
    ": \"${?:+$'{m}'}\"",
    ": ${?:+'{m}'}",
    ": \"${?:?'{m}'}\"",
    ": \"${#:+'{m}'}\"",
    ": \"${#+'{m}'}\"",
    ": \"${-:+'{m}'}\"",
    ": \"${$:+'{m}'}\"",
    "true & wait; : \"${!:+'{m}'}\"",
    "set --; : \"${@:-'{m}'}\"",
    "set -- a; : \"${*:+'{m}'}\"",
    "set -- a b c d e f g h i j; : \"${10:+'{m}'}\"",
    ": \"${0:+'{m}'}\"",
    "U=V; V=set; : \"${!U:+'{m}'}\"",
    "U=V; unset V; : \"${!U:-'{m}'}\"",
    "U=abc; : \"${#U}${U:+'{m}'}\"",
    "set -- a; : \"${#}${1:+'{m}'}\"",
    # Round 4 review: ``${##pat}`` strips from ``$#``, only ``${##}`` measures it.
    "set -- a b; : \"${##'{m}'}\"",
    "set -- a b; : \"${#%'{m}'}\"",
    "set -- a b; : \"${#/2/'{m}'}\"",
    "set -- a b; : \"${#?'{m}'}\"",
    "set -- a; : \"${##}${1:+'{m}'}\"",
    "A=(x y); : \"${#A[@]}${A[0]:+'{m}'}\"",
)

#: The cases bash 3.2 and bash 5.2 read differently, with whether bash 3.2
#: runs the substitution (bash 5.2 does the opposite).  Bash 3.2 reads the
#: replacement of a double-quoted ``${x/pat/rep}`` by the double-quote rules,
#: so its single quotes are characters and the substitution between them
#: runs (bash 4.3 started removing the replacement's quotes, so they quote;
#: ``shopt -s compat42`` restores the old reading); it inserts the text of an
#: ANSI-C ``$'...'`` unquoted into a double-quoted ``${...}``, so in a pattern
#: or a replacement the substitution runs (bash 4.3 single-quotes it in a
#: ``${x/pat/rep}``, bash 5.2 in every pattern); and it cuts the offset of
#: ``${x:${y:-off}}`` at the colon of the nested expansion ("bad
#: substitution"), so nothing in it runs.  The versions in between changed
#: these readings one at a time; the test pins the two the product runs.
BASH_32_RUNS = {
    "U=abc; echo \"${U/b/'{m}'}\"": True,
    "unset U; V=aaa; echo \"${U:-${V/a/'{m}'}}\"": True,
    "U=aaa; echo \"${U/a/${V:-'{m}'}}\"": True,
    "U=aaa; echo \"${U/a/${V:-${W:-'{m}'}}}\"": True,
    "U=aaa; echo \"${U#$'{m}'}\"": True,
    "U=aaa; echo \"${U/a/$'{m}'}\"": True,
    "U=aaa; echo ${U:${V:-'{m}'}}": False,
    "set -- a b; : \"${#/2/'{m}'}\"": True,
}


def _bash_version() -> tuple[int, int]:
    """The major and minor version of the bash that runs the cases: 3.2 on macOS, 5.2 on Linux."""
    output = subprocess.run(
        ["bash", "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    major, minor = output.split()
    return int(major), int(minor)


def _bash_runs(command: str, tmp_path: Path) -> bool:
    """Whether bash runs the ``{m}`` substitution of *command*: it touches a marker."""
    marker = tmp_path / "marker"
    marker.unlink(missing_ok=True)
    env = {key: value for key, value in os.environ.items() if key not in ("U", "V")}
    subprocess.run(
        ["bash", "-c", command.replace("{m}", f"$(touch {marker})")],
        capture_output=True, check=False, timeout=10, env=env,
    )
    return marker.exists()


def _scanned(command: str) -> bool:
    """Whether the scanner lists the ``{m}`` substitution's ``touch`` among the commands."""
    programs = [simple.program() for simple in simple_commands(command, "/w")]
    return "touch" in programs


@pytest.mark.parametrize("case", CASES)
def test_the_scanner_sees_a_substitution_exactly_when_bash_runs_it(
    case: str, tmp_path: Path,
) -> None:
    command = case.replace("{m}", "$(touch marker)")
    runs = _bash_runs(case, tmp_path)
    if case not in BASH_32_RUNS:
        assert _scanned(command) == runs, command
        return
    # One of the two versions runs it, so the scanner lists it on both; bash 3.2
    # and bash 5.2 run it as documented (the versions between are not pinned).
    assert _scanned(command), command
    version = _bash_version()
    if version < (4, 0):
        assert runs == BASH_32_RUNS[case], command
    elif version >= (5, 2):
        assert runs != BASH_32_RUNS[case], command


def test_the_version_dependent_cases_are_cases() -> None:
    assert BASH_32_RUNS.keys() <= set(CASES)


def test_the_end_of_the_expansion_is_still_found_as_bash_finds_it() -> None:
    """The literal quotes are still skipped whole to the closing brace; words follow."""
    for command, inner in (
        ("echo \"${U:-'}'}\" tail", []),
        ("echo \"${U:-'$(echo })'}\" tail", ["echo"]),
    ):
        outer, *rest = simple_commands(command, "/w")
        assert [simple.program() for simple in rest] == inner, command
        assert [word.text for word in outer.words][-1] == "tail", command
        assert len(outer.words) == 3, command


def test_the_guards_see_the_command_hidden_in_the_quoted_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    target = tmp_path / "f"
    target.write_text("ORIGINAL\n")
    command = "unset U; : \"${U:-'$(cat > f <<'EOF'\nREPLACEMENT\nEOF\n)'}\""
    assert heredoc_guard(command, str(tmp_path), os.path.lexists) == HEREDOC_VERDICT.format(
        path=target,
    )
    result = UsefulTools(work_dir=str(tmp_path)).Bash(command, "hidden rewrite", timeout_seconds=5)
    assert result == HEREDOC_VERDICT.format(path=target)
    assert target.read_text() == "ORIGINAL\n"
    assert sleep_guard("echo \"${U:-'$(sleep 60)'}\"") is not None
    assert sleep_guard("echo \"${U:-'$(sleep 1)'}\"") is None
    assert sleep_guard("echo ${U:-'$(sleep 60)'}") is None
    # The special parameter ``?`` is not the ``?`` operator: the word runs.
    assert sleep_guard(": \"${?:+'$(sleep 60)'}\"") is not None
    assert heredoc_guard(
        ": \"${?:+'$(cat > f <<EOF\nREPLACEMENT\nEOF\n)'}\"", str(tmp_path), os.path.lexists,
    ) == HEREDOC_VERDICT.format(path=target)
    # Bash 3.2 reads the replacement of a double-quoted ``${x/pat/rep}`` by the
    # double-quote rules and runs the sleep: refused on every version.
    assert sleep_guard("U=aaa; unset V; : \"${U/a/${V:-'$(sleep 60)'}}\"") is not None


def test_the_guards_let_a_quoted_operand_that_bash_does_not_run_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 3 (N1): single quotes quote in a ``:?`` message and in a pattern's nested word.

    The pattern's nested default value holds a here-document that would
    rewrite an existing file; the shell never runs it (``/bin/sh`` and
    bash agree on this one), so the guard lets the command run and the
    file is untouched.
    """
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    target = tmp_path / "f"
    target.write_text("ORIGINAL\n")
    command = "U=aaa; unset V; : \"${U#${V:-'$(cat > f <<EOF\nREPLACEMENT\nEOF\n)'}}\"; echo ran"
    assert heredoc_guard(command, str(tmp_path), os.path.lexists) is None
    result = UsefulTools(work_dir=str(tmp_path)).Bash(command, "literal pattern", timeout_seconds=5)
    assert "ran" in result and "REFUSED" not in result, result
    assert target.read_text() == "ORIGINAL\n"
    assert sleep_guard("unset U; : \"${U:?'$(sleep 60)'}\"") is None
    assert sleep_guard("U=aaa; unset V; : \"${U#${V:-'$(sleep 60)'}}\"") is None
    # Bash runs these: the parser double-quotes the ANSI-C string, the offset is arithmetic.
    assert sleep_guard("unset U; : \"${U:?$'$(sleep 60)'}\"") is not None
    assert sleep_guard("U=aaa; : ${U:'$(sleep 60)'}") is not None
