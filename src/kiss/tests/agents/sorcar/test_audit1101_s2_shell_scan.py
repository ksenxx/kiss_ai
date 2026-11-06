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

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
)
from kiss.agents.sorcar.shell_scan import (
    MAX_NESTING,
    TOO_DEEP_VERDICT,
    TooDeepError,
    simple_commands,
)
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


def test_a_heredoc_delimiter_is_not_expanded() -> None:
    """The shell removes quotes from a delimiter but expands nothing in it (merge review
    2026-11-04): a substitution in it is text of the delimiter, not a command of the line,
    and an arithmetic assignment in it does not happen."""
    for delimiter in ("$(echo EOF)", '"$(echo EOF)"', "`echo EOF`", "$((i=5))", "${i:-$(date)}"):
        command = f"i=1; cat <<{delimiter} > tmp/x\nbody\n{delimiter.strip(chr(34))}\necho $i"
        assert _programs(command) == ["", "cat", "echo"], delimiter  # the assignment, no date
        cat, echo = simple_commands(command, "/w")[1:]
        assert cat.heredoc_body == "body\n" and [w.text for w in cat.outputs] == ["tmp/x"]
        assert cat.redirections[0][2].text == delimiter.strip('"')
        assert [w.text for w in echo.argv()] == ["echo", "1"], delimiter
    # ...while the body after an unquoted delimiter is expanded, and scanned
    assert _programs("cat <<EOF\n$(date)\nEOF") == ["cat", "date"]
    assert sleep_guard("cat <<$(sleep 60)\nbody\n$(sleep 60)") is None
    assert sleep_guard("cat <<EOF\n$(sleep 60)\nEOF") is not None


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


def test_a_scan_past_the_nesting_limit_refuses_rather_than_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-05 A09-1 (astra): the commands past ``MAX_NESTING`` levels are unread,
    so a guard's scan raises :class:`TooDeepError` and the Bash tool runs nothing, instead of
    returning the commands before the nesting and letting the rewrite after it through.  The
    gate's optimistic reading ends with the commands found so far, as before."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    target = tmp_path / "tracked.txt"
    target.write_text("ORIGINAL\n")
    rewrite = "; cat > tracked.txt <<'EOF'\nREPLACEMENT\nEOF"
    command = "( " * (MAX_NESTING + 1) + ": " + ") " * (MAX_NESTING + 1) + rewrite
    with pytest.raises(TooDeepError) as raised:
        simple_commands(command, "/w")
    assert str(raised.value) == TOO_DEEP_VERDICT
    assert TOO_DEEP_VERDICT.startswith("REFUSED: the command did not run: ")
    with pytest.raises(TooDeepError):
        heredoc_guard(command, "/w", lambda path: True)
    with pytest.raises(TooDeepError):
        UsefulTools(work_dir=str(tmp_path)).Bash(command, "too deep", timeout_seconds=5)
    assert target.read_text() == "ORIGINAL\n"
    assert simple_commands(command, "/w", optimistic=True) == []
    # At the limit the whole command is read, and the rewrite in it is refused.
    within = "( " * MAX_NESTING + ": " + ") " * MAX_NESTING + rewrite
    assert _programs(within) == [":", "cat"]
    assert heredoc_guard(within, "/w", lambda path: True) == HEREDOC_VERDICT.format(
        path="/w/tracked.txt")


def test_nested_expansions_count_toward_the_nesting_limit() -> None:
    """Audit 2026-11-05 A09-7 (astra): ``${x:-${x:-...}}`` 600 deep raised ``RecursionError``
    out of the scanner, since only groups counted toward ``MAX_NESTING``; every nested
    expansion counts now, so a scan ends with the limit's own error, and ``MAX_NESTING``
    levels of any mix are read whole."""
    nested = "echo " + "${S2_UNSET:-" * 600 + "x" + "}" * 600
    with pytest.raises(TooDeepError):
        simple_commands(nested, "/w")
    assert [c.program() for c in simple_commands(nested, "/w", optimistic=True)] == ["echo"]
    unterminated = "echo " + "${S2_UNSET:-" * MAX_NESTING  # ends with the text, as before
    assert _programs(unterminated) == ["echo"]
    with pytest.raises(TooDeepError):
        simple_commands(unterminated + "${S2_UNSET:-", "/w")
    within = "echo " + "${S2_UNSET:-" * MAX_NESTING + "x" + "}" * MAX_NESTING + " tail"
    assert _bash(within).split() == ["x", "tail"]
    [simple] = simple_commands(within, "/w")
    assert [word.text for word in simple.words] == ["echo", within[5:-5], "tail"]
    assert not simple.words[1].literal
    half = MAX_NESTING // 2
    mixed = "echo " + '"${S2_UNSET:-$(echo ' * half + "x" + ')}"' * half
    assert _bash(mixed).split() == ["x"]
    assert _programs(mixed) == ["echo"] * (half + 1)
    with pytest.raises(TooDeepError):
        simple_commands("echo " + '"${S2_UNSET:-$(echo ' * half + "${x:-x}", "/w")


def test_a_cd_after_and_is_not_trusted_past_its_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-05 A09-2 (astra): in ``false && cd tmp; cat > f <<EOF`` the ``cd`` runs
    only when ``false`` succeeds, so past the ``;`` the directory is unknown and the guard
    refuses the copy, rather than check ``tmp/f`` while bash writes ``f``; the rest of the
    and-list still knows the directory, as the gate's optimistic reading does throughout."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    body = "<<'EOF'\nREPLACEMENT\nEOF"
    command = f"false && cd tmp; cat > tracked.txt {body}"
    assert [c.cwd for c in simple_commands(command, "/w")] == ["/w", "/w", None]
    assert [c.cwd for c in simple_commands(command, "/w", optimistic=True)] == [
        "/w", "/w", "/w/tmp"]
    unresolved = HEREDOC_UNRESOLVED_VERDICT.format(path="tracked.txt")
    assert heredoc_guard(command, "/w", lambda path: False) == unresolved
    for text, cwd in (
        (f"false && cd tmp\ncat > f {body}", None),
        (f"false && pushd tmp; cat > f {body}", None),
        (f"false && cd tmp || cat > f {body}", None),
        (f"false && {{ cd tmp; }}; cat > f {body}", None),
        (f"cd tmp && cd sub; cat > f {body}", None),
        (f"false && cd tmp && cat > f {body}", "/w/tmp"),
        (f"false && cd tmp && {{ cat > f {body}\n}}", "/w/tmp"),
        (f"cd tmp && cat > f {body}", "/w/tmp"),
        (f"cd tmp && false; cat > f {body}", "/w/tmp"),
        (f"false && (cd tmp); cat > f {body}", "/w"),
        (f"false && cd tmp & cat > f {body}", "/w"),
        (f"false && echo; cat > f {body}", "/w"),
    ):
        assert next(c.cwd for c in simple_commands(text, "/w") if c.program() == "cat") == cwd, text
    target = tmp_path / "tracked.txt"
    target.write_text("ORIGINAL\n")
    result = UsefulTools(work_dir=str(tmp_path)).Bash(command, "conditional cd", timeout_seconds=5)
    assert result == unresolved
    assert target.read_text() == "ORIGINAL\n"
