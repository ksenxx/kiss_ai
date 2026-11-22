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

import functools
import json
import os
import shutil
import signal
import subprocess
import time
import timeit
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNKNOWN_VERDICT,
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
    rewrite_guard,
)
from kiss.agents.sorcar.shell_scan import (
    MAX_NESTING,
    TOO_DEEP_VERDICT,
    UNREAD_DELIMITER_VERDICT,
    UNREAD_VALUES_VERDICT,
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


@functools.cache
def _bash_version() -> tuple[int, int]:
    """The major and minor version of that bash: 3.2 on macOS, 5.2 on Linux."""
    major, minor = _bash('echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"').split()
    return int(major), int(minor)


#: The bash a construct came with, by a text of the commands below that only it has (each
#: measured with bash 3.2 to 5.3, full run 2026-11-22): ``coproc``, ``mapfile`` and the
#: handler of a missing program; a descriptor bash numbers and ``printf -v`` to an element;
#: a name reference, ``-v`` of an element and an arithmetic list a failed assignment ends;
#: the prompt operator; the evaluation of what ``RANDOM``, ``SRANDOM`` and ``SECONDS`` are
#: assigned.
_SINCE = (
    ("coproc ", (4, 0)), ("mapfile ", (4, 0)), ("command_not_found_handle", (4, 0)),
    ("exec {FD}", (4, 1)), ('printf -v "$X"', (4, 1)),
    ("declare -n ", (4, 3)), ('test -v "$X"', (4, 3)), ("[[ -v $X ]]", (4, 3)),
    ("readonly A; ((A = 0", (4, 3)), ("${X@P}", (4, 4)),
    ("RANDOM=", (5, 1)), ("SECONDS=", (5, 1)), ("for RANDOM in", (5, 1)),
)


def _oracle_runs(command: str) -> bool:
    """Whether the bash of this machine runs *command* as the bash the scan reads it for.

    The macOS system bash (3.2) lacks the constructs of :data:`_SINCE`, and stock macOS
    the ``setsid`` of util-linux: what bash rewrites there is no oracle for the scan,
    whose verdict on the command is checked on every machine.
    """
    if "setsid " in command and shutil.which("setsid") is None:
        return False
    return all(_bash_version() >= since for mark, since in _SINCE if mark in command)


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
    legacy = 'printf "%s\\n" ' + "$[" * 1200 + "1" + "]" * 1200  # the older arithmetic nests too
    with pytest.raises(TooDeepError):
        simple_commands(legacy, "/w")
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


_OWNED = "<<'EOF'\nowned\nEOF\n"
_UNRESOLVED = HEREDOC_UNRESOLVED_VERDICT.format


def _default_sigint() -> None:
    """Give the bash that is started next the default SIGINT.

    An ignored SIGINT survives exec (a suite launched as ``( pytest & )`` ignores it), and
    bash cannot trap a signal ignored at entry: its ``trap 'cd sub' INT`` would never run.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _overwritten(tmp_path: Path, command: str, files: list[str]) -> set[str]:
    """The *files* (relative, holding ORIGINAL) that bash, run in *tmp_path*, rewrites.

    ``HOME`` is ``tmp_path/home``, so a ``~`` target is under it.
    """
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("ORIGINAL\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, check=False,
                   timeout=10, env=env, preexec_fn=_default_sigint)
    return {name for name in files if (tmp_path / name).read_text() != "ORIGINAL\n"}


def _verdict(command: str, existing: set[str]) -> str | None:
    """The heredoc guard's verdict on *command* run in ``/w``, where *existing* exist."""
    paths = {f"/w/{name}" for name in existing}
    return heredoc_guard(command, "/w", lambda path: path in paths)


def test_a_comment_after_a_list_operator_does_not_end_the_list(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-2: bash skips a comment and its newline after ``&&``, ``||``
    or ``|``, so ``true || # why\\ncd sub`` is ``true || cd sub``, and the ``cat`` after it runs
    in the start directory; the scanner ended the list at the newline, took the ``cd`` for
    unconditional and checked ``sub/existing`` while bash rewrote ``existing``.  After a pipe,
    the assignment on the next line is the pipeline's subshell's."""
    for command in (f"true || # why\ncd sub\ncat > existing {_OWNED}",
                    f"false && # why\ncd sub\ncat > existing {_OWNED}"):
        assert _overwritten(tmp_path, command, ["existing", "sub/.keep"]) == {"existing"}
        assert [c.cwd for c in simple_commands(command, "/w")] == ["/w", "/w", None], command
        assert _verdict(command, {"existing"}) == _UNRESOLVED(path="existing")
    command = f"true | # c\nT=sub/; cat > ${{T}}existing {_OWNED}"
    assert _overwritten(tmp_path, command, ["existing", "sub/existing"]) == {"existing"}
    assert _verdict(command, {"existing"}) == _UNRESOLVED(path="${T}existing")
    # the list goes on past the comment: the pipe's reader is the command on the next line,
    # and the here-document's body follows the comment's line
    piped = "cat <<EOF | # c\nbody\nEOF\nwc -l"
    assert _bash(piped).strip() == "1"  # BSD wc pads its count
    cat, wc = simple_commands(piped, "/w")
    assert cat.reader is wc and wc.piped and cat.heredoc_body == "body\n"
    assert _programs("true && # c\n\n  # d\ncd sub && cat > f") == ["true", "cd", "cat"]
    assert simple_commands("true && # c\ncd sub && cat > f", "/w")[-1].cwd == "/w/sub"


def test_a_tilde_in_an_assigned_value_is_the_home_directory(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-3: bash expands a ``~`` after the ``=`` of an assignment
    (``T=~/x``, ``export T=~/x``) or a ``:`` in its value (``P=~/a:~/b``), so ``$T/f`` names a
    file under ``$HOME``; the scanner kept ``~/x`` as a literal path under the directory.  A
    quoted or escaped tilde is a character of the value."""
    for command in (f"T=~/x; cat > $T/f {_OWNED}", f"export T=~/x; cat > $T/f {_OWNED}",
                    f"T=a:~/x; cat > $T/f {_OWNED}", f"echo T=~/x; T=a:~/x; cat > $T/f {_OWNED}"):
        assert _verdict(command, set()) == _UNRESOLVED(path="$T/f"), command
    assert _overwritten(tmp_path, f"T=~/x; cat > $T/f {_OWNED}", ["home/x/f"]) == {"home/x/f"}
    assert _overwritten(tmp_path, f"T=a:~/x; cat > $T/f {_OWNED}", ["a:/x/f"]) == set()
    for command in (f'T="~/x"; cat > $T/f {_OWNED}', f"T=\\~/x; cat > $T/f {_OWNED}",
                    f"T='a:'~/x; cat > $T/f {_OWNED}"):
        literal = command.split(";")[0].split("=")[1].replace('"', "").replace("\\", "")
        literal = literal.replace("'", "")
        assert _overwritten(tmp_path, command, [f"{literal}/f"]) == {f"{literal}/f"}, command
        assert _verdict(command, {f"{literal}/f"}) == HEREDOC_VERDICT.format(
            path=f"/w/{literal}/f"), command


def test_a_function_body_runs_only_when_the_function_is_called(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-4: ``f() { cd sub; }`` defines ``f`` and runs nothing, so
    the ``cat`` after it writes ``existing`` in the start directory; the scanner read the body
    as a brace group that ran and moved to ``sub``.  ``function f { ... }`` was no definition
    at all: its ``cat`` was an argument of a command named ``function``.  The body's commands
    are read (the function may be called), its ``cd`` and assignments as ones that may not run.
    """
    for command in (f"f() {{ cd sub; }}\ncat > existing {_OWNED}",
                    f"function f {{ cd sub; }}\ncat > existing {_OWNED}",
                    f"function f () # c\n{{\n  cd sub\n}}\ncat > existing {_OWNED}",
                    f"my-fn() {{ cd sub; }}; cat > existing {_OWNED}",
                    f"f() if true; then cd sub; fi; cat > existing {_OWNED}"):
        assert _overwritten(tmp_path, command, ["existing", "sub/.keep"]) == {"existing"}, command
        assert simple_commands(command, "/w")[-1].cwd is None, command
        assert _verdict(command, {"existing"}) == _UNRESOLVED(path="existing"), command
    assigned = f"T=sub/; g() {{ T=.; }}; cat > ${{T}}existing {_OWNED}"
    assert _overwritten(tmp_path, assigned, ["existing", "sub/existing"]) == {"sub/existing"}
    assert _verdict(assigned, {"sub/existing"}) == _UNRESOLVED(path="${T}existing")
    # a subshell body keeps its directory to itself: the directory stays known
    subshell = f"f() ( cd sub ); cat > existing {_OWNED}"
    assert _overwritten(tmp_path, subshell, ["existing", "sub/.keep"]) == {"existing"}
    assert _verdict(subshell, {"existing"}) == HEREDOC_VERDICT.format(path="/w/existing")
    # the body's copy is refused when the function is called, in either form
    for command in (f"f() {{ cat > existing {_OWNED}}}; f",
                    f"function f {{ cat > existing {_OWNED}}}; f",
                    f"function f\n{{\ncat > existing {_OWNED}}}\nf"):
        assert _overwritten(tmp_path, command, ["existing"]) == {"existing"}, command
        assert _verdict(command, {"existing"}) == HEREDOC_VERDICT.format(path="/w/existing")
        assert _programs(command) == ["cat", "f"], command
    assert _programs("function") == [] and _programs("echo function f") == ["echo"]


def test_a_brace_expansion_names_several_files(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_astra-2: ``tee {src,other}/victim`` writes ``src/victim`` and
    ``other/victim`` (``v{1..3}`` three files); the scanner took the unquoted braces for a
    literal path.  Quoted or escaped braces, and braces without a comma or ``..``, are bash's
    characters."""
    command = f"tee {{src,other}}/victim {_OWNED}"
    assert _overwritten(tmp_path, command, ["src/victim", "other/victim"]) == {
        "src/victim", "other/victim"}
    assert _verdict(command, {"src/victim"}) == _UNRESOLVED(path="{src,other}/victim")
    sequence = f"tee v{{1..3}} {_OWNED}"  # a redirection to several words is ambiguous
    assert _overwritten(tmp_path, sequence, ["v2"]) == {"v2"}
    assert _verdict(sequence, {"v2"}) == _UNRESOLVED(path="v{1..3}")
    assert _verdict(f"cat > a{{b,{{c,d}}}}e {_OWNED}", set()) == _UNRESOLVED(path="a{b,{c,d}}e")
    for command, target in ((f"cat > 'v{{1,2}}' {_OWNED}", "v{1,2}"),
                            (f'cat > "v{{1..2}}" {_OWNED}', "v{1..2}"),
                            (f"cat > v\\{{1,2\\}} {_OWNED}", "v{1,2}"),
                            (f"cat > v{{1}} {_OWNED}", "v{1}"), (f"cat > {{}} {_OWNED}", "{}"),
                            (f"cat > v{{1.2}} {_OWNED}", "v{1.2}")):
        assert _overwritten(tmp_path, command, [target]) == {target}, command
        assert _verdict(command, {target}) == HEREDOC_VERDICT.format(path=f"/w/{target}")


def test_a_directory_changing_wrapper_moves_the_programs_operands(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_astra-6 / A13_fable-3: ``env -C src tee victim`` runs ``tee`` in
    ``src``, so its operand is ``src/victim``, while a redirection of the same command is the
    shell's and resolves in the shell's directory; the scanner dropped the option with the
    wrapper and checked ``victim``.  The operands of a program under such a wrapper (``env
    -C``, ``--chdir``, ``sudo -D``) are only the shell's to resolve.  BSD ``env`` (macOS)
    has ``-C`` but not GNU's ``--chdir``, so bash is the oracle of a long form only where
    ``env`` takes it."""
    gnu_env = subprocess.run(["env", "--chdir=/", "true"], capture_output=True).returncode == 0
    for command in (f"env -C src tee victim {_OWNED}", f"env --chdir=src tee victim {_OWNED}",
                    f"env -Csrc tee victim {_OWNED}", f"env --chdir src tee victim {_OWNED}",
                    f"env -i -C src /usr/bin/tee victim {_OWNED}"):
        if gnu_env or "--chdir" not in command:
            assert _overwritten(tmp_path, command, ["src/victim", "victim"]) == {"src/victim"}
        assert _verdict(command, {"src/victim"}) == _UNRESOLVED(path="victim"), command
    tee = simple_commands("sudo -D src tee victim", "/w")[0]
    assert [(w.text, w.literal) for w in tee.argv()] == [("tee", True), ("victim", False)]
    command = f"env -C src cat > victim {_OWNED}"
    assert _overwritten(tmp_path, command, ["src/victim", "victim"]) == {"victim"}
    assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")
    assert _verdict(f"env -u X tee victim {_OWNED}", {"victim"}) == HEREDOC_VERDICT.format(
        path="/w/victim")
    # Audit 2026-11-10 S10-F2: the literal ``-c`` text of a shell under such a wrapper is no
    # path: it is scanned, as commands in a directory only the shell knows, while the
    # redirection of the shell itself stays the caller's.
    child = "env -C src bash -c 'cat > victim <<EOF\nowned\nEOF\n'"
    assert _overwritten(tmp_path, child, ["src/victim", "victim"]) == {"src/victim"}
    assert _verdict(child, {"src/victim"}) == _UNRESOLVED(path="victim")
    outer = "env -C src bash -c 'cat' > victim <<'EOF'\nowned\nEOF\n"
    assert _overwritten(tmp_path, outer, ["src/victim", "victim"]) == {"victim"}
    assert _verdict(outer, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")
    assert simple_commands("env -C src", "/w")[0].argv() == []  # a wrapper with no program
    sed = "sed -i 's/a/b\\nc/' tmp/x.txt"
    assert rewrite_guard(sed, "/w", lambda path: False) is None
    assert rewrite_guard(f"env -C src {sed}", "/w", lambda path: False) is not None


def test_eval_and_source_may_change_the_directory(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_astra-7 / A13_fable-2 / A13_claude-7: after ``eval``, ``source``
    or ``.`` the directory is unknown, as the variables already were (the text they run may
    ``cd``); the scanner kept the directory and checked ``victim`` while bash wrote
    ``src/victim``.  The literal text ``eval`` runs is scanned for its commands: a
    here-document copy in it is refused (a shell's literal ``-c`` text likewise:
    ``test_a_shells_literal_command_string_is_scanned``)."""
    (tmp_path / "setup.sh").write_text("cd src\n")
    for command in (f'eval "cd src"; cat > victim {_OWNED}',
                    f"source setup.sh; cat > victim {_OWNED}",
                    f". ./setup.sh\ncat > victim {_OWNED}"):
        assert _overwritten(tmp_path, command, ["src/victim", "victim"]) == {"src/victim"}
        assert simple_commands(command, "/w")[-1].cwd is None, command
        assert _verdict(command, {"src/victim"}) == _UNRESOLVED(path="victim"), command
    assert simple_commands(f"eval x; cd /v; cat > victim {_OWNED}", "/w")[-1].cwd == "/v"
    assert simple_commands(f"(eval x); cat > victim {_OWNED}", "/w")[-1].cwd == "/w"
    for command in ("eval 'cat <<EOF > existing\nowned\nEOF\n'",
                    'eval "cat > existing" "<<EOF\nowned\nEOF\n"',
                    "T=existing; eval 'cat > $T <<EOF\nowned\nEOF\n'",
                    "eval 'cat' > existing <<'EOF'\nowned\nEOF\n"):
        assert _overwritten(tmp_path, command, ["existing"]) == {"existing"}, command
        assert _verdict(command, {"existing"}) == HEREDOC_VERDICT.format(path="/w/existing")
    # text only the shell knows is not scanned; a sleep in the text is seen
    assert _programs("eval \"$T\" 'date'; eval 'sleep 60'") == ["eval", "eval", "sleep"]
    assert sleep_guard("eval 'sleep 60'") is not None
    # Fix batch W1B02 X5: the text may set ``IFS`` too, which splits a later unquoted value
    split = f'eval "$(echo IFS=%)"; T={tmp_path}/victim%; cat > $T {_OWNED}'
    assert _overwritten(tmp_path, split, ["victim"]) == {"victim"}
    assert heredoc_guard(split, str(tmp_path), os.path.lexists) == _UNRESOLVED(path="$T")


def test_a_shells_literal_command_string_is_scanned(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-7 (the ``bash -c`` half): the literal text a shell's ``-c``
    runs is scanned as ``eval``'s is, so a here-document copy or a sleep in it is seen; the
    child shell runs in the caller's directory and a ``cd`` in its text does not move the
    caller.  Text only the shell knows, and a shell that is an argument, are not scanned."""
    for command in ("bash -c 'cat <<EOF > existing\nowned\nEOF\n'",
                    "sh -ceu 'cat > existing <<EOF\nowned\nEOF\n'",
                    "export T=existing; /bin/bash -ec 'cat > $T <<EOF\nowned\nEOF\n'",
                    "cd sub && bash -c 'cat <<EOF > existing\nowned\nEOF\n'"):
        assert _overwritten(tmp_path, command, ["existing", "sub/existing"]) == {
            "sub/existing" if command.startswith("cd ") else "existing"}, command
        assert _verdict(command, {"existing", "sub/existing"}) == HEREDOC_VERDICT.format(
            path="/w/sub/existing" if command.startswith("cd ") else "/w/existing"), command
    assert sleep_guard("bash -c 'sleep 60'") is not None
    assert simple_commands("bash -c 'cd sub'; cat > victim", "/w")[-1].cwd == "/w"
    assert _programs("bash -c \"$T\"; bash -c 'date'; docker exec c sh -c 'date'") == [
        "bash", "bash", "date", "docker"]


def test_a_startup_file_exported_any_way_runs_before_a_child_shells_text(tmp_path: Path) -> None:
    """Audit 2026-11-21 S13/F3: a child bash reads the file ``BASH_ENV`` names however the
    variable was exported; the scanner knew ``export`` and an assignment before the command
    only, so after ``set -a``, ``declare -x``, an assignment through a name reference or
    behind ``env`` it read the child's text as written: ``victim``, a new file, while the
    startup file's ``cd`` made it the existing ``sub/victim``."""
    (tmp_path / "startup").write_text("cd sub\n")
    for exported, shell in (("set -a; BASH_ENV=$PWD/startup", "bash"),
                            ("declare -x BASH_ENV=$PWD/startup", "bash"),
                            ("typeset -x BASH_ENV=$PWD/startup", "bash"),
                            ("declare -n R=BASH_ENV; R=$PWD/startup; export BASH_ENV", "bash"),
                            ("export BASH_ENV=$PWD/startup", "env bash")):
        command = f"{exported}; {shell} -c 'cat > victim' {_OWNED}"
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["sub/victim"]) == {"sub/victim"}, command
        assert heredoc_guard(command, str(tmp_path), os.path.lexists) == _UNRESOLVED(
            path="victim"), command


def test_an_exec_redirection_holds_for_the_commands_after_it(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-5: ``exec > existing`` redirects the shell's own output, so
    the ``cat`` after it writes ``existing``; the scanner filed the redirection under the
    ``exec`` alone.  A command's own redirection of its output to a file replaces it, a
    subshell's ``exec`` ends with the subshell."""
    command = f"exec > existing\ncat {_OWNED}"
    assert _overwritten(tmp_path, command, ["existing"]) == {"existing"}
    assert _verdict(command, {"existing"}) == HEREDOC_VERDICT.format(path="/w/existing")
    assert _verdict(f"exec >> existing; cat {_OWNED}", {"existing"}) == HEREDOC_VERDICT.format(
        path="/w/existing")
    assert _verdict(f"exec 1> existing 2>&1; {{ cat {_OWNED}}}", {"existing"}) == (
        HEREDOC_VERDICT.format(path="/w/existing"))
    assert _verdict(f'exec > "$LOG"; cat {_OWNED}', set()) == _UNRESOLVED(path="$LOG")
    command = f"exec > existing; cat > other {_OWNED}"
    _overwritten(tmp_path, command, ["existing", "other"])  # exec truncates, cat writes other
    assert [(tmp_path / f).read_text() for f in ("existing", "other")] == ["", "owned\n"]
    assert _verdict(command, {"existing"}) is None
    assert _verdict(command, {"other"}) == HEREDOC_VERDICT.format(path="/w/other")
    # the copy reaches no file (``exec`` itself truncates a file it opens)
    for command in (f"(exec > existing); cat {_OWNED}",
                    f"exec > existing; exec > /dev/null; cat {_OWNED}",
                    f"exec 2> existing; cat {_OWNED}", f"exec 3> existing; cat {_OWNED}",
                    f"exec true; cat {_OWNED}", f"exec > /dev/null; cat {_OWNED}"):
        _overwritten(tmp_path, command, ["existing"])
        assert (tmp_path / "existing").read_text() != "owned\n", command
        assert _verdict(command, {"existing"}) is None, command


def test_a_read_write_redirection_of_standard_output_writes_the_file(tmp_path: Path) -> None:
    """Audit 2026-11-07 A13_claude-6: ``cat 1<> existing`` opens ``existing`` read-write on
    standard output and overwrites its start; the scanner counted ``<>`` as a read only.  On the
    default descriptor (``cat <> existing``) it is a read."""
    command = f"cat 1<> existing {_OWNED}"
    assert _overwritten(tmp_path, command, ["existing"]) == {"existing"}
    assert _verdict(command, {"existing"}) == HEREDOC_VERDICT.format(path="/w/existing")
    assert [w.text for w in simple_commands(command, "/w")[0].outputs] == ["existing"]
    read = f"cat <> existing {_OWNED}"
    assert _overwritten(tmp_path, read, ["existing"]) == set()
    assert _verdict(read, {"existing"}) is None
    assert simple_commands(read, "/w")[0].outputs == []


def test_the_text_a_command_runs_is_piped_where_the_command_is(tmp_path: Path) -> None:
    """Audit 2026-11-10 S10-F1: the commands of the literal text ``bash -c`` or ``eval`` runs
    write to the pipe after the command, so a here-document they copy reaches the ``tee`` (or
    the group) it is piped to; the scanner linked the pipe to the command alone, after it had
    read the text, and the copy ended nowhere."""
    for command in ("bash -c 'cat <<EOF\nowned\nEOF\n' | tee existing",
                    "bash -c 'cat' <<'EOF' | tee existing\nowned\nEOF\n",
                    "eval 'cat' <<'EOF' | tee existing\nowned\nEOF\n",
                    "bash -c 'cat' <<'EOF' | { tee existing; }\nowned\nEOF\n"):
        assert _overwritten(tmp_path, command, ["existing"]) == {"existing"}, command
        assert _verdict(command, {"existing"}) == HEREDOC_VERDICT.format(
            path="/w/existing"), command


def _linked(tmp_path: Path) -> str:
    """Make ``link``, a symbolic link to ``sub/deep``, in *tmp_path*; return that directory."""
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "link").symlink_to("sub/deep")
    return str(tmp_path)


def test_a_parent_directory_after_a_symbolic_link_is_the_filesystems(tmp_path: Path) -> None:
    """Audit 2026-11-10 S10-F6: ``link/../victim`` is a file of the directory the link points
    into, not one next to the link; the scanner collapsed ``link/..`` and the guard checked the
    sibling.  The ``..`` is kept for the guard's filesystem to resolve, after a ``cd`` into the
    link too (the shell is in the directory it names), while ``cd link; cd ..`` is bash's
    logical parent, the start directory.  A device or ``/proc`` path, which the guards tell by
    its name, is unknown once a ``..`` may lead out of it or into it."""
    cwd = _linked(tmp_path)
    for command in (f"cat > link/../victim {_OWNED}", f"cat > {cwd}/link/./../victim {_OWNED}",
                    f"cd link; cat > ../victim {_OWNED}"):
        assert _overwritten(tmp_path, command, ["sub/victim"]) == {"sub/victim"}, command
        assert heredoc_guard(command, cwd, os.path.lexists) == HEREDOC_VERDICT.format(
            path=f"{cwd}/link/../victim"), command
    rooted = f"/..{cwd}/sub/victim"  # the root is its own parent
    assert heredoc_guard(f"cat > {rooted} {_OWNED}", cwd, os.path.lexists) == (
        HEREDOC_VERDICT.format(path=rooted))
    logical = f"cd link; cd ..; cat > victim {_OWNED}"
    assert _overwritten(tmp_path, logical, ["victim"]) == {"victim"}
    assert heredoc_guard(logical, cwd, os.path.lexists) == HEREDOC_VERDICT.format(
        path=f"{cwd}/victim")
    for target in ("/dev/../w/victim", "../dev/stdout", "/proc/self/../self/fd/1"):
        assert _verdict(f"cat > {target} {_OWNED}", {"victim"}) == _UNRESOLVED(path=target)


def test_a_physical_cd_leaves_a_directory_only_the_shell_knows(tmp_path: Path) -> None:
    """Audit 2026-11-10 S10-F7: ``cd -P link`` enters the directory the link points to, so the
    ``cd ..`` after it is that directory's parent; the scanner dropped the option and stayed
    next to the link.  The last of ``-L`` and ``-P`` decides (an option the shell expands may
    be ``-P``), and ``-L`` is the default.

    Review round 1: ``set -P``, ``set -o physical`` and ``shopt -so physical`` make every later
    ``cd`` physical, as a shell's own ``-o physical`` makes those of its text.  Round 2: in
    the shell that ran it (its subshells and ``eval`` included) and in no other; and a
    ``cd ..`` to a sibling directory is followed, as every logical ``cd`` is."""
    cwd = _linked(tmp_path)
    for cd, written in (("cd -P", "sub/victim"), ("cd -L -P", "sub/victim"),
                        ("cd -LP --", "sub/victim"), ("read M <<< P; cd -$M", "sub/victim"),
                        ("cd -`printf '\\120'`", "sub/victim"), ("cd", "victim"),
                        ("cd -P -L", "victim"), ("cd -PL", "victim"),
                        ("set -P; cd", "sub/victim"), ("set -o physical; cd", "sub/victim"),
                        ("shopt -so physical; cd", "sub/victim"), ("set -- -P; cd", "victim"),
                        ("set -P; pushd", "sub/victim"), ("eval 'set -P'; cd", "sub/victim"),
                        ("(set -P); cd", "victim"), ("X=$(set -P); cd", "victim"),
                        ("set -P | cat; cd", "victim"), ("set -P & wait; cd", "victim")):
        command = f"{cd} link; cd ..; cat > victim {_OWNED}"
        assert _overwritten(tmp_path, command, ["sub/victim", "victim"]) == {written}, command
        assert heredoc_guard(command, cwd, os.path.lexists) == (
            HEREDOC_VERDICT.format(path=f"{cwd}/victim") if written == "victim"
            else _UNRESOLVED(path="victim")), command
    for command in ("bash -o physical -c 'cd link; cd ..; cat > victim <<EOF\nowned\nEOF\n'",
                    f"set -P; (cd link; cd ..; cat > victim {_OWNED})"):
        assert _overwritten(tmp_path, command, ["sub/victim", "victim"]) == {"sub/victim"}, command
        assert heredoc_guard(command, cwd, os.path.lexists) == _UNRESOLVED(path="victim"), command
    for cd, written in (("cd sub/deep/..", "sub/victim"),
                        ("cd sub/deep; cd ../sub2", "sub/sub2/victim")):
        passed = f"{cd}; cat > victim {_OWNED}"
        assert _overwritten(tmp_path, passed, [written, "victim"]) == {written}, passed
        assert heredoc_guard(passed, cwd, os.path.lexists) == HEREDOC_VERDICT.format(
            path=f"{cwd}/{written}"), passed


def test_an_attribute_is_its_own_variables(tmp_path: Path) -> None:
    """Review round 2 of S10-F3: the variable an attribute is declared for stays unknown, in
    ``eval``'s text too and handed to a child, while the others stay known; none is once the
    declaration makes a name reference or names a variable only bash can read."""
    for command, written in (
            (f'declare -u T; T=victim; cat > "$T" {_OWNED}', "VICTIM"),
            (f"eval 'declare -u T'; T=victim; cat > \"$T\" {_OWNED}", "VICTIM"),
            (f"export T=victim; readonly T; T=tmp/x bash -c 'cat > \"$T\"' {_OWNED}", "victim")):
        assert _overwritten(tmp_path, command, [written]) == {written}, command
        assert _verdict(command, {written}) == _UNRESOLVED(path="$T"), command
    for declaration, known in (
            ("readonly CONFIG=release", True), ("declare -i count=1", True),
            ("f() { local -r x=1; }; f", True), ("export E=v", True), ("declare -n R=T", False),
            ('readonly "$N"', False), ("declare -u X{1,2}", False)):
        command = f'T=tmp/kept; {declaration}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, ["tmp/kept"]) == {"tmp/kept"}, command
        expected = HEREDOC_VERDICT.format(path="/w/tmp/kept") if known else _UNRESOLVED(path="$T")
        assert _verdict(command, {"tmp/kept"}) == expected, command


def _scan_cpu_seconds(command: str) -> float:
    """The CPU time this thread takes to scan *command*, best of three.

    That is the scan's own work.  Wall time also counts the wait for a core: in a full
    run on a 10-core Mac that took the best of three scans over the bound below.
    """
    scan = functools.partial(simple_commands, command, "/w")
    return min(timeit.repeat(scan, timer=time.thread_time, number=1, repeat=3))


def test_the_assignments_of_a_command_are_read_in_one_pass() -> None:
    """Audit 2026-11-10 S10-F3, review round 1: what the earlier assignments of a command set
    is kept as the scan goes; replayed for every word, 8,000 of them took 18 seconds."""
    known = "; ".join(f"T{i}=x" for i in range(4000))
    for command in (" ".join(["T=x"] * 8000),
                    " ".join(f"T{i}=$T{i - 1}" for i in range(8000)),
                    f"{known}; " + " ".join(f"T{i}=$T{i - 1}" for i in range(1, 4000)),
                    # arithmetic that assigns nothing: one long name is read once
                    "test 'a[" + "a" * 12000 + "]'", "(( " + "a" * 12000 + " ))"):
        assert _scan_cpu_seconds(command) < 0.5, command[:40]
    # An assignment costs the same whatever the shell knows by then: 40,000 of them after
    # 4,000 statements take the time of the two parts.  With a copy of the shell's variables
    # for each assignment, as Bash 3 was scanned, the whole took 14 times that.  The bound
    # above did not show it: an idle Mac scanned its third command in 0.36 s that way.
    assignments = " ".join(f"U{i}=$U{i - 1}" for i in range(1, 40001))
    parts = _scan_cpu_seconds(known) + _scan_cpu_seconds(assignments)
    assert _scan_cpu_seconds(f"{known}; {assignments}") < 4 * parts


def test_the_values_and_the_directory_a_scan_follows_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-22 (left open by FX05e): the scan put a value into every word that
    expands it and resolved every ``cd`` against the directory so far, at the cost of their
    length each time.  Values built from one another doubled line after line (30 lines took
    minutes, 60 all the memory), one long value read by thousands of words, arithmetic
    commands or loops took seconds, and so did thousands of ``cd`` into an ever deeper
    directory.  Past a megabyte of values read the scan ends as one nested too deep does,
    and the Bash tool runs nothing: read on with the values unknown, it let a ``sed`` script
    through that the rewrite guard refuses when it knows it (review of 2026-11-22).  Past a
    directory of ``PATH_MAX`` characters only the shell knows where it is."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    lines = [f'X{i}="$X{i - 1}$X{i - 1}"' for i in range(1, 61)]
    known = simple_commands("; ".join(["X0=aaaaaaaa", *lines[:10], 'echo "$X10"']), "/w")
    assert [(word.text, word.literal) for word in known[-1].words[1:]] == [("a" * 8192, True)]
    doubled = "; ".join(["X0=aaaaaaaa", *lines, 'echo "$X60"'])
    long = "A=" + "a" * 20000
    script = "S='s/a/b\\nc/'; sed -i \"$S\" notes.md"
    rewrite = long + "; :" + ' "$A"' * 53 + "; " + script
    for command in (
        doubled,
        rewrite,
        long + "; " + "Y=$A;" * 4000,
        "X='" + "1+" * 5000 + "1'; " + "((X));" * 1666,
        "X='" + "1" * 80000 + "';" + "(for i in 1; do X=1; done);" * 1000,
    ):
        started = time.thread_time()
        with pytest.raises(TooDeepError) as raised:
            simple_commands(command, "/w")
        assert time.thread_time() - started < 0.5, command[:40]
        assert str(raised.value) == UNREAD_VALUES_VERDICT
    gate = simple_commands(doubled, "/w", optimistic=True)  # its reading: the commands so far
    assert "echo" not in [simple.program() for simple in gate]
    target = tmp_path / "notes.md"
    target.write_text("a\n")
    assert rewrite_guard(script, "/w", lambda path: True) is not None
    with pytest.raises(TooDeepError):
        rewrite_guard(rewrite, "/w", lambda path: True)
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Bash(rewrite, "too long", timeout_seconds=5) == UNREAD_VALUES_VERDICT
    assert target.read_text() == "a\n"
    deep = "cd " + "a/" * 2046 + "b"
    assert simple_commands(f"{deep}; pwd", "/w")[-1].cwd == "/w/" + "a/" * 2046 + "b"
    assert simple_commands(f"{deep}; cd c; pwd", "/w")[-1].cwd is None
    assert _scan_cpu_seconds("; ".join(['cd "$PWD/$PWD"'] * 200)) < 0.5
    # a ``cd`` costs a step, not the length of the directory the steps before it led to
    assert _scan_cpu_seconds("cd a; " * 12000) < 8 * _scan_cpu_seconds("cd a; " * 3000)


def test_the_text_env_splits_is_the_command_line_it_runs(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F6: ``env -S 'text'`` (``-Stext``, ``--split-string=text``) splits
    the text into the command line it runs, with the words after it; the scanner skipped the
    text as an option's value and saw no command.  The text is read by env's rules (``\\_`` is
    a blank, ``\\t`` a tab, a ``>`` is a word, an ``-S`` in it splits again, any word with an
    ``=`` is an assignment and ends the options), the words after it keep what the shell makes
    of them (a redirection before the text is none of them), and the command stays unknown
    where the shell or env expands the text."""
    long_split = subprocess.run(
        ["env", "--split-string=true"], capture_output=True, timeout=10, check=False,
    ).returncode == 0
    for command in ("env -S 'bash -c \"cat > victim\"'", "env '-Sbash -c \"cat > victim\"'",
                    "env -iS 'tee\\_victim'", "env -S tee victim", "env -S 'tee' victim",
                    "env --split-string='tee victim'", "env -S '-u X tee victim'",
                    "env -S '-S \"tee victim\"'", "env -S -S tee victim", "env a-b=1 tee victim",
                    "env -S 'X=1 --split-string=tee tee victim'", "env -- --s=x tee victim",
                    "env -S 'sh -c \"cat\\t>victim\"'", "env -S 11>x 'tee victim'"):
        command = f"{command} {_OWNED}"
        if long_split or "--split-string" not in command:
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    moved = f"env -S '-C src tee victim' {_OWNED}"
    assert _overwritten(tmp_path, moved, ["src/victim", "victim"]) == {"src/victim"}
    assert _verdict(moved, {"src/victim"}) == _UNRESOLVED(path="victim")
    word = f"env -S 'cat > victim' {_OWNED}"  # env's ``>`` is an operand of ``cat``
    assert _overwritten(tmp_path, word, ["victim"]) == set()
    assert _verdict(word, {"victim"}) is None
    globbed = f"env -S tee vic* {_OWNED}"  # a later word only the shell knows
    assert _overwritten(tmp_path, globbed, ["victim"]) == {"victim"}
    assert _verdict(globbed, {"victim"}) == _UNRESOLVED(path="vic*")
    assert _programs("env -S 'tee a' b; env -S \"$X\" c; env -S 'a ${X}' d; env -S") == [
        "tee", "-S", "-S", ""]


def test_a_quoted_or_escaped_assignment_is_a_command_word(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F8: bash assigns only where ``NAME=`` starts the word unquoted;
    ``'T=/dev/null'`` and ``T\\=/dev/null`` are commands that fail, and ``$T`` keeps its value.
    The scanner took the unquoted text for an assignment and checked ``/dev/null``."""
    for word in ("'T=/dev/null'", "T\\=/dev/null", '"T"=/dev/null'):
        command = f'export T=victim; {word}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    assert _programs("'T=x' cmd; A=1 'B=2' cmd") == ["T=x", "B=2"]
    # a wrapper's operand is its assignment however it is quoted; a continued name is the shell's
    for command in (f"env 'T=a b' tee victim {_OWNED}", f"T=x; T\\\n=victim; cat > $T {_OWNED}",
                    f"T=x; T\\\n=victim V=$T; cat > $V {_OWNED}"):
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        verdict = (
            _UNRESOLVED(path="$V")
            if _bash_version() < (4, 0) and "V=$T" in command
            else HEREDOC_VERDICT.format(path="/w/victim")
        )
        assert _verdict(command, {"victim"}) == verdict, command
    # only ``env`` and ``sudo`` take assignments: after another wrapper the word is the program
    script = tmp_path / "x=y"
    script.write_text("#!/bin/sh\necho ran $1\n")
    script.chmod(0o755)
    for wrapper in ("nice", "nohup", "command"):
        assert _bash(f"cd {tmp_path}; {wrapper} ./x=y a") == "ran a\n", wrapper
        assert _programs(f"{wrapper} ./x=y a") == ["x=y"], wrapper
    assert _bash(f"cd {tmp_path}; env ./x=y echo ran") == "ran\n"
    assert _programs("env ./x=y a; sudo ./x=y a; sudo /p/x=y a; T=1 nohup U=2 a") == [
        "a", "a", "x=y", "U=2"]
    # Fix batch W1B02 X4: a name is ASCII, so ``$T\u00e9`` is ``$T`` and a letter
    named = f"T=a; T\u00e9=x; cat > $T\u00e9 {_OWNED}"
    assert _overwritten(tmp_path, named, ["a\u00e9"]) == {"a\u00e9"}
    assert _verdict(named, {"a\u00e9"}) == HEREDOC_VERDICT.format(path="/w/a\u00e9")


def test_an_unquoted_expansion_of_nothing_is_no_word(tmp_path: Path) -> None:
    """Audit 2026-11-21 S13/F1: bash drops an unquoted expansion of nothing, so ``E=; $E tee
    victim`` runs ``tee`` and ``$E read T`` the builtin; the scanner kept an empty command
    word, which is no copier and no builtin.  The word after it is the command, an
    assignment as written included, and a quoted one stays a word: a command bash cannot
    find."""
    rewritten = HEREDOC_VERDICT.format(path="/w/victim")
    for command, verdict in (
            (f"E=; $E tee victim {_OWNED}", rewritten),
            (f"E=; A=1 ${{E}}$E tee victim {_OWNED}", rewritten),
            (f"IFS=x; E=; $E tee victim {_OWNED}", rewritten),
            (f"E=; coproc $E tee victim {_OWNED}", rewritten),
            (f'T=/dev/null; E=; $E read T <<< victim; cat > "$T" {_OWNED}', _UNRESOLVED(path="$T")),
            (f'B=/dev/null; E=; B=victim $E; cat > "$B" {_OWNED}', rewritten),
            (f'B=victim; E=; A=1 $E B=/dev/null; cat > "$B" {_OWNED}', rewritten)):
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == verdict, command
    for word in ('"$E"', '$E""', "$E''"):
        command = f"E=; {word} tee victim {_OWNED}"
        assert _overwritten(tmp_path, command, ["victim"]) == set(), command
        assert _programs(command) == ["", ""] and _verdict(command, {"victim"}) is None, command


def test_the_operands_of_let_are_arithmetic_expressions(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F4: ``let ++T`` and ``let "0,T=2"`` assign ``T``; the scanner read
    each operand as a variable's name, found none and kept the old value."""
    for operand, written in (("++T", "2"), ('"0,T=2"', "2"), ("--T", "0"), ('"$S2_UNSET"', "1")):
        command = f'T=1; let {operand}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, [written]) == {written}, command
        assert _verdict(command, {written}) == _UNRESOLVED(path="$T"), command
    kept = f'T=1; W=0; let V=2 W++; cat > "$T" {_OWNED}'
    assert _overwritten(tmp_path, kept, ["1"]) == {"1"}
    assert _verdict(kept, {"1"}) == HEREDOC_VERDICT.format(path="/w/1")
    # Audit 2026-11-21 S13/F2: bash evaluates the value of a variable an expression reads as
    # an expression too, so one holding ``T=1`` assigns ``T`` wherever its name is an operand;
    # the scan kept ``T=0``, a new file, and let the command rewrite the existing ``1``
    for indirect in ("let X", "Y=X; ((Y))", 'A[0]=0; : "${A[X]}"', ": $(($X))", ": $((X++))",
                     "declare -i N; N=X", "[[ X -eq 1 ]]", "V=abc; : ${V:X}", "Y='A[X]'; ((Y))",
                     'read -r "A[X]" <<< v', "for ((i=X; i<1; i++)); do :; done"):
        command = f'T=0; X="T=1"; {indirect}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, ["1"]) == {"1"}, command
        assert _verdict(command, {"1"}) == _UNRESOLVED(path="$T"), command
    # a number is none, nor is what a length names: no operands
    plain = (f"T=0; X=5; N=X; P='a[$]'; let \"Y = X + N\"; "
             f': $((X + ${{#P}})); cat > "$T" {_OWNED}')
    assert _overwritten(tmp_path, plain, ["0"]) == {"0"}
    assert _verdict(plain, {"0"}) == HEREDOC_VERDICT.format(path="/w/0")


#: FX45 O1: what leaves ``T=1`` in an operand the scan cannot prove a number, then evaluates it.
_UNPROVED_OPERANDS = (
    "X=$(echo T=1); let X", "if true; then X=T=1; fi; ((X))", "for X in T=1; do let X; done",
    "set -- T=1; X=$1; : $((X))", 'set -- T=1; let "$1"', "read X <<< T=1; [[ X -eq 1 ]]",
    "[[ T=1 =~ .* ]]; ((BASH_REMATCH))", ": T=1; let _", "X=$(echo T=1); declare -i N=X",
    "X=$(echo T=1); A[X]=x", "X=$(echo T=1); V=abc; : ${V:X}", ": $((0, $(echo T=1)))",
    "X=$(echo T=1); while ((X)); do break; done", "X=$(echo T=1); Y=X; ((Y))",
    "f() { X=T=1; }; f; let X", "X=0; for i in 1 2; do let X; X=T=1; done",
    # (review 1) what a builtin or a loop gives an integer, and an array value's subscripts
    "declare -i N; read N <<< T=1", "declare -i N; printf -v N %s T=1",
    "declare -i N; for N in T=1; do :; done", "declare -ai A; mapfile A <<< T=1",
    "X=$(echo T=1); A=([X]=v)", "X=$(echo T=1); declare -a A=([X]=v)",
)
#: FX45 O3: what evaluates a subscript that runs ``cat > victim``, under the here-document.
_SUBSCRIPT_COMMANDS = (
    "let X", "((X))", "[[ X -eq 0 ]]", "declare -i N=X", 'test -v "$X"', '[[ -v $X ]]',
    'printf -v "$X" x', "{ : $((X)); }", "{ : ${A[X]}; }", "{ B=([X]=v); }", '{ : "${!X}"; }',
)
#: Evaluations of numbers: a counter, a list of integers, what bash computes at every use.
_NUMBERS = (
    "i=0; while [ $i -lt 3 ]; do i=$((i+1)); done", "for i in 1 2 3; do : $((i * 2)); done",
    'n=0; ((n++)); let n+=1 "m = n * 2"', "i=0; for ((i = 0; i < 3; i++)); do :; done",
    "n=3; m=$((n * 2 + 1)); k=$((m / 2)); echo $((k % 3 == 0)) ${#n} ${n:-0} ${n%3}",
    ": $((RANDOM % 5 + $# + $? + SECONDS))", "n=5; a[n]=x; a[n+1]=y; : ${a[n]}",
    "n=3; if ((n > 2)); then n=$((n - 1)); else ((n += 2)); fi; [[ n -lt 9 ]]",
)


def test_an_operand_the_scan_cannot_prove_a_number_may_assign_and_run(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45 O1/O3: bash evaluates the value of an operand as an expression and
    expands a subscript in it, so an operand of unknown value may assign any variable and run
    a command.  The scan followed the values it knew and trusted the rest: ``X=$(echo T=1);
    let X`` kept ``T=0``, a new file, while bash rewrote ``1``, and a ``cat`` in a subscript
    copied the here-document unseen.  A number stays harmless: counters keep the target."""
    for form in _UNPROVED_OPERANDS:
        command = f'T=0; {form}; cat > "$T" {_OWNED}'
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["1"]) == {"1"}, command
        assert _verdict(command, {"1"}) == _UNRESOLVED(path="$T"), command
    for value in ("'a[$(cat > victim)]'", "$(echo 'a[$(cat > victim)]')"):
        # (a word is expanded before its command's redirections: there it reads the pipe)
        for command in (*(f"X={value}; {form} {_OWNED}" for form in _SUBSCRIPT_COMMANDS),
                        f"X={value}; cat <<'EOF' | : $((X))\nowned\nEOF\n"):
            if _oracle_runs(command):
                assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
                assert (tmp_path / "victim").read_text() == "owned\n", command
            assert _verdict(command, {"victim"}) is not None, command
    for form in _NUMBERS:
        command = f'T=new; {form}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, ["victim"]) == set(), command
        assert _verdict(command, {"victim"}) is None, command
        assert (tmp_path / "new").read_text() == "owned\n", command
        (tmp_path / "new").unlink()


#: Functions with a here-document, called where the scan does not see it (by a word only the
#: shell knows, by a trap): one that may not have been defined, ones whose definition is given
#: the here-document, one a trap set before the definition names.
_UNSEEN_CALLS = (
    "true && f() { cat > victim <<'EOF'\nowned\nEOF\n}\ncd sub; $(echo f)",
    f"f() {{ cat > victim; }} {_OWNED}cd sub; $(echo f)",
    f"f() {{ tee victim; }}; g() {{ f; }} {_OWNED}cd sub; $(echo g)",
    f"trap f EXIT; f() {{ cat > victim; }} {_OWNED}cd sub",
)
#: FX45 O2: commands only the shell knows, each with the file bash rewrites: a word that is
#: nothing, a copier, ``eval`` or a builtin that moves the shell; text bash computes or runs
#: later; a script on the standard input; a pattern only ``extglob`` makes one.
_ONLY_THE_SHELL_KNOWS = (
    (f"$SUDO tee victim {_OWNED}", "victim"),
    *((command, "sub/victim") for command in _UNSEEN_CALLS),
    (f'X=$(echo tee); "$X" victim {_OWNED}', "victim"),
    (f"$(echo tee) victim {_OWNED}", "victim"),
    (f'set -- tee; "$@" victim {_OWNED}', "victim"),
    (f'X=$(echo "tee victim x"); $X/bin/python {_OWNED}', "victim"),
    (f"shopt -s nullglob; nosuch* tee victim {_OWNED}", "victim"),
    (f"$SUDO cd sub; cat > victim {_OWNED}", "sub/victim"),
    (f'T=new; $SUDO export T=victim; cat > "$T" {_OWNED}', "victim"),
    (f'T=new; $SUDO read T <<< victim; cat > "$T" {_OWNED}', "victim"),
    (f"$SUDO exec > victim; cat {_OWNED}", "victim"),
    (f'X=$(echo "eval python3(){{ tee victim;}}"); $X; python3 {_OWNED}', "victim"),
    (f'run() {{ "$@"; }}; run tee victim {_OWNED}', "victim"),
    (f"f() {{ tee victim; }}; f {_OWNED}", "victim"),
    (f'eval "$(echo cat) > victim" {_OWNED}', "victim"),
    (f'bash -c "$(echo cat) > victim" {_OWNED}', "victim"),
    ("X=$(echo eval); $X 'cat > victim <<E\nowned\nE'", "victim"),
    ("T=new; trap 'cat > \"$T\" <<E\nowned\nE' EXIT; T=victim", "victim"),
    ("bash <<'EOF'\ncat > victim <<'X'\nowned\nX\nEOF\n", "victim"),
    ("bash -s <<'EOF'\ncat > victim\nowned\nEOF\n", "victim"),
    ("bash <<EOF\n$(echo cat) > victim <<X\nowned\nX\nEOF\n", "victim"),
    ("cat <<'EOF' | sh\ncat > victim <<'X'\nowned\nX\nEOF\n", "victim"),
    (". /dev/stdin <<'EOF'\ncat > victim <<'X'\nowned\nX\nEOF\n", "victim"),
    # review 1: a function named as a wrapper or called with an assignment, the handler of a
    # missing program, a trap on a signal the text sends, a prompt bash expands, a name
    # reference to an element, a function or a profile handed to a child shell
    (f"timeout() {{ cat > victim; }}; timeout 5 python3 - {_OWNED}", "victim"),
    (f'T=new; f() {{ cat > "$T"; }}; T=victim f {_OWNED}', "victim"),
    (f"command_not_found_handle() {{ cat > victim; }}; nosuchprog9 - {_OWNED}", "victim"),
    (f"trap 'cd sub' INT; kill -INT $$; cat > victim {_OWNED}", "sub/victim"),
    (f"X='$(cat > victim)'; {{ : \"${{X@P}}\"; }} {_OWNED}", "victim"),
    (f"PS4='$(cat > victim)'; set -x; {{ :; }} {_OWNED}", "victim"),
    (f"PS4='$(cat > victim)' bash -xc : {_OWNED}", "victim"),
    (f"declare -n R='A[$(cat > victim)]'; {{ : \"$R\"; }} {_OWNED}", "victim"),
    (f"env 'BASH_FUNC_cat%%=() {{ tee victim; }}' bash -c 'cat > new' {_OWNED}", "victim"),
    ("mkdir -p h; echo 'cd sub' > h/.bash_profile; HOME=$PWD/h bash -lc 'cat > victim' "
     + _OWNED, "sub/victim"),
    # a descriptor the text opened, under its number or a name; a copier another program runs
    (f"exec 3>victim; cat >&3 {_OWNED}", "victim"),
    (f"{{ cat >&3; }} 3>victim {_OWNED}", "victim"),
    (f"exec 4>victim; exec 1>&4; cat {_OWNED}", "victim"),
    (f"tee /dev/fd/3 3>victim >/dev/null {_OWNED}", "victim"),
    (f"{{ cat > /dev/stdout; }} > victim {_OWNED}", "victim"),
    (f"setsid -w tee victim >/dev/null {_OWNED}", "victim"),
    (f"find . -maxdepth 0 -exec tee victim \\; >/dev/null {_OWNED}", "victim"),
    # a path's name is a function's; a value that ends in a literal name: a copier's, one
    # that may not have been assigned, one bash splits
    (f'/usr/bin/python3() {{ tee victim; }}; D=$(echo /usr); "$D/bin/python3" {_OWNED}', "victim"),
    (f'P="$(dirname "$(command -v tee)")/tee"; "$P" victim {_OWNED}', "victim"),
    (f'P=tee; false && P="$(pwd)/nope/python3"; "$P" victim {_OWNED}', "victim"),
    (f'P="$(echo tee victim) ./nope/python3"; $P {_OWNED}', "victim"),
    # a name the command itself gave an existing file, or a directory, before it writes
    (f"ln -s victim alias; cat > alias {_OWNED}", "victim"),
    (f"ln -s sub link; cd link; cat > victim {_OWNED}", "sub/victim"),
    (f'A=$(echo alias); cp -l victim "$A"; cat > alias {_OWNED}', "victim"),
    (f"for i in 1 2; do cat > alias {_OWNED}ln -sf victim alias; done", "victim"),
    (f"shopt -s extglob\ntee @(victim) {_OWNED}", "victim"),
    (f'G="tee victim"; $G {_OWNED}', "victim"),
)
#: Commands whose here-document rewrites nothing, by a proof the scan has: a program named
#: by a path's literal end or by a known value bash splits, a header that runs nothing, a
#: sourced file (its commands are not the text's), a script or a function that only creates.
_PROVED = (
    f'"$HOME/bin/python" - {_OWNED}', f'PY="python3 -c pass"; $PY {_OWNED}',
    f". ./activate; python3 -c pass {_OWNED}",
    f'case "$1" in a) echo a;; *) echo b;; esac; cat > new {_OWNED}',
    f'for f in "$HOME"/*.txt; do echo "$f"; done; cat > new {_OWNED}',
    f'[ -n "$S2_UNSET" ] && echo set; cat > new {_OWNED}',
    "bash <<'EOF'\ncat > new <<'X'\nowned\nX\nEOF\n", f"f() {{ cat > new; }}; f {_OWNED}",
    f"trap 'rm -f scratch.tmp' EXIT; cat > new {_OWNED}",
    f"(X=$(echo cd); $X sub); cat > new {_OWNED}",
    f"set -euo pipefail; shopt -s nullglob; cd sub 2>/dev/null; cat > ../new {_OWNED}",
    f"{{ echo a; echo b; }} > out.txt; for f in a b; do [ -f $f ] || continue; done\n"
    f"g() {{ [ -n \"$1\" ] || return 1; }}; g x\nT=new; cat > \"$T\" {_OWNED}",
    "f() { cat /dev/null | { read a; }; }; n=$(( $(date +%s) + 1 )); "
    "f() { cat /dev/null | { read a; }; }; while f; do sleep 1; done; echo $n",
    '[ -n "$S2_UNSET" ] && g() { echo a; }; $S2_UNSET true; echo done',
    # review 1: the tool's own descriptors, a remote copier, a plain name reference
    f"echo x > out.log 2>&1; cat >&2 {_OWNED}", f"ssh -G localhost tee victim {_OWNED}",
    f"declare -A M=([a]=1); declare -n R=M; R[b]=2; cat > new {_OWNED}",
    # a variable assigned one field that ends in a literal name, quoted: that program
    f'P="$(pwd)/nope/python3"; "$P" - {_OWNED}',
    f'ROOT=$(pwd); export P=$ROOT/nope/python3; Q="$P"; "${{Q}}" - {_OWNED}',
    # a file moved into place after it is written, a link beside the new file
    f"cat > x.tmp {_OWNED}mv x.tmp x.txt",
    f'mkdir -p d; ln -sfn "$(pwd)/sub" d/link; cat > d/n.md {_OWNED}',
)
#: FX45: what bash skips or undoes while the text reads as run, each with the file bash
#: rewrites.  A failed evaluation assigns nothing; a failed expansion or assignment skips the
#: rest of its line; a builtin whose redirection fails does not run; ``return``, ``break``
#: and ``continue`` skip the rest of a body; a definition may not run, or end with its
#: subshell; an option changes what an assignment outlives and what a failure skips.
_X = "X='a[$(cat > victim)]'"
_SKIPPED = (
    (f"{_X}; ((X = 1/0)); let X {_OWNED}", "victim"),
    (f"{_X}; readonly A; ((A = 0, X = 1)); let X {_OWNED}", "victim"),
    (f"{_X}; Y=08; ((X = Y)); let X {_OWNED}", "victim"),
    (f"{_X}\nX=$((1/0))\nlet X {_OWNED}", "victim"),
    (f"{_X}; for ((X = 1/0; X < 1; X++)); do :; done; let X {_OWNED}", "victim"),
    (f'T=victim\n: $((1/0)); T=new\ncat > "$T" {_OWNED}', "victim"),
    (f'T=victim\nU=$((1/0)) T=new\ncat > "$T" {_OWNED}', "victim"),
    (f"cd sub\ncd .. $((1/0))\ncat > victim {_OWNED}", "sub/victim"),
    (f"x=abc\n: ${{x:1/0}}; cd sub\ncat > victim {_OWNED}", "victim"),
    (f"readonly R=1\nR=2; cd sub\ncat > victim {_OWNED}", "victim"),
    (f"f() {{ : $((1/0)); }}\nf; cd sub\ncat > victim {_OWNED}", "victim"),
    (f"if true; then\n: $((1/0))\nfi; cd sub\ncat > victim {_OWNED}", "victim"),
    (f": $((1/0)); cat() {{ :; }}\ncat > victim {_OWNED}", "victim"),
    (f"cd sub > /nonexistent/x; cat > victim {_OWNED}", "victim"),
    (f'T=victim; export T=new > /nonexistent/x; cat > "$T" {_OWNED}', "victim"),
    (f"{{ cd sub; }} > /nonexistent/x; cat > victim {_OWNED}", "victim"),
    (f"eval 'cat() {{ :; }}' > /nonexistent/x; cat > victim {_OWNED}", "victim"),
    (f'T=new; for i in 1; do T=victim; break; T=new; done; cat > "$T" {_OWNED}', "victim"),
    (f'T=new; for a in 1; do T=victim; for b in 1; do continue 2; done; T=new; done; '
     f'cat > "$T" {_OWNED}', "victim"),
    (f"f() {{ return; cat() {{ :; }}; }}; f; cat > victim {_OWNED}", "victim"),
    (f"false && cat() {{ :; }}; cat > victim {_OWNED}", "victim"),
    (f"(cat() {{ :; }}); cat > victim {_OWNED}", "victim"),
    (f"cat() {{ :; }} | true; cat > victim {_OWNED}", "victim"),
    (f"if false; then cat() {{ :; }}; fi; cat > victim {_OWNED}", "victim"),
    (f'set -o posix; T=new; T=victim :; cat > "$T" {_OWNED}', "victim"),
    (f"bash --posix -c 'T=new; T=victim :; cat > \"$T\"' {_OWNED}", "victim"),
    (f'shopt -s failglob\nT=victim\n: nosuch*; T=new\ncat > "$T" {_OWNED}', "victim"),
)


def test_what_bash_skips_is_not_taken_for_run(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45: the scan read every command of a line as run and every
    arithmetic target as assigned.  Bash assigns nothing when an evaluation fails, skips the
    rest of a line when an expansion or an assignment fails, does not run a builtin whose
    redirection fails, leaves a body at ``return``, ``break`` and ``continue``, and runs
    another text under ``posix`` or ``failglob``: the state the scan kept (a directory, a
    variable, a function named ``cat``) was not the shell's, and the existing file was
    rewritten.  What may not have run is in doubt from there to the end of the line."""
    for command, written in _SKIPPED:
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == {written}, command
            assert (tmp_path / written).read_text() == "owned\n", command
        assert _verdict(command, {"victim", "sub/victim"}) is not None, command


#: FX45 (review of FX12, S13/F3): no startup variable reaches these child shells with a value.
_NO_STARTUP_FILE = (
    "unset BASH_ENV; bash -c 'cat > new'", "BASH_ENV=; bash -c 'cat > new'",
    "export BASH_ENV=; bash -c 'cat > new'", "BASH_ENV=sub/env; bash -c 'cat > new'",
    "export BASH_ENV=sub/env; env -i bash -c 'cat > new'", "BASH_ENV=sub/env sh -c 'cat > new'",
    "declare -n R=X; R=1; bash -c 'cat > new'", "export BASH_ENV=sub/env; dash -c 'cat > new'",
)


def test_what_only_the_shell_knows_is_no_proof(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45 O2: a command word of unknown value may be nothing, ``tee``,
    ``eval`` or ``cd``, and what it runs may redefine every later command; the scan took such
    a command for a harmless program, and a script on a shell's standard input for no
    commands at all.  A here-document that reaches what only the shell knows is refused, and
    nothing is known after it; what the scan can prove still runs."""
    (tmp_path / "activate").write_text("export VIRTUAL_ENV=venv\n")
    for command, written in _ONLY_THE_SHELL_KNOWS:
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == {written}, command
            assert (tmp_path / written).read_text() == "owned\n", command
        assert _verdict(command, {"victim", "sub/victim"}) is not None, command
    assert _verdict(f"$SUDO tee new {_OWNED}", set()) == HEREDOC_UNKNOWN_VERDICT.format(
        command="$SUDO")
    # the here-document of a body may be run where only the shell knows; no body holds one
    # in a command without ``<<`` (the two last of _PROVED)
    for command in _UNSEEN_CALLS:
        assert "only the shell knows what that runs" in (
            _verdict(command, {"sub/victim"}) or ""), command
    for command in _PROVED:
        assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == set(), command
        assert _verdict(command, {"victim", "sub/victim"}) is None, command
    # a child shell is read as written unless a startup variable may reach it with a value
    (tmp_path / "sub" / "env").write_text("cd sub\n")
    for form in _NO_STARTUP_FILE:
        command = f"{form} {_OWNED}"
        assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == set(), command
        assert _verdict(command, {"victim", "sub/victim"}) is None, command
        if "dash" not in form or shutil.which("dash"):
            assert (tmp_path / "new").read_text() == "owned\n", command
            (tmp_path / "new").unlink()
        assert not (tmp_path / "sub" / "new").exists(), command


def _scanned_from(frames: int, command: str) -> int:
    """The number of simple commands of *command*, scanned from *frames* more frames down."""
    if frames:
        return _scanned_from(frames - 1, command)
    return len(simple_commands(command, "/w"))


def test_a_scan_at_the_nesting_limit_stays_within_the_interpreters_stack() -> None:
    """Audit 2026-11-10 A/S10-F14: a level costs up to nine Python frames, so 100 legal levels
    raised ``RecursionError`` from a caller 100 frames deep, instead of being scanned (or
    refused as too deep one level further)."""
    for opener, closer in (('a > "$(', ')"'), ('x=("$(', ')")')):
        within = opener * MAX_NESTING + "b" + closer * MAX_NESTING
        assert _scanned_from(200, within) == MAX_NESTING + 1, opener
        with pytest.raises(TooDeepError):
            _scanned_from(200, opener + within + closer)


def test_a_program_named_with_a_digit_is_no_wrappers_number(tmp_path: Path) -> None:
    """Audit 2026-11-10 A/S10-F9: ``7z a out.7z`` ran the program ``a`` to the scanner, which
    skipped every word starting with a digit as a wrapper's number.  Only ``timeout`` takes
    one, its first operand, however it is spelt."""
    assert _programs("7z a out.7z dir; 2to3 -w f.py; timeout 5 7z x a.7z; timeout -k 3 5s tee") == [
        "7z", "2to3", "7z", "tee"]
    command = f"timeout ${{S2_UNSET:-5}} tee victim {_OWNED}"
    if shutil.which("timeout"):  # GNU coreutils
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}
    assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")


def test_a_wrappers_option_takes_its_value_as_getopt_reads_it(tmp_path: Path) -> None:
    """Fix batch W1B02 X1, X2: in a cluster the first letter that takes a value takes the rest
    of the word, or the next word when it ends the cluster (``env -iu T``, ``timeout -vk 1``);
    the scanner knew the option alone (``-u``) and named the value the program.  GNU ``env
    -a``/``--argv0`` (coreutils 9.2) takes one too."""
    gnu_argv0 = subprocess.run(["env", "-a", "x", "true"], capture_output=True).returncode == 0
    for wrapper in ("env -iu T", "env -ui", "timeout -vk 1 5", "stdbuf -oL", "env -a zero",
                    "env --argv0 zero", "env -ia zero"):
        command = f"{wrapper} tee victim {_OWNED}"
        if shutil.which(wrapper.split()[0]) and (gnu_argv0 or "zero" not in wrapper):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command


def test_a_program_wrapper_runs_no_builtin(tmp_path: Path) -> None:
    """Fix batch W1B02 X3: ``env cd sub`` (``sudo``, ``nohup``, ``nice``...) runs a program
    named ``cd``, if there is one, and the shell stays where it is; the scanner followed the
    builtin and checked ``sub/victim``.  ``command`` and ``builtin`` run the builtin."""
    for command, written in (
            (f"env cd sub; cat > victim {_OWNED}", "victim"),
            (f"nohup cd sub; cat > victim {_OWNED}", "victim"),
            (f"nice pushd sub; cat > victim {_OWNED}", "victim"),
            (f'T=victim; env export T=sub/victim; cat > "$T" {_OWNED}', "victim"),
            (f"exec > victim; env exec >> sub/victim; cat {_OWNED}", "victim"),
            (f"command cd sub; cat > victim {_OWNED}", "sub/victim"),
            (f"X=/usr/bin/env builtin cd sub; cat > victim {_OWNED}", "sub/victim")):
        assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == {written}, command
        assert _verdict(command, {written}) == HEREDOC_VERDICT.format(
            path=f"/w/{written}"), command


def test_a_cd_looks_in_cdpath_first(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F3: with ``CDPATH=src``, ``cd sub`` enters ``src/sub`` when it
    exists; the scanner resolved the operand in the directory alone.  Once the command may have
    set ``CDPATH``, only the shell knows where such a ``cd`` leads; ``./sub`` is not searched."""
    for cd in ("CDPATH=src; cd sub", "CDPATH=src cd sub", "export CDPATH=src; pushd sub",
               "T=CDPATH; : ${!T=src}; cd sub"):
        command = f"{cd}; cat > victim {_OWNED}"
        written = (
            "sub/victim"
            if _bash_version() < (4, 0) and cd.startswith("T=CDPATH")
            else "src/sub/victim"
        )
        assert _overwritten(
            tmp_path, command, ["src/sub/victim", "sub/victim"],
        ) == {written}, command
        assert _verdict(command, {written}) == _UNRESOLVED(path="victim"), command
    for cd in ("CDPATH=src; cd ./sub", "CDPATH=; cd sub", "CDPATH=src; CDPATH= cd sub"):
        command = f"{cd}; cat > victim {_OWNED}"
        assert _overwritten(tmp_path, command, ["src/sub/victim", "sub/victim"]) == {
            "sub/victim"}, command
        assert _verdict(command, {"sub/victim"}) == HEREDOC_VERDICT.format(
            path="/w/sub/victim"), command
    evaluated = f'eval "$(echo CDPATH=src)"; cd {tmp_path}; cd sub; cat > victim {_OWNED}'
    assert _overwritten(tmp_path, evaluated, ["src/sub/victim", "sub/victim"]) == {"src/sub/victim"}
    assert heredoc_guard(evaluated, str(tmp_path), f"{tmp_path}/src/sub/victim".__eq__) == (
        _UNRESOLVED(path="victim"))


def test_a_logical_cd_that_names_no_directory_follows_the_links(tmp_path: Path) -> None:
    """Audit 2026-11-10 L/L11-F4: ``cd link/../deep`` enters ``deep`` next to the link when it
    is a directory, else the ``deep`` of the directory the link points into; the scanner
    claimed the first unseen.  A caller that can tell directories (*isdir*) gets a directory
    only the shell knows there; one the command itself creates is followed as before."""
    cwd = str(tmp_path)
    for name in ("other/sub", "other/deep", "other/gone", "both", "plain/a", "plain/b"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "link").symlink_to("other/sub")
    command = f"cd link/../deep; cat > victim {_OWNED}"
    assert _overwritten(tmp_path, command, ["other/deep/victim"]) == {"other/deep/victim"}
    walked = f"cd link/../gone/../both; cat > victim {_OWNED}"  # no ``gone`` next to the link
    assert _overwritten(tmp_path, walked, ["other/both/victim", "both/victim"]) == {
        "other/both/victim"}
    for text, directory in ((command, None), ("cd link; cd ../deep; cat", None), (walked, None),
                            ("cd link/..; cat", cwd), ("cd link; cd ../plain; cat", f"{cwd}/plain"),
                            ("cd plain/a/../b; cat", f"{cwd}/plain/b"),
                            ("mkdir -p new/a new/b && cd new/a/../b && cat", f"{cwd}/new/b")):
        assert simple_commands(text, cwd, isdir=os.path.isdir)[-1].cwd == directory, text
    assert simple_commands(command, cwd)[-1].cwd == f"{cwd}/deep"  # no oracle: as written
    # Audit 2026-11-12 B5 (X4): no guard passed the oracle, so the Bash tool judged ``deep/victim``
    # (no such file: a new one) while bash rewrote the unread ``other/deep/victim``.
    victim = tmp_path / "other" / "deep" / "victim"
    victim.write_text("ORIGINAL\n")
    tools = UsefulTools(work_dir=cwd)
    for text in (command, f"cd link; cd ../deep; cat > victim {_OWNED}",
                 f"cd link/../deep; tee victim {_OWNED}"):
        assert tools.Bash(text, "through a link", timeout_seconds=5) == _UNRESOLVED(path="victim")
    out = tools.Bash("cd link/../deep; sed -i 's/ORIGINAL/a\\nb/' victim", "sed", timeout_seconds=5)
    assert out.startswith("REFUSED: the `sed -i` call did NOT run"), out
    report = tools.run_commands_parallel(json.dumps([command]))
    assert report.startswith("1 commands: 0 succeeded, 1 failed"), report
    assert _UNRESOLVED(path="victim") in report
    assert victim.read_text() == "ORIGINAL\n"
    # A directory the command walks to is judged as before: a new file there is created.
    assert tools.Bash(f"cd plain/a/../b; cat > fresh {_OWNED}", "walked", timeout_seconds=5) == ""
    assert (tmp_path / "plain" / "b" / "fresh").read_text() == "owned\n"


def test_a_conditional_command_holds_no_redirection(tmp_path: Path) -> None:
    """Audit 2026-11-10 A/S10-F12: in ``[[ b > $V ]]`` the ``>`` compares; the scanner read a
    redirection to the file ``$V`` names, and split the command at ``&&``, ``||`` and
    parentheses.  The expansions in it run, and its comments are skipped."""
    command = f"V=a; [[ b > $V && ( -n x || -n y ) ]] && cat > victim {_OWNED}"
    assert _overwritten(tmp_path, command, ["victim", "a"]) == {"victim"}
    simples = simple_commands(command, "/w")
    assert [simple.program() for simple in simples] == ["", "[[", "cat"]
    assert simples[1].outputs == []
    assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")
    several = "[[ $(date) < <(id) ]]; [[ x =~ ^(a|x)$ # it's\n && -n y ]] | tee f; [[x ]] > f"
    assert _programs(several) == ["[[", "date", "id", "[[", "tee", "[[x"]
    # the parentheses of a pattern hold text (a ``#``, a blank, a ``]]``); a group's a comment
    for test in ("[[ x =~ ^(#|x)$ ]]", "[[ x == @(a|#b|x) ]]", '[[ "a ]] b" =~ (a ]] b) ]]',
                 "[[ (# it's\n -n x ) ]]", "case '[[' in\n [[ ) true;; esac"):
        command = f"{test} && cat > victim {_OWNED}"
        if _bash_version() >= (4, 0) or "@(a|#b|x)" not in test:
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    assigned = f'T=1; [[ T=2 -eq 2 ]]; cat > "$T" {_OWNED}'  # an operand of ``-eq`` is arithmetic
    assert _overwritten(tmp_path, assigned, ["2"]) == {"2"}
    assert _verdict(assigned, {"2"}) == _UNRESOLVED(path="$T")


def test_the_alternatives_of_a_case_pattern_are_no_pipeline(tmp_path: Path) -> None:
    """Audit 2026-11-10 A/S10-F4: the ``|`` of ``y|x)`` was read as a pipe, so the branch's
    first ``;;`` restored the variables and the directory from before the pattern, as after a
    pipeline's stage; what the branch set was claimed as it was before the ``case``."""
    for command, written, target in (
            (f"DIR=a; case x in y|x) DIR=b ;; esac; cat > $DIR/victim {_OWNED}", "b/victim",
             "$DIR/victim"),
            (f"case x in y|x) cd sub ;; esac; cat > victim {_OWNED}", "sub/victim", "victim")):
        assert _overwritten(tmp_path, command, [written, "a/victim", "victim"]) == {written}
        assert _verdict(command, {written}) == _UNRESOLVED(path=target), command
    piped = f"T=p; case x in y|x) T=q | cat ;; esac; cat > $T {_OWNED}"  # a stage's own
    assert _overwritten(tmp_path, piped, ["p", "q"]) == {"p"}
    assert _verdict(piped, {"p"}) == HEREDOC_VERDICT.format(path="/w/p")


def test_a_comment_in_an_array_value_is_skipped(tmp_path: Path) -> None:
    """Audit 2026-11-10 A/S10-F3: bash skips a comment between the words of ``NAME=( ... )``;
    the scanner read its quote or parenthesis as the value's, and hid the commands after the
    array or took the rest of the array for commands."""
    for comment in ("# don't use bar", "# see (note", "# closed) early", "# `tick $(sub"):
        command = f"ARGS=(\n  --foo   {comment}\n  --baz\n)\ncat > victim {_OWNED}"
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _programs(command) == ["", "cat"], command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    # a ``#`` inside a word is no comment; one after a continued line is
    for word in (f"A=(a\\ #b 'c'#d e#f); cat > victim {_OWNED}",
                 f"A=(a \\\n# it's)\n b); cat > victim {_OWNED}"):
        assert _overwritten(tmp_path, word, ["victim"]) == {"victim"}, word
        assert _verdict(word, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), word
    # an operator in the value is a syntax error: bash drops the line, with the ``cd`` and the
    # here-document it read of it, and goes on with the next one
    for broken in ("cd sub; A=(a ; b)", "A=(a ( # x))", "cd sub; A=(a > b) \\", "cat <<EOF; A=(()"):
        command = f"{broken}\ncat > victim {_OWNED}"
        assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == _UNRESOLVED(path="victim"), command
    substituted = f"A=(<(cat > victim {_OWNED}))"  # a process substitution is a word of the value
    assert _overwritten(tmp_path, substituted, ["victim"]) == {"victim"}
    assert _verdict(substituted, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")


def test_a_function_call_sets_what_its_body_sets(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F9: calling a function runs its body, so the variables it may set
    are unknown after the call, and the directory when it may change it (those of every
    function defined by then: the called one may call them); the scanner kept both as they
    were before the call.  A body that runs a word only the shell knows (``"$@"``) may run any
    builtin, and one that redirects the shell's output does so at the call.  What no body sets
    stays known."""
    for command, written, target in (
            (f'f() {{ T=victim; }}; T=/dev/null; f; cat > "$T" {_OWNED}', "victim", "$T"),
            (f"f() {{ cd src; }}; cd {tmp_path}; f; cat > victim {_OWNED}", "src/victim", "victim"),
            (f"main() {{ g; }}; function g {{ cd src; }}; cd {tmp_path}; main; cat > victim "
             f"{_OWNED}", "src/victim", "victim"),
            (f'run() {{ "$@"; }}; cd {tmp_path}; run cd src; cat > victim {_OWNED}', "src/victim",
             "victim"),
            (f'run() {{ $1 T=victim; }}; T=/dev/null; run export; cat > "$T" {_OWNED}', "victim",
             "$T")):
        assert _overwritten(tmp_path, command, [written, "victim"]) == {written}, command
        assert heredoc_guard(command, str(tmp_path), os.path.lexists) == _UNRESOLVED(
            path=target), command
    for body in ('echo "$@"', "V=1 | cat", "( cd src; T=x )"):
        kept = f'T=victim; log() {{ {body}; }}; log x; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, kept, ["victim"]) == {"victim"}, kept
        assert _verdict(kept, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), kept
    redirected = f"f() {{ exec > victim; }}; exec > /dev/null; f; cat {_OWNED}"
    assert _overwritten(tmp_path, redirected, ["victim"]) == {"victim"}
    assert _verdict(redirected, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim")
    for body in ('eval "$S2_UNSET"', 'read "$S2_UNSET" <<< x', "g() { :; }"):  # may set any
        command = f'f() {{ {body}; }}; T=victim; f; cat > "$T" {_OWNED}'
        assert _verdict(command, {"victim"}) == _UNRESOLVED(path="$T"), command


#: The variables bash 5.2 does not keep as assigned (asked of bash itself, name by name).
_SHELL_SET = (
    "BASHOPTS BASHPID BASH_ALIASES BASH_ARGC BASH_ARGV BASH_CMDS BASH_COMMAND BASH_LINENO "
    "BASH_REMATCH BASH_SOURCE BASH_SUBSHELL BASH_VERSINFO DIRSTACK EPOCHREALTIME EPOCHSECONDS "
    "EUID FUNCNAME GROUPS HISTCMD LINENO OLDPWD OPTARG OPTIND PIPESTATUS PPID PWD RANDOM SECONDS "
    "SHELLOPTS SRANDOM UID _"
).split()
#: The command after which one of them holds another value (any command, for the others).
_SETS = {
    "PWD": "cd sub", "OLDPWD": "cd sub; cd ..", "DIRSTACK": "cd sub",
    "BASH_REMATCH": "[[ other =~ .* ]]", "OPTARG": "getopts a: o -a other",
    "OPTIND": "getopts a: o -a other",
    # the tables of ``hash`` and ``alias``: element 0 is what the bare name expands to
    "BASH_CMDS": "hash -p other 0", "BASH_ALIASES": "alias 0=other",
}


def test_a_variable_bash_sets_itself_is_not_followed_by_its_assignment(tmp_path: Path) -> None:
    """Audit 2026-11-12 B5/A9: bash sets some variables itself (``PWD`` at a ``cd``, ``_`` after
    every command, ``RANDOM`` at every use) and refuses to assign others (``UID``), so what a
    command assigns one is not what it expands to later.  The scanner followed the assignment:
    after ``PWD=tmp; cd src`` it took ``"$PWD/f"`` for ``tmp/f`` while bash wrote ``src/f``.  Only
    the shell knows such a variable, however it is assigned, and a child shell sets its own."""
    (tmp_path / "sub").mkdir()
    modern = _bash_version() >= (5, 1)  # an older bash keeps the ones it does not have yet
    for name in _SHELL_SET:
        sets = _SETS.get(name, ": other")
        command = f': first; {name}=victim; {sets}; cat > "${name}" {_OWNED}'
        if modern or name in ("PWD", "_"):
            assert _overwritten(tmp_path, command, ["victim"]) == set(), command
        unknown = _UNRESOLVED(path=f"${name}")
        assert _verdict(command, {"victim"}) == unknown, command
        exported = f'export {name}=victim; {sets}; cat > "${name}" {_OWNED}'
        assert _verdict(exported, {"victim"}) == unknown, exported
        # Not run here, where ``$_`` names the shell's own program; a starting shell's
        # ``PWD`` is its directory, which is no file.
        child = f"{name}=victim bash -c 'cat > \"${name}\"' {_OWNED}"
        assert _verdict(child, {"victim"}) == (None if name == "PWD" else unknown), child
    counted = f"SHLVL=victim bash -c 'cat > \"$SHLVL\"' {_OWNED}"  # the child counts itself
    assert _overwritten(tmp_path, counted, ["victim"]) == set()
    assert _verdict(counted, {"victim"}) == _UNRESOLVED(path="$SHLVL")
    # A variable bash leaves alone is still followed, one of the user's named ``BASH_...`` too.
    for name in ("HOSTNAME", "BASH_CUSTOM"):
        kept = f': first; {name}=victim; : other; cat > "${name}" {_OWNED}'
        assert _overwritten(tmp_path, kept, ["victim"]) == {"victim"}
        assert _verdict(kept, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), name


def test_the_command_of_a_coproc_is_scanned(tmp_path: Path) -> None:
    """Audit 2026-11-12 B5 (owner): ``coproc`` is a reserved word of bash that the scanner took
    for a program, so the command after it was that program's arguments and its rewrite of an
    existing file passed the guards.  ``coproc [NAME] command`` runs the command in a subshell
    (its ``cd`` ends with it) and makes NAME (``COPROC`` without one) an array of descriptors."""
    if _bash_version() < (4, 0):
        pytest.skip("coproc needs bash 4")
    (tmp_path / "sub").mkdir()
    for command in (f"coproc cat > victim {_OWNED}", f"coproc tee victim {_OWNED}",
                    f"coproc {{ cat > victim; }} {_OWNED}", f"coproc C {{ cat > victim {_OWNED}}}",
                    f"coproc C (cat > victim {_OWNED})",
                    f"N=C; coproc $N {{ cat > victim {_OWNED}}}",
                    f"coproc C if :; then cat > victim {_OWNED}fi",
                    f"! coproc cat > victim {_OWNED}", f"true | coproc cat > victim {_OWNED}",
                    f"coproc cd sub; cat > victim {_OWNED}"):
        assert _overwritten(tmp_path, command, ["victim", "sub/victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    assert _programs("coproc sleep 100") == ["sleep"]
    assert sleep_guard("coproc sleep 100") is not None
    for command, target in ((f'COPROC=victim; coproc cat; cat > "$COPROC" {_OWNED}', "$COPROC"),
                            (f'T=victim; coproc T {{ cat; }}; cat > "$T" {_OWNED}', "$T"),
                            (f'T=victim; N=T; coproc $N (cat); cat > "$T" {_OWNED}', "$T"),
                            # ``NAME_PID`` is the coprocess's process id
                            (f'COPROC_PID=victim; coproc cat; cat > "$COPROC_PID" {_OWNED}',
                             "$COPROC_PID"),
                            (f'C_PID=victim; coproc C {{ cat; }}; cat > "$C_PID" {_OWNED}',
                             "$C_PID"),
                            # one that ends with the group it is in, at no operator
                            (f'T=victim; {{ coproc T {{ cat; }} }}; cat > "$T" {_OWNED}', "$T"),
                            (f'T=victim; if :; then coproc T {{ cat; }} fi; cat > "$T" {_OWNED}',
                             "$T")):
        assert _overwritten(tmp_path, command, ["victim"]) == set(), command
        assert _verdict(command, {"victim"}) == _UNRESOLVED(path=target), command
    # The variables are the coprocess's as they were: its command knows them, and the
    # commands after it know the ones it does not name.
    for command in (f'T=victim; coproc cat > "$T" {_OWNED}',
                    f'T=victim; coproc C {{ T=other; }}; cat > "$T" {_OWNED}'):
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command


#: What bash rewrites after ``T=victim; <form>; cat > "$T"``: none of it by a literal assignment.
_BASH_SETS_T: tuple[tuple[str, set[str]], ...] = (
    ("wait -p T", set()), ("exec {T}>/dev/null", {"10"}), (": {T}<<< x", {"10"}),
    # options and builtins after which later commands are not what the text says
    ("shopt -s lastpipe; echo other | read T", {"other"}),
    ("shopt -s lastpipe; true | cd sub", {"sub/victim"}),
    ("shopt -s cdable_vars; S=sub; cd S", {"sub/victim"}),
    ("trap 'T=other' DEBUG; :", {"other"}), ("trap 'T=other' err; false", {"other"}),
    ("trap 'cd sub' DEBUG; :", {"sub/victim"}),
    ("alias x='T=other'; shopt -s expand_aliases\nx", {"other"}),
    ("set -o posix; alias x='T=other'\nx", {"other"}),
    ("enable -n export; export T=other", {"victim"}), ("enable -n cd; cd sub", {"victim"}),
    ("export() { :; }; export T=other", {"victim"}),
    ("export() { :; }; command export T=other", {"other"}),
    ("cd() { :; }; builtin cd sub", {"sub/victim"}),
    ("mapfile -C 'T=other; :' -c 1 lines <<< x", {"other"}),
    # a pipeline's last stage is a subshell, also where the group it ends closes the list
    ("{ true | { T=other; } }", {"victim"}), ("{ true | { cd sub; } }", {"victim"}),
    ("if :; then true | { T=other; } fi", {"victim"}),
    # arithmetic bash evaluates outside ``((...))``: a subscript, an offset, an integer's value
    ("declare -i U; U=T=5", {"5"}), ("declare -i U=T=5", {"5"}), ("a=(1 2); a[T=1]=x", {"1"}),
    ("declare -i U; declare U=T=5", {"5"}), ("declare -i U; readonly U=T=5", {"5"}),
    ("a=(1 2); : ${a[T=1]}", {"1"}), ('a=(1 2); : "${#a[T=1]}"', {"1"}),
    ("x=abc; : ${x:T=1}", {"1"}), ("x=abc; : ${x:0:T=1}", {"1"}), ("test -v 'a[T=5]'", {"5"}),
    ("printf -v 'a[T=5]' x", {"5"}), ("read 'a[T=5]' <<< x", {"5"}),
    ("a=(1); unset 'a[T=5]'", {"5"}), ("declare 'a[T=5]=x'", {"5"}), (": $[T=5]", {"5"}),
    ("U=T; (($U=5))", {"5"}), ("U=T; : $(($U=5))", {"5"}),
    ("T=; U=T; : $(( ${!U:=5} ))", {"5"}),
    # a builtin's here-document is expanded by the shell itself
    (": <<E\n${T:+}$[T=5]\nE", {"5"}), ("T=; : <<E\n${T:=other}\nE", {"other"}),
    # one that goes on over the lines of the body, or is longer than the scan reads of it
    ("x=abc; : <<END\n${x:0+\n(T=5)}\nEND", {"5"}),
    ("x=abc; : <<END\n${x:" + "0+" * 600 + "(T=5)}\nEND", {"5"}),
    # a quoted string in it that the line's end, or the end of what is read, cuts
    ('x=abc; : <<END\n${x:"0\n"+(T=5)}\nEND', {"5"}),
    ("x=abc; : <<END\n${x:" + "0+" * 497 + '"0000"+(T=5)}\nEND', {"5"}),
    ("cat <<E >/dev/null\n${U:-x} $[1+1] ${broken\nE", {"victim"}),
)
#: The same in a child shell that does not start as its command line reads.
_CHILD_SETS_T = (
    ("bash -O lastpipe -c 'T=victim; echo other | read T; cat > \"$T\"'", {"other"}),
    ("T=victim BASH_ENV=sub/env bash -c 'cat > \"$T\"'", {"other"}),
)
#: What leaves ``T`` as assigned: the scan keeps following it.
_BASH_KEEPS_T = (
    "trap 'T=other' EXIT", "trap -- 'T=other' 0", "shopt -s nullglob",
    "set -euo pipefail", "printf -v U x", "wait", "a[0]=x", "a=(1 2); : ${a[0]} ${T:1:2}",
    "export U=other", "exec 3>/dev/null", "cat <<E >/dev/null\n${U:-x} $[1+1]\nE",
    # an option set in a pipeline's stage ends with its subshell
    "true | shopt -s lastpipe",
    # text that is no arithmetic: a string's value, the value of an element
    'declare U="$HOME --all=T"', 'readonly U="$HOME -=T"', 'a[0]="$HOME --all=T"',
    'declare -A a; k=0; a[k]="$HOME++ T=other"', "U=T=other", "declare U=T=other",
)


def test_the_scanner_names_the_file_bash_writes_or_none(tmp_path: Path) -> None:
    """Audit 2026-11-12 B5 (owner), found by running bash beside the scanner: bash sets a
    variable or changes directory in ways the scan did not read (``wait -p T``, a ``{T}>``
    descriptor, a subscript's or an offset's arithmetic, a ``DEBUG`` trap, an alias, ``shopt -s
    lastpipe``), so the guard judged ``victim`` while bash wrote another file, or the reverse.
    The scan names the file bash writes, or leaves it to the shell."""
    files = ["victim", "other", "sub/victim", "1", "5", "10"]
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "env").write_text("T=other\n")
    commands = [(f'T=victim; {form}\ncat > "$T" {_OWNED}', expected)
                for form, expected in _BASH_SETS_T]
    commands += [(f"{child} {_OWNED}", expected) for child, expected in _CHILD_SETS_T]
    for command, expected in commands:
        if _bash_version() >= (5, 1):
            assert _overwritten(tmp_path, command, files) == expected, command
        named = [HEREDOC_VERDICT.format(path=f"/w/{name}") for name in expected]
        assert _verdict(command, set(files)) in (_UNRESOLVED(path="$T"), *named), command
    # A child shell's text is no option of its command line, whatever it starts with.
    # (Its first word is one only the shell knows unquoted, so the target is too; quoted,
    # with a literal name at its end, it is a program and the scan goes on.)
    for missing, verdict in (("$HOME/missing", _UNRESOLVED(path="$T")),
                             ('"$HOME/missing"', HEREDOC_VERDICT.format(path="/w/victim"))):
        child = f"bash -c '{missing} 2>/dev/null; T=victim; cat > \"$T\"' {_OWNED}"
        assert _overwritten(tmp_path, child, files) == {"victim"}
        assert _verdict(child, set(files)) == verdict, child
    for form in _BASH_KEEPS_T:
        command = f'T=victim; {form}\ncat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, files) == {"victim"}, command
        assert _verdict(command, set(files)) == HEREDOC_VERDICT.format(path="/w/victim"), command


#: FX45b F1/F2: a mover's name as data, a trap that sets no action: nothing is renamed or run.
_NO_MOVER = (
    "grep -c mv README.md", "echo ln", "which mv", "type ln", "command -v mv", "rg rename",
    "git grep mv", "man -w link", "git log --grep rename", "timeout 5 grep -c mv README.md",
    "uv run pytest -k rename", "find . -name mv", "git mv a b", "trap '' HUP", "trap - INT",
    "trap -- '' HUP INT", "trap -p", "for t in ln mv cp; do command -v $t; done",
)
#: FX45b F1/B3: the movers bash runs, each before ``cat >`` the name in the second column.
_MOVERS = (
    ("ln -s victim alias", "alias"), ("env ln -s victim alias", "alias"),
    ("command ln -s victim alias", "alias"), ("setsid -w ln -s victim alias", "alias"),
    ("find . -maxdepth 0 -exec ln -s victim alias ';'", "alias"),
    ('"$(dirname /bin/ln)/ln" -s victim alias', "alias"),
    ("ln -s -- victim -alias", "./-alias"), ("cp -l -- victim -x", "./-x"),
    # B4: in the script a shell reads, which the scan reads after the writer
    ("bash <<'S'\nln -s victim alias\nS", "alias"), ("bash -c 'ln -s victim alias'", "alias"),
)
#: FX45b B4: a mover in a substitution, which bash runs before the writer's redirection.
_SUBSTITUTED_MOVERS = (
    f"cat $(ln -s victim alias) > alias {_OWNED}", f"X=$(ln -s victim alias) cat > alias {_OWNED}",
    f"cat 2>$(ln -s victim alias; echo /dev/null) > alias {_OWNED}",
    "cat <<EOF > alias\nowned $(ln -s victim alias)\nEOF\n",
)
#: FX45b B5: a writer the shell does not wait for, and the mover written after it.
_RACING_WRITERS = (
    "coproc { sleep 0.3; cat >alias; } <<'EOF'\nowned\nEOF\n\nln -s victim alias; wait",
    "{ sleep 0.3; cat > alias; } <<'EOF' &\nowned\nEOF\nln -s victim alias; wait",
    "sleep 0.3 && cat > alias <<'EOF' &\nowned\nEOF\nln -s victim alias; wait",
    "{ sleep 0.3; cat; } <<'EOF' | { sleep 0.3; cat > alias; } & ln -s victim alias; wait\n"
    "owned\nEOF\n",
)


def test_a_mover_is_a_command_bash_runs_before_the_writer_opens_its_file(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45b (review of FX45, F1/F2/B3/B4/B5): any word ``mv``, ``ln``, ``cp``
    made every later target unresolvable (``grep -c mv README.md``), as did ``trap '' HUP``;
    while a mover's operand after ``--``, a mover in a shell's here-document script or in a
    substitution of the writer itself, and one written after a background or coprocess writer
    gave an existing file a new name unseen."""
    for form in _NO_MOVER:
        assert _verdict(f"{form}; cat > new.txt {_OWNED}", {"README.md", "victim"}) is None, form
    for form, name in _MOVERS:
        command = f"{form}\ncat > {name} {_OWNED}"
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
            (tmp_path / name).unlink()
        assert _verdict(command, {"victim"}) == _UNRESOLVED(path=name), command
    # ``git mv`` names its operands; behind another runner the mover's are the runner's to give
    assert _verdict(f"git mv victim alias; cat > alias {_OWNED}", {"victim"}) == _UNRESOLVED(
        path="alias")
    assert _verdict(f"git mv a b; cat > alias {_OWNED}", {"victim"}) is None
    assert _verdict(f"strace -o log ln a b; cat > new.txt {_OWNED}", set()) == _UNRESOLVED(
        path="new.txt")
    for command in (*_SUBSTITUTED_MOVERS, *_RACING_WRITERS):
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
            (tmp_path / "alias").unlink()
        assert _verdict(command, {"victim"}) == _UNRESOLVED(path="alias"), command
    # a writer bash waits for is done before a later mover; another name is no concern
    for command in (f"cat > new.txt {_OWNED}mv new.txt final.txt",
                    f"sleep 0.1 & cat > new.txt {_OWNED}mv new.txt final.txt",
                    "cat > alias <<'EOF' &\nowned\nEOF\nwait; ln -s victim other"):
        assert _verdict(command, {"victim"}) is None, command
    # an action in a lone operand only the shell knows may run a function unseen
    trap = ("f() { cat > victim; }; X=$(echo f USR1); trap $X; "
            f"{{ kill -USR1 $$; sleep 0.2; }} {_OWNED}")
    assert _overwritten(tmp_path, trap, ["victim"]) == {"victim"}
    assert _verdict(trap, {"victim"}) == HEREDOC_UNKNOWN_VERDICT.format(command="kill")


#: FX45b B1: a here-document on another descriptor than 0, and the command that copies it.
_OTHER_DESCRIPTOR = (
    "{ cat <&3 >victim; } 3<<'EOF'\nowned\nEOF\n", "cat /dev/fd/3 >victim 3<<'EOF'\nowned\nEOF\n",
    "f() { cat <&3 >victim; }; f 3<<'EOF'\nowned\nEOF\n",
    "bash -c 'cat <&3 >victim' 3<<'EOF'\nowned\nEOF\n", "cat 3<<'EOF' <&03 >victim\nowned\nEOF\n",
    "tee victim 3<<'EOF' 0<&3 >/dev/null\nowned\nEOF\n",
    "bash <<'S' 3<<'EOF'\ncat <&3 >victim\nS\nowned\nEOF\n",
    "cat 2<<'EOF' /dev/stderr >victim\nowned\nEOF\n",  # a name of the descriptor
)
#: The same where only the shell knows who reads it: the shell's own descriptor after an
#: ``exec``, one bash names, a script that is one of two here-documents or a descriptor's name.
_UNKNOWN_READERS = (
    ("exec 3<<'EOF'\nowned\nEOF\ncat <&3 >victim", "a command the shell runs on its own"),
    ("exec <<'EOF'\nowned\nEOF\ncat > victim", "a command the shell runs on its own"),
    ("exec {FD}<<'EOF'\nowned\nEOF\ncat <&$FD >victim", "a command the shell runs on its own"),
    ("bash <<'S' 11<<'D'\ncat > victim <<'EOF'\nowned\nEOF\nS\nx\nD\n",
     "a command the shell runs on its own"),
    ("bash /dev/fd/3 3<<'S'\ncat > victim <<'EOF'\nowned\nEOF\nS\n", "bash"),
)
#: FX45b B2: a descriptor bash numbers as the scan did not: a move (``3-``), leading zeros,
#: two digits, ``<>`` and ``<&`` on the standard output, a name of the standard output.
_DESCRIPTOR_SPELLINGS = (
    f"exec 3>>victim; cat >&3- {_OWNED}", f"exec 3>>victim; cat >&03 {_OWNED}",
    f"exec 03>>victim; cat >&3 {_OWNED}", f"exec 3>>victim; exec 4>&3-; cat >&4 {_OWNED}",
    f"cat 3>>victim 4>&3- >&4 {_OWNED}", f"exec 3>>victim; cat 1<&3- {_OWNED}",
    f"exec 3>>victim; exec 1>&3-; cat {_OWNED}", f"exec 3>>victim; {{ cat; }} >&3- {_OWNED}",
    f"exec 1<>victim; cat {_OWNED}", f"cat 01<>victim {_OWNED}", f"exec 01>>victim; cat {_OWNED}",
    f"exec 13>>victim; cat >&13 {_OWNED}", f"exec 3>>victim; exec 1<&3; cat {_OWNED}",
    # the output still goes down the pipe: descriptor 11 is not the standard output, a
    # name of the standard output is the pipe, and so is a copy of it put back
    *(f"cat {sink} <<'EOF' | tee victim >/dev/null\nowned\nEOF\n"
      for sink in ("11>new", "> /dev/stdout", "3>&1 >new >&3")),
)


def test_a_descriptor_is_the_one_bash_numbers(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45b (review of FX45, B1/B2): a here-document on descriptor 3 was no
    one's input, so ``cat <&3 > victim`` under it passed, and ``>&3-``, ``>&03``, ``01<>`` or a
    two-digit descriptor named no file the scan knew: bash rewrote ``victim`` each time."""
    refused = HEREDOC_VERDICT.format(path="/w/victim")
    for command in (*_OTHER_DESCRIPTOR, *_DESCRIPTOR_SPELLINGS):
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == refused, command
    for command, reader in _UNKNOWN_READERS:
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_UNKNOWN_VERDICT.format(
            command=reader), command
    # a descriptor that a piped group made a copy of its output is the pipe
    piped = "{ cat >&3 <<'EOF'\nowned\nEOF\n} 3>&1 | tee victim >/dev/null\n"
    assert _overwritten(tmp_path, piped, ["victim"]) == {"victim"}
    assert _verdict(piped, {"victim"}) == _UNRESOLVED(path="&1")
    # no copy of the text: a descriptor ``cat`` does not read, another program's script or
    # data, the streams the tool gave the command
    for command in ("cat 3<<EOF > victim\nowned\nEOF\n",
                    "diff /dev/fd/3 /dev/fd/4 3<<'A' 4<<'B'\nx\nA\ny\nB\n",
                    "python3 /dev/fd/3 3<<'EOF'\npass\nEOF\n", f"cat >&2 {_OWNED}",
                    f"cat > /dev/stderr {_OWNED}", f"exec 3>&1; cat >&3 {_OWNED}",
                    f"cat >& new.txt {_OWNED}", "cat 2>&1 <<'EOF' | tee new.log\nowned\nEOF\n"):
        assert _verdict(command, {"victim"}) is None, command
    assert _overwritten(tmp_path, "cat 3<<EOF > victim\nowned\nEOF\n", ["victim"]) == {"victim"}
    assert (tmp_path / "victim").read_text() == ""  # emptied, as ``: > victim`` does: no copy


#: FX45b: a copier fed a here-document inside ``<(...)``: the copier that holds the
#: substitution, or one in the group it is the standard input of, reads the text.
_STREAM_COPIES = (
    f"tee victim >/dev/null < <(cat {_OWNED})\n", f"cat <(cat {_OWNED}) > victim\n",
    "cat <(cat <<'EOF' | cat\nowned\nEOF\n) > victim\n", f"cat <({{ cat; }} {_OWNED}) > victim\n",
    f"f() {{ cat; }}; cat <(f {_OWNED}) > victim\n", f"cat <(bash -c cat {_OWNED}) > victim\n",
    f"cat < <(cat <(cat {_OWNED})) >> victim\n", f"env cat -- <(cat {_OWNED}) | tee victim\n",
    f"{{ cat; }} < <(cat {_OWNED}) > victim\n",
    f"if true; then tee victim >/dev/null; fi 0< <(cat {_OWNED})\n",
)
#: The same where only the shell knows who reads the stream: a descriptor the shell or a
#: group keeps, a shell or a function that holds it, the output ``>(...)`` inherits.
_STREAMS = (
    f"exec 3< <(cat {_OWNED}); cat <&3 > victim\n",
    f"{{ cat <&3 > victim; }} 3< <(cat {_OWNED})\n",
    f"for f in <(cat {_OWNED}); do cat \"$f\" > victim; done\n",
    f"f() {{ cat \"$1\" > victim; }}; f <(cat {_OWNED})\n",
    f"bash -c 'cat \"$0\" > victim' <(cat {_OWNED})\n",
    f"{{ : > >(cat {_OWNED}); wait $!; }} >> victim\n",
    f"exec >> victim; : > >(cat {_OWNED}); wait $!\n",
)


def test_the_output_of_a_process_substitution_is_a_stream(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45b: what a ``cat`` fed a here-document wrote inside ``<(...)`` or
    ``>(...)`` was taken for a word, as a command substitution's output is: the command that
    read the stream, or the output the shell had there, put the text in ``victim``."""
    for command in _STREAM_COPIES:
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == HEREDOC_VERDICT.format(path="/w/victim"), command
    for command in _STREAMS:
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == _UNRESOLVED(
            path="a process substitution's output"), command
    # no copy into an existing file: a new one, a program that is no copier reads the
    # stream, no copier in the substitution is fed the here-document
    for command in (f"cat <(cat {_OWNED}) > new.txt\n", f"grep -c o <(cat {_OWNED}) > new.txt\n",
                    f"while read -r line; do echo \"$line\"; done < <(cat {_OWNED}) > new.txt\n",
                    f"python3 <(echo 'print(1)') {_OWNED}", f"grep -c o <(python3 - {_OWNED})\n",
                    f"diff <(sort a) <(sort b) > new.diff; cat > new.txt {_OWNED}",
                    f"cat <(sort victim) > new.txt; tee new.log >/dev/null {_OWNED}"):
        assert _verdict(command, {"victim"}) is None, command
    # bash expands a redirection's operand with the redirections before it done: a command
    # substituted in one written after the here-document has it as its input
    unseen = HEREDOC_UNKNOWN_VERDICT.format(command="a command the shell runs on its own")
    for head in (": <<'EOF' < <(cat >> victim)", ": <<'EOF' 3< <(tee victim >/dev/null)",
                 ': <<\'EOF\' 2> "$(cat >> victim; echo /dev/null)"',
                 "{ :; } <<'EOF' < <(cat >> victim)", ": 3<<'EOF' > >(cat <&3 >> victim)"):
        command = f"{head}\nowned\nEOF\nwait $!\n"
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) == unseen, command
    # the substitution ran before the here-document was opened; the input of ``>(...)`` is the pipe
    for command in (f": < <(cat >> victim) {_OWNED}wait $!\n",
                    ": <<'EOF' > >(cat >> victim)\nowned\nEOF\nwait $!\n",
                    "python3 - <<'EOF' 2> >(grep -v Warning >&2)\nprint(1)\nEOF\n"):
        assert _overwritten(tmp_path, command, ["victim"]) == set(), command
        assert _verdict(command, {"victim"}) is None, command


#: FX45b B6: what assigns ``X`` to a variable bash starts with as an integer: it evaluates it.
_INTEGER_ASSIGNMENTS = (
    "RANDOM=X", "export OPTIND=X", "declare HISTCMD=X", "{ SRANDOM=X; }", "readonly RANDOM=X",
    "declare SECONDS=X", "{ printf -v OPTIND %s X; }", "{ for RANDOM in X; do :; done; }",
)


def test_an_integer_bash_starts_with_evaluates_what_it_is_assigned(tmp_path: Path) -> None:
    """Audit 2026-11-21 FX45b (review of FX45, B6): ``RANDOM``, ``SRANDOM``, ``OPTIND``,
    ``HISTCMD``, ``SECONDS`` and ``BASHPID`` are integers without a ``declare -i``, so
    ``RANDOM=X`` evaluates the value of ``X``: a subscript in it ran ``cat > victim`` under
    the here-document, and an assignment in it moved the target (``Y=T=1; RANDOM=Y``)."""
    for form in _INTEGER_ASSIGNMENTS:
        command = f"X='A[$(cat >victim)]'; {form} {_OWNED}"
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) is not None, command
    for form in ("RANDOM=Y", "export OPTIND=Y"):
        command = f'T=0; Y=T=1; {form}; cat > "$T" {_OWNED}'
        if _oracle_runs(command):
            assert _overwritten(tmp_path, command, ["0", "1"]) == {"1"}, command
        assert _verdict(command, {"0", "1"}) == _UNRESOLVED(path="$T"), command
    # a number as written evaluates to itself, as does a variable that holds one
    for form in ("SECONDS=0", "OPTIND=1", "RANDOM=42", "X=1; RANDOM=X", "LINENO=X"):
        assert _verdict(f"{form}; cat > new.txt {_OWNED}", {"victim"}) is None, form


def _cdpath_sub() -> str:
    """The ``CDPATH`` the shells of another environment inherit: ``sub``."""
    return "sub"


def test_a_cd_looks_in_the_cdpath_the_shell_inherits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-21 FX45b (review of FX45, B7): with ``CDPATH=sub`` in the environment the
    tool hands its shell, ``cd d && cat > victim`` rewrote ``sub/d/victim`` while the guard
    judged ``d/victim``: a ``CDPATH`` the text never assigned was taken to be empty."""
    (tmp_path / "sub" / "d").mkdir(parents=True)
    victim = tmp_path / "sub" / "d" / "victim"
    work = str(tmp_path)
    searched = f"cd d && cat > victim {_OWNED}"
    victim.write_text("ORIGINAL\n")
    subprocess.run(["bash", "-c", searched], cwd=tmp_path, check=True, capture_output=True,
                   timeout=10, env={"PATH": os.environ["PATH"], "CDPATH": "sub"})
    assert victim.read_text() == "owned\n"
    monkeypatch.setenv("CDPATH", "sub")
    assert heredoc_guard(searched, work, os.path.lexists, os.path.isdir) == _UNRESOLVED(
        path="victim")
    # what bash does not look up there, or the text decides, is judged as written
    refused = HEREDOC_VERDICT.format(path=str(victim))
    for command in (f"cd ./sub/d && cat > victim {_OWNED}",
                    f"CDPATH=; cd sub/d && cat > victim {_OWNED}",
                    f"cd {tmp_path}/sub/d && cat > victim {_OWNED}"):
        assert heredoc_guard(command, work, os.path.lexists, os.path.isdir) == refused, command
    assert heredoc_guard(f"cat > new.txt {_OWNED}", work, os.path.lexists, os.path.isdir) is None
    # another environment's shell inherits its own: the caller tells which
    monkeypatch.delenv("CDPATH")
    assert heredoc_guard(f"cd d && cat > new.txt {_OWNED}", work, os.path.lexists) is None
    assert heredoc_guard(searched, work, os.path.lexists, cdpath=_cdpath_sub) == _UNRESOLVED(
        path="victim")


#: A program that logs its first operand and how many it got.
_MARK = '#!/bin/sh\necho "$1 $#" >> "$MARKS"\n'
#: Commands whose calls of ``mark`` the scan reads as bash makes them.  ``mark NOT`` is text
#: of a string or of a here-document body: bash never runs it.
_ANSI_C_STRINGS = (
    r"echo $'it\'s' ; mark SECOND",  # the two rows of the FX05c request, and its nested one
    r"mark -rf $'don\'t # x' /tmp/w",
    r"""echo "$(echo $'a\'b'; mark SECOND)" """,
    r"echo $'a\\'; mark SECOND; echo $'a\\\'b'; mark THIRD 'x y'",  # ``\\`` escapes no quote
    r"""echo "$'a\'; mark NOT; echo '" 'a\'; mark SECOND; echo 'b'""",  # no ``$'...'`` there
    r"X=$'a\'b' mark SECOND && echo hi > $'a\'b'; mark THIRD < $'a\'b' | cat; mark FOURTH",
    r"echo $'a\' | mark NOT `mark NOT` $(mark NOT) # ' ; mark SECOND",
    "echo $'a\\'\nmark NOT\n' ; mark SECOND",
    r"echo `echo $'a\'b)'; mark SECOND` $(echo $'a\')'; mark THIRD)",
    r"for i in $'a\'b'; do mark SECOND; done; case x in $'a\'b') ;; *) mark THIRD;; esac",
    r"""f() { a=($'a\'b)' c); mark SECOND "${a[0]}"; }; f; declare -a b=($'\'' y); mark THIRD""",
    r"[[ x == $'a\'b' || x =~ $'a\'b' ]] || mark SECOND; (( 1 + $'1\'' )); mark THIRD $'a\'b'",
    r"echo ${x:-$'a\'b}'} ${x#$'a\'b'} ${x/$'a\'b'/$'c\'d'} ${x:+$'a\'b'}; mark SECOND",
    r"""x=1; echo "${x:+$'a\'b $(mark FIRST)\' c'}"; mark SECOND""",
    r"""eval 'echo $'\''a\'\''b'\''; mark SECOND'; bash -c "echo \$'a\\'b'; mark THIRD" """,
    "cat <<< $'a\\'b'; mark SECOND",
    r"""echo "${x:-$'\n\t\a\$(mark NOT)\`mark NOT\`'}"; mark SECOND""",
    # the parser ends the string at its quote whatever it holds: bash fails the first line
    "echo \"${x:-$'a\\'b'}\"\nmark SECOND",
    "echo \"${x:-$'${y'}\"\nmark SECOND; echo '}'",
    "echo \"${x:-$'$('}\"\nmark SECOND; echo ')'",
    "echo \"${x:-$'$(cat <<E)'}\"\nmark SECOND\nE",  # that here-document has no body
    # a here-document delimiter is the string bash's parser decoded
    "cat <<$'E\\'OF'\n$(mark NOT)\nE'OF\nmark SECOND",
    "cat <<x$'E\\x4f\\106\\t\\\\\\\"\\?\\q\\x{141}'y\nmark NOT\nxEOF\t\\\"?\\qAy\nmark SECOND",
    "cat <<$'\\a\\b\\e\\E\\f\\r\\v\\x\\$'\nmark NOT\n\a\b\x1b\x1b\f\r\v\\x\\$\nmark SECOND",
    "cat <<\"$'E\"\n$(mark NOT)\n$'E\nmark SECOND",  # no string in double quotes
    # bash looks for the delimiter of ``<<-`` in the line as written, then without its tabs
    "cat <<-$'\\tE'\n\tmark NOT\n\tE\nmark SECOND",
    "cat <<-'\tE'\n\tmark NOT\n\tE\nmark SECOND",
    "cat <<-E\n\tmark NOT\n\tE\nmark SECOND",
)
#: Bash 5.2 quotes such a string in a pattern; the versions before it fail the line.
_ANSI_C_STRINGS_52 = (
    r"""x="a'bc"; echo "${x#$'a\'b'}" "${x/$'a\'b'/$'c\'d'}"; mark SECOND""",)
#: Text bash decodes and then expands: it runs ``mark FIRST``, which the scan cannot read.
_DECODED_COMMANDS = (
    r"""echo "${x:-$'\x24(mark FIRST)'}"; mark SECOND""",
    r"""echo "${x:-$'\\\$(mark FIRST)'}"; mark SECOND""",
    r"""echo "${x:-$'\c\$(mark FIRST)'}"; mark SECOND""",
    r"""echo "${x:-$'$(mark\tFIRST)'}"; mark SECOND""",
    r"""x=abc; echo "${x:$'\044(mark FIRST; echo 1)'}"; mark SECOND""",
)
#: Delimiters only bash can decode: its versions, or two locales, make different strings.
_UNREAD_DELIMITERS = ("$'E\\cAF'", "$'E\\u004fF'", "$'E\\U0000004fF'", "$'E\\xc3\\xa9'",
                      "$'E\\0F'", "$'E\\x{}F'", "$'E\\001F'", "$'E\\177F'", '$"EOF"', 'x$"EOF"')


def _marks(tmp_path: Path, command: str) -> list[str]:
    """The first operand and the number of operands of each call of ``mark`` bash makes."""
    program = tmp_path / "bin" / "mark"
    program.parent.mkdir(exist_ok=True)
    program.write_text(_MARK)
    program.chmod(0o755)
    log = tmp_path / "marks"
    log.write_text("")
    subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, check=False,
                   timeout=10, stdin=subprocess.DEVNULL,
                   env={"PATH": f"{program.parent}:{os.environ['PATH']}", "MARKS": str(log)})
    return log.read_text().splitlines()


def _marked(command: str) -> list[str]:
    """The same of each ``mark`` command the scan reads in *command*."""
    return [f"{simple.argv()[1].text} {len(simple.argv()) - 1}"
            for simple in simple_commands(command, "/w") if simple.program() == "mark"]


def test_an_ansi_c_string_ends_at_its_first_unescaped_quote(tmp_path: Path) -> None:
    """Audit 2026-11-21 LI (leftover I1, the FX05c request): the scan ended a ``$'...'`` at an
    escaped quote, or past its quote in an expansion left open in it, so the commands after
    it were glued into the word or read as a comment and no guard saw them; a here-document
    delimited by such a string never ended; and text bash decodes and then expands
    (``"${x:-$'\\x24(cmd)'}"``) ran unseen."""
    recent = _ANSI_C_STRINGS_52 if _bash_version() >= (5, 2) else ()
    for command in (*_ANSI_C_STRINGS, *recent):
        calls = _marks(tmp_path, command)
        assert calls and "NOT 1" not in calls, command
        assert _marked(command) == calls, command
    for command in _DECODED_COMMANDS:
        assert _marks(tmp_path, command) == ["FIRST 1", "SECOND 1"], command
        scanned = simple_commands(command, "/w", derived=True)
        assert any(simple.unknown and not simple.words for simple in scanned), command
        assert _marked(command) == ["SECOND 1"], command
    # each of these hid a rewrite from the here-document guard; in the last one the quote the
    # decoded text leaves open fails the line, so ``T=new.txt`` never runs
    for command in (f"echo $'it\\'s'; cat > victim {_OWNED}",
                    f"cat <<$'EOF'\nx\nEOF\ncat > victim {_OWNED}",
                    ": <<EOF > \"${x:-$'\\x24(cat > victim)'}\"\nowned\nEOF\n",
                    f"T=victim; echo \"${{x:-$'a\"b'}}\"; T=new.txt\ncat > \"$T\" {_OWNED}"):
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        assert _verdict(command, {"victim"}) is not None, command
    # plain text and letter escapes are what they show: the scan still knows what it knew
    plain = f"T=new.txt; x=\"${{T%$'\\n'}}${{T:-$'\\t\\e\\?\\q'}}\"; cat > \"$T\" {_OWNED}"
    assert _overwritten(tmp_path, plain, ["victim"]) == set()
    assert _verdict(plain, {"victim"}) is None
    # where only bash knows the end of a here-document, the text after it is unread: refused
    for delimiter in _UNREAD_DELIMITERS:
        command = f"mark FIRST; cat <<{delimiter}\nx\nEOF\ncat > victim {_OWNED}"
        with pytest.raises(TooDeepError) as raised:
            simple_commands(command, "/w")
        assert str(raised.value) == UNREAD_DELIMITER_VERDICT, command
        with pytest.raises(TooDeepError):
            _verdict(command, {"victim"})
        assert [simple.program() for simple in simple_commands(
            command, "/w", optimistic=True)] == ["mark", "cat"], command


#: How a command hands a text to a shell, to ``eval`` or to a trap: ``{}`` is the text, one
#: quoted string.  Plain-quoted, the scan reads the text.
_TEXT_FORMS = (
    "eval {}", "bash -c {}", "sh -c {}", "bash -ec {} arg0", "eval eval {}", "command eval {}",
    "$'eval' {}", "bash $'-c' {}", "e=eval; $e {}", "trap {} EXIT",
    'X={}; eval "$X"', 'X={}; bash -c "$X"', 'X=""{}; eval "$X"', 'X={}; Y="$X"; eval "$Y"',
    "export X={}; bash -c 'eval \"$X\"'", "X={} bash -c 'eval \"$X\"'",
    # text only the shell knows, in which a here-document shows: the shell may copy it
    "eval ${{Y:-{}}}", 'eval "$(echo {})"',
)
#: Words of ``$'...'`` strings every bash decodes alike: quoted text, whatever it holds.
_DECODED_WORDS = (
    r"$'a b\tc\nd' $'' x$''y a$'b c'd",
    r"$'\a\b\e\E\f\r\v\\\'\"\?' $'\101\x42\x{43}\1011\x444\x7e\2'",
    r"$'\q\x\8\ \$' $'*' $'~' $'{a,b}' $'$HOME`x`$(x)' " "$'a\\\nb'",
)
#: Escapes two versions of bash, or two locales, decode differently; ``\cJ`` is a newline.
_UNDECODED = ("\\cJ", "\\u000a", "\\U0000000a", "\\0", "\\x{}", "\\1", "\\177", "\\x80")


def test_an_ansi_c_string_is_the_text_bash_decodes(tmp_path: Path) -> None:
    """Audit 2026-11-22 LH3 (the side finding of LH2): the scan kept a ``$'...'`` word as
    written, a value only the shell knows, so the text of ``eval $'cat > victim <<EOF\\n...'``
    or ``bash -c $'...'`` was never read; its ``\\n`` is no newline as written, so the net for
    a here-document hidden in such a word missed it too, and bash rewrote ``victim``."""
    text = "cat > victim <<EOF\nowned\nEOF"
    string = "$'" + text.replace("\n", "\\n") + "'"
    for form in _TEXT_FORMS:
        command = form.format(string)
        assert _overwritten(tmp_path, command, ["victim"]) == {"victim"}, command
        verdict = _verdict(command, {"victim"})
        assert verdict is not None, command
        assert verdict == _verdict(form.format(f"'{text}'"), {"victim"}), command
    # the text is read, not merely refused: a copy into a new file runs, a ``cd`` in it counts
    moved = "bash -c $'cd sub\\ncat > victim <<EOF\\nowned\\nEOF'"
    assert _overwritten(tmp_path, moved, ["victim", "sub/other"]) == set()
    assert _verdict(moved, {"victim"}) is None
    assert _verdict(moved, {"sub/victim"}) == HEREDOC_VERDICT.format(path="/w/sub/victim")
    assert _verdict("X=$'cat > new.txt <<EOF\\nfresh\\nEOF'; eval \"$X\"", {"victim"}) is None
    # bash settles what each string decodes to (its bytes: no ``\r`` is lost); the word itself
    # stays as written, and not literal, for the guards that decode it themselves
    for words in _DECODED_WORDS:
        command = f"printf '<%s>' {words}"
        scanned = simple_commands(command, "/w")[0].words[2:]
        assert " ".join(word.text for word in scanned) == words, words
        assert not any(word.literal or word.path for word in scanned), words
        printed = subprocess.run(["bash", "-c", command], capture_output=True, check=True,
                                 timeout=10).stdout.decode()
        assert "".join(f"<{word.passed()}>" for word in scanned) == printed, words
    # a string only bash can decode stays a value only the shell knows; handed to a command
    # that may run it, it may hold a here-document
    unknown = HEREDOC_UNKNOWN_VERDICT.format(command="a command the shell runs on its own")
    newline = "eval $'cat > victim <<EOF\\cJowned\\cJEOF'"
    assert _overwritten(tmp_path, newline, ["victim"]) == {"victim"}
    for escape in _UNDECODED:
        string = f"$'cat > victim <<EOF{escape}owned\\nEOF'"
        word = simple_commands(f"echo {string}", "/w")[0].words[1]
        assert (word.text, word.literal, word.passed()) == (string, False, None), escape
        assert _verdict(f"echo {string}", {"victim"}) is None, escape
        for form in ("eval {}", "bash -c {}", "$RUN {}", "trap {} EXIT"):
            assert _verdict(form.format(string), {"victim"}) == unknown, (form, escape)
    # an escape no bash makes an ASCII character of hides neither the operator nor a newline:
    # the command runs, as its plain-quoted spelling does, unless the here-document shows
    for command in ("$RUN $'caf\\u00e9'",
                    "bash -c $'echo \\xe2\\x9c\\x93 \\342\\234\\223 \\U0001f600'"):
        assert _verdict(command, {"victim"}) is None, command
    shown = "eval $'cat > victim <<EOF\\nowned \\u00e9\\nEOF'"
    assert _overwritten(tmp_path, shown, ["victim"]) == {"victim"}
    assert _verdict(shown, {"victim"}) == unknown
    assert _verdict("$RUN $'\\uff1c\\uff1cEOF\\n'", {"victim"}) == unknown  # a wide ``<``
