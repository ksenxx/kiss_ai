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
import subprocess
import timeit
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
    rewrite_guard,
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


def _overwritten(tmp_path: Path, command: str, files: list[str]) -> set[str]:
    """The *files* (relative, holding ORIGINAL) that bash, run in *tmp_path*, rewrites.

    ``HOME`` is ``tmp_path/home``, so a ``~`` target is under it.
    """
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("ORIGINAL\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, check=False,
                   timeout=10, env=env)
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
                    "export T=existing; /bin/bash -lc 'cat > $T <<EOF\nowned\nEOF\n'",
                    "cd sub && bash -c 'cat <<EOF > existing\nowned\nEOF\n'"):
        assert _overwritten(tmp_path, command, ["existing", "sub/existing"]) == {
            "sub/existing" if command.startswith("cd ") else "existing"}, command
        assert _verdict(command, {"existing", "sub/existing"}) == HEREDOC_VERDICT.format(
            path="/w/sub/existing" if command.startswith("cd ") else "/w/existing"), command
    assert sleep_guard("bash -c 'sleep 60'") is not None
    assert simple_commands("bash -c 'cd sub'; cat > victim", "/w")[-1].cwd == "/w"
    assert _programs("bash -c \"$T\"; bash -c 'date'; docker exec c sh -c 'date'") == [
        "bash", "bash", "date", "docker"]


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


def test_the_assignments_of_a_command_are_read_in_one_pass() -> None:
    """Audit 2026-11-10 S10-F3, review round 1: what the earlier assignments of a command set
    is kept as the scan goes; replayed for every word, 8,000 of them took 18 seconds."""
    known = "; ".join(f"T{i}=x" for i in range(4000))
    for command in (" ".join(["T=x"] * 8000),
                    " ".join(f"T{i}=$T{i - 1}" for i in range(8000)),
                    f"{known}; " + " ".join(f"T{i}=$T{i - 1}" for i in range(1, 4000)),
                    # arithmetic that assigns nothing: one long name is read once
                    "test 'a[" + "a" * 12000 + "]'", "(( " + "a" * 12000 + " ))"):
        scan = functools.partial(simple_commands, command, "/w")
        assert min(timeit.repeat(scan, number=1, repeat=3)) < 0.5, command[:40]


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


def test_the_operands_of_let_are_arithmetic_expressions(tmp_path: Path) -> None:
    """Audit 2026-11-10 B/S10-F4: ``let ++T`` and ``let "0,T=2"`` assign ``T``; the scanner read
    each operand as a variable's name, found none and kept the old value."""
    for operand, written in (("++T", "2"), ('"0,T=2"', "2"), ("--T", "0"), ('"$S2_UNSET"', "1")):
        command = f'T=1; let {operand}; cat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, [written]) == {written}, command
        assert _verdict(command, {written}) == _UNRESOLVED(path="$T"), command
    kept = f'T=1; let V=2 W++; cat > "$T" {_OWNED}'
    assert _overwritten(tmp_path, kept, ["1"]) == {"1"}
    assert _verdict(kept, {"1"}) == HEREDOC_VERDICT.format(path="/w/1")


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
    "trap 'T=other' EXIT", "trap -- 'T=other' INT TERM", "shopt -s nullglob",
    "set -euo pipefail", "printf -v U x", "wait", "a[0]=x", "a=(1 2); : ${a[0]} ${T:1:2}",
    "export U=other", "exec 3>/dev/null", "cat <<E >/dev/null\n${U:-x} $[1+1]\nE",
    # an option set in a pipeline's stage ends with its subshell
    "true | shopt -s lastpipe",
    # text that is no arithmetic: a string's value, the value of an element
    'declare U="$HOME --all=T"', 'readonly U="$HOME -=T"', 'a[0]="$HOME --all=T"',
    'declare -A a; a[k]="$HOME++ T=other"', "U=T=other", "declare U=T=other",
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
    child = f"bash -c '$HOME/missing 2>/dev/null; T=victim; cat > \"$T\"' {_OWNED}"
    assert _overwritten(tmp_path, child, files) == {"victim"}
    assert _verdict(child, set(files)) == HEREDOC_VERDICT.format(path="/w/victim")
    for form in _BASH_KEEPS_T:
        command = f'T=victim; {form}\ncat > "$T" {_OWNED}'
        assert _overwritten(tmp_path, command, files) == {"victim"}, command
        assert _verdict(command, set(files)) == HEREDOC_VERDICT.format(path="/w/victim"), command
