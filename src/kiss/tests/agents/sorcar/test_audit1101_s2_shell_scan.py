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
