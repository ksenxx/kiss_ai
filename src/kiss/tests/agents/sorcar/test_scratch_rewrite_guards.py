# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (task 4aa14638), proposal 2: shell rewrites of scratch files run.

63 of the 68 shell-rewrite refusals of the audited task were on files
under ``tmp/`` (probe scripts, ``PROGRESS.md``, logs), most through a
variable (``T=tmp/4aa; cat > $T/out.txt``) the scanner could not
resolve.  Two changes, bash and GNU sed being the oracles:

* the scanner follows a variable the command itself assigned a literal
  value, in the same scope, so ``$T`` / ``${T}`` targets resolve;
* a ``sed -i``, python heredoc or here-document rewrite of a scratch file
  (:func:`kiss.agents.sorcar.useful_tools._is_scratch_path`) runs as
  written, as ``Write`` overwrites one unread; a tracked file keeps
  the refusal naming the ``Edit`` to make.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
    python_written_paths,
    rewrite_guard,
)
from kiss.agents.sorcar.shell_scan import SimpleCommand, resolve, simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import host_sed_is_gnu, posix_only

pytestmark = posix_only("bash, GNU sed and python3 are the oracles")

_BODY = "<<'EOF'\nbody\nEOF\n"


def _created(tmp_path: Path, command: str) -> set[str]:
    """The files *command* leaves under *tmp_path* when bash runs it there."""
    subprocess.run(["bash", "-c", command], cwd=tmp_path, check=False, capture_output=True,
                   env={"PATH": "/usr/bin:/bin"})
    return {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("command, expected", [
    ("T=tmp/x; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    ("export T=tmp/x; mkdir -p $T && cat > ${T}/f " + _BODY, "tmp/x/f"),
    ('D="my dir"; mkdir -p "$D"; cat > "$D"/f ' + _BODY, "my dir/f"),
    ("T=a; T=tmp/x; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    # a subshell's assignment ends with it; the outer one holds
    ("T=tmp/x; (T=other); mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    # the value of a backquoted command's variable comes from the outer command
    ("T=tmp/x; mkdir -p $T; echo `cat > $T/f " + _BODY + "`", "tmp/x/f"),
    # a pipeline segment's and a backgrounded list's assignments end with them,
    # a brace group's included
    ("T=tmp/x; T=other | cat; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    ("T=tmp/x; { T=other; } | cat; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    ("T=tmp/x; T=other && true & wait; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    ("T=tmp/x; { T=other; } & wait; mkdir -p $T; cat > $T/f " + _BODY, "tmp/x/f"),
    # inside a brace group after &&, its assignments hold for its own commands
    ("mkdir -p tmp/x && { T=tmp/x; cat > $T/f " + _BODY + "}", "tmp/x/f"),
    # builtins assigning other variables, and an arithmetic command, leave T alone
    ("T=tmp/x; read -r X <<< y; unset X; (( i = 1 )); mkdir -p $T; cat > $T/f " + _BODY,
     "tmp/x/f"),
    # as do another variable's array value, read's valued options and a subshell
    ("T=tmp/x; Y=(a b); read -p 'a' -n 1 X <<< y; (T=z); mkdir -p $T; cat > $T/f " + _BODY,
     "tmp/x/f"),
])
def test_assigned_variables_resolve_to_the_file_bash_writes(
    tmp_path: Path, command: str, expected: str,
) -> None:
    assert heredoc_targets(command) == [expected]
    assert _created(tmp_path, command) == {expected}
    [target] = [c.outputs[0] for c in simple_commands(command, "/w") if c.outputs]
    assert resolve(target_cwd(command), target) == f"/w/{expected}"


def target_cwd(command: str) -> str | None:
    """The directory the ``cat`` of *command* runs in, as the scanner sees it."""
    return next(c.cwd for c in simple_commands(command, "/w") if c.program() == "cat")


@pytest.mark.parametrize("command", [
    "cat > $T/f " + _BODY,  # never assigned here: the environment's
    "T=tmp/x || T=y; cat > $T/f " + _BODY,  # the second may run
    "if true; then T=tmp/x; fi; cat > $T/f " + _BODY,
    "T=tmp; T+=/x; cat > $T/f " + _BODY,  # appends to a value the scan may not know
    "T=$HOME/x; cat > $T/f " + _BODY,  # a value only the shell knows
    "T=tmp/x; cat > ${T:-y}/f " + _BODY,  # not a bare variable
    # an assignment after && runs only when the command before it succeeded,
    # as does one in a brace group after && or ||
    "T=notes.md; false && T=tmp/x; cat > $T/f " + _BODY,
    "T=notes.md; false && { :; T=tmp/x; }; cat > $T/f " + _BODY,
    "T=notes.md; true || { :; T=tmp/x; }; cat > $T/f " + _BODY,
    # a variable set by a builtin, a loop, an arithmetic command or sourced text
    "T=tmp/x; read -r T <<< notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; printf -v T notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; unset T; cat > $T/f " + _BODY,
    "T=tmp/x; declare T=notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; for T in notes.md; do :; done; cat > $T/f " + _BODY,
    "T=tmp/x; (( T = 1 )); cat > $T/f " + _BODY,
    "T=tmp/x; eval 'T=notes.md'; cat > $T/f " + _BODY,
    "T=tmp/x; . tmp/env.sh; cat > $T/f " + _BODY,
    # the variable a builtin fills when no operand names one (bash's REPLY, MAPFILE)
    "REPLY=tmp/x; read <<< notes.md; cat > \"$REPLY\" " + _BODY,
    "REPLY=tmp/x; read -r -p '' <<< notes.md; cat > $REPLY/f " + _BODY,
    "MAPFILE=tmp/x; mapfile <<< notes.md; cat > $MAPFILE/f " + _BODY,
    "MAPFILE=tmp/x; readarray -t <<< notes.md; cat > $MAPFILE/f " + _BODY,
    "REPLY=tmp/x; select x in a; do break; done <<< notes.md; cat > $REPLY/f " + _BODY,
    # the array name attached to read's -a; an array made by an element
    # assignment, a value in parentheses or an appended one; a name reference;
    # readonly
    "T=tmp/x; read -aT <<< notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; read -r -d '' -raT <<< notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; T[0]=notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; T=(notes.md); cat > $T/f " + _BODY,
    "T=tmp/x; T=( 'notes.md' \"x)\" ); cat > $T/f " + _BODY,
    "T=tmp/x; T+=(notes.md); cat > $T/f " + _BODY,
    "T=tmp/x; declare -n R=T; R=notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; local -gn R=T; R=notes.md; cat > $T/f " + _BODY,
    "T=tmp/x; readonly T=notes.md; cat > $T/f " + _BODY,
    # unquoted, a value with a blank or a pattern is several words to bash
    'T="tmp/x notes.md"; cat > $T/f ' + _BODY,
    "T=tmp/*; cat > $T/f " + _BODY,
])
def test_values_only_the_shell_knows_stay_unresolved(command: str) -> None:
    assert heredoc_guard(command, "/w", lambda path: True) == (
        HEREDOC_UNRESOLVED_VERDICT.format(path=heredoc_targets(command)[0]))


def test_an_optimistic_scan_keeps_what_a_list_assigned() -> None:
    """The gate reads a command as it was meant to run: an assignment or a ``cd``
    after ``&&`` or ``||`` holds past the ``;`` that ends its list (the 2026-11-03
    audit's ``cd d && T=tmp/x && a; cp b ../../$T/``); a guard keeps forgetting
    them, since the list may not have run through.  Builtins that assign a value
    only the shell knows, and a compound command's body, stay unknown to both."""
    command = "cd /w && T=tmp/x && tail $T/a; cp $T/a ../../$T/; false || cd /v; cp b $T/c"
    conservative = [c.words[-1].text for c in simple_commands(command, "/r")]
    optimistic = [c.words[-1].text for c in simple_commands(command, "/r", optimistic=True)]
    assert conservative == ["/w", "T=tmp/x", "tmp/x/a", "../../$T/", "false", "/v", "$T/c"]
    assert optimistic == ["/w", "T=tmp/x", "tmp/x/a", "../../tmp/x/", "false", "/v", "tmp/x/c"]
    assert [c.cwd for c in simple_commands(command, "/r")][-1] is None
    assert [c.cwd for c in simple_commands(command, "/r", optimistic=True)][-1] == "/v"
    for unknown in ("T=tmp/x; read -r T <<< n; cp a $T/b", "if x; then T=tmp/x; fi; cp a $T/b",
                    "T=tmp/x; for T in n; do :; done; cp a $T/b"):
        assert simple_commands(unknown, "/r", optimistic=True)[-1].words[-1].text == "$T/b"
    # $PWD is the directory the scan knows, in both readings, unless the command set it
    for pwd, last in (("cd /w && S=$PWD/tmp/x && rm -f $S/.cov*", "/w/tmp/x/.cov*"),
                      ("S=$PWD/tmp/x; rm -f $S/.cov*", "/r/tmp/x/.cov*"),
                      ("cd $D && rm -f $PWD/x", "$PWD/x"), ("PWD=/p; rm -f $PWD/x", "/p/x"),
                      ("PWD=$X; rm -f $PWD/x", "$PWD/x")):
        assert simple_commands(pwd, "/r")[-1].words[-1].text == last, pwd
        assert simple_commands(pwd, "/r", optimistic=True)[-1].words[-1].text == last, pwd
    assert simple_commands("rm -f $PWD/x", None)[-1].words[-1].text == "$PWD/x"


@pytest.mark.skipif(not host_sed_is_gnu(), reason="GNU sed is the oracle")
def test_an_unquoted_value_with_a_blank_is_split_as_bash_splits_it(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("one\n")
    command = 'T="tmp/out notes.md"; sed -i "s/one/one\\ntwo/" $T'
    # bash hands sed two operands, the second the tracked file: refused
    assert rewrite_guard(command, str(tmp_path), lambda path: path.endswith("notes.md")) is not None
    assert [w.literal for w in _program("sed", command).argv()][-1] is False
    _created(tmp_path, command)
    assert (tmp_path / "notes.md").read_text() == "one\ntwo\n"
    # double-quoted, the value is the one operand it reads as
    quoted = _program("sed", 'T="tmp/out notes.md"; sed -i s/a/b/ "$T"')
    assert [(w.text, w.literal) for w in quoted.argv()][-1] == ("tmp/out notes.md", True)


def _program(name: str, command: str) -> SimpleCommand:
    """The simple command of *command* running *name*, as the scanner reads it."""
    return next(c for c in simple_commands(command, "/w") if c.program() == name)


def test_guards_follow_an_assigned_variable_to_the_file() -> None:
    def guarded(path: str) -> bool:
        return path == "/w/notes.md"
    verdict = heredoc_guard("F=notes.md; cat > $F " + _BODY, "/w", guarded)
    assert verdict == HEREDOC_VERDICT.format(path="/w/notes.md")
    # a reassignment that ends with its pipeline or its backgrounded list
    assert heredoc_guard("F=notes.md; { F=tmp/n; } | cat; cat > $F " + _BODY, "/w",
                         guarded) == verdict
    assert heredoc_guard("F=notes.md; F=tmp/n && true & wait; cat > $F " + _BODY, "/w",
                         guarded) == verdict
    # ``read`` gives the variable a value only the shell knows: not scratch any more
    read = 'F=tmp/probe; read F <<< notes.md; cat > "$F" ' + _BODY
    assert heredoc_guard(read, "/w", guarded) == HEREDOC_UNRESOLVED_VERDICT.format(path="$F")
    assert heredoc_guard("F=tmp/notes.md; cat >> $F " + _BODY, "/w", guarded) is None
    # Without a predicate, rewrite_guard refuses as before: every multi-line sed -i.
    sed = "F=tmp/notes.md; sed -i 's/a/a\\nb/' $F"
    assert rewrite_guard(sed) is not None
    assert rewrite_guard(sed, "/w", guarded) is None
    assert rewrite_guard("sed -i 's/a/a\\nb/' notes.md", "/w", guarded) is not None
    # One guarded operand among several keeps the refusal.
    assert rewrite_guard("sed -i 's/a/a\\nb/' tmp/notes.md notes.md", "/w", guarded) is not None


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A work dir holding the tracked ``notes.md`` and the scratch ``tmp/notes.md``."""
    (tmp_path / "notes.md").write_text("one\n")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "notes.md").write_text("one\n")
    return tmp_path


@pytest.mark.skipif(not host_sed_is_gnu(), reason="GNU sed is the oracle")
def test_multiline_sed_runs_on_a_scratch_file_only(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    out = tools.Bash("sed -i 's/one/&\\ntwo/' tmp/notes.md", "scratch")
    assert "REFUSED" not in out and (work / "tmp" / "notes.md").read_text() == "one\ntwo\n"
    out = tools.Bash("T=tmp; sed -i 's/two/&\\nthree/' $T/notes.md", "scratch, variable")
    assert "REFUSED" not in out
    assert (work / "tmp" / "notes.md").read_text() == "one\ntwo\nthree\n"
    out = tools.Bash("sed -i 's/one/&\\ntwo/' notes.md", "tracked")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run") and (
        (work / "notes.md").read_text() == "one\n")
    # The file need not have been shown: a one-substitution sed on a scratch
    # file is applied as the Edit it stands for, unread.
    out = tools.Bash("sed -i 's/three/3/' tmp/notes.md", "mechanical")
    assert out.startswith("Applied as Edit(") and (
        (work / "tmp" / "notes.md").read_text() == "one\ntwo\n3\n")
    out = tools.Bash("sed -i 's/one/1/' notes.md", "mechanical, tracked")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run") and "not shown" in out
    assert (work / "notes.md").read_text() == "one\n"


_REWRITE = (
    "python3 - <<'EOF'\nimport re\np = {path!r}\ns = open(p).read()\n"
    "s = re.sub(r'one', 'uno', s)\nopen(p, 'w').write(s)\nEOF"
)


def test_python_heredoc_rewrites_a_scratch_file_only(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    out = tools.Bash(_REWRITE.format(path="tmp/notes.md"), "scratch")
    assert "REFUSED" not in out and (work / "tmp" / "notes.md").read_text() == "uno\n"
    out = tools.Bash(_REWRITE.format(path="notes.md"), "tracked")
    assert out.startswith("REFUSED: the python heredoc did NOT run") and (
        (work / "notes.md").read_text() == "one\n")
    # A path the program builds at run time may name any file: refused as before.
    built = _REWRITE.replace("p = {path!r}", "d = 'tmp'\np = f'{d}/notes.md'")
    out = tools.Bash(built, "f-string path")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    assert (work / "tmp" / "notes.md").read_text() == "uno\n"
    # The write need not be the pattern python_edit converts: a tracked file
    # rewritten with the .replace inline in the write stays refused...
    inline = ("python3 - <<'EOF'\np='notes.md'\ns=open(p).read()\n"
              "open(p,'w').write(s.replace('one','x'))\nEOF")
    out = tools.Bash(inline, "inline")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    # ...and a program writing a scratch file and a tracked one too.
    both = _REWRITE.format(path="tmp/notes.md").replace(
        "open(p, 'w')", "open('notes.md', 'a')\nopen(p, 'w')")
    out = tools.Bash(both, "two files")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    assert (work / "notes.md").read_text() == "one\n"
    # The here-document need not be the whole command (audit 2026-11-03: a ``cd``
    # into the scratch dir before it, a check after it): the program is read from
    # the python command's own here-document, and its relative path resolves
    # from the directory the ``cd`` left it in.
    (work / "tmp" / "notes.md").write_text("one\n")
    around = "cd tmp && " + _REWRITE.format(path="notes.md") + "\ngrep -c uno notes.md"
    out = tools.Bash(around, "cd before, grep after")
    assert out.strip() == "1" and (work / "tmp" / "notes.md").read_text() == "uno\n"
    assert (work / "notes.md").read_text() == "one\n"
    out = tools.Bash("cd tmp && " + _REWRITE.format(path="../notes.md"), "cd, then tracked")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    assert (work / "notes.md").read_text() == "one\n"
    # an expanding here-document (unquoted delimiter) with a ``$`` is unknown text
    expanding = _REWRITE.format(path="tmp/notes.md").replace("<<'EOF'", "<<EOF").replace(
        "'uno'", "'$uno'") + "\necho done"
    out = tools.Bash(expanding, "expanding")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    # every python of the command is read (review 2026-11-03): a scratch rewrite
    # first does not let a tracked rewrite after it run
    (work / "tmp" / "notes.md").write_text("one\n")
    two = _REWRITE.format(path="tmp/notes.md") + "\n" + _REWRITE.format(path="notes.md")
    out = tools.Bash(two, "scratch, then tracked")
    assert out.startswith("REFUSED: the python heredoc did NOT run")
    assert (work / "notes.md").read_text() == "one\n"
    assert (work / "tmp" / "notes.md").read_text() == "one\n"
    two = _REWRITE.format(path="tmp/notes.md") + "\n" + _REWRITE.format(path="tmp/notes.md")
    assert "REFUSED" not in tools.Bash(two, "scratch twice")
    assert (work / "tmp" / "notes.md").read_text() == "uno\n"


@pytest.mark.parametrize("program, expected", [
    ("p='a.py'\ns=open(p).read()\nopen(p,'w').write(s.replace('x','y'))", ["a.py"]),
    ("from pathlib import Path\np = Path('tmp/x.py')\np.write_text(p.read_text())", ["tmp/x.py"]),
    ("import pathlib\np = pathlib.Path('tmp/x.py')\np.write_text(p.read_text())", ["tmp/x.py"]),
    ("import os\np = os.path.join('tmp', 'x.py')\nopen(p, 'w').write('x')", None),
    ("with open('tmp/a', 'w') as f:\n    f.write('x')\nopen('tmp/b', mode='ab').write(b'')",
     ["tmp/a", "tmp/b"]),
    ("from pathlib import Path\nwith Path('tmp/c').open('r+') as f:\n    f.write('x')", ["tmp/c"]),
    ("print(open('a.py').read())", None),  # nothing written
    ("import sys\nopen(sys.argv[1], 'w').write('x')", None),
    ("T='x'\nopen(f'{T}/f', 'w').write('x')", None),
    ("p = 'a'\np = 'b'\nopen(p, 'w').write('x')", None),  # assigned twice
    ("m = 'w'\nopen('a', m).write('x')", None),  # a mode only the program knows
    ("open('a', 'w'", None),  # does not parse
    # a parameter shadowing the module-level name: the write is to the argument
    ("p = 'tmp/out'\ndef rewrite(p):\n    open(p, 'w').write('x')\nrewrite('notes.md')", None),
    ("p = 'tmp/out'\nfor p in ['notes.md']:\n    open(p, 'w').write('x')", None),
    ("p = 'tmp/out'\nimport p\nopen(p, 'w').write('x')", None),
    # the keyword forms of open()
    ("open('tmp/a', 'w').write('x')\nopen(file='notes.md', mode='w').write('y')",
     ["tmp/a", "notes.md"]),
    ("kw = {'mode': 'w'}\nopen('notes.md', **kw)", None),
    ("open('tmp/a', 'w').write('x')\nkw = {'file': 'notes.md', 'mode': 'w'}\nopen(**kw)", None),
    # a match pattern's captures bind the name too
    ("p = 'tmp/out'\nmatch 'notes.md':\n    case p:\n        pass\nopen(p, 'w').write('x')", None),
    ("p = 'tmp/out'\nmatch ['notes.md']:\n    case [*p]:\n        pass\nopen(p, 'w')", None),
    ("p = 'tmp/out'\nmatch {}:\n    case {**p}:\n        pass\nopen(p, 'w')", None),
])
def test_python_written_paths(program: str, expected: list[str] | None) -> None:
    assert python_written_paths(program) == expected


def test_heredoc_copies_through_a_variable_reach_scratch_files(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    out = tools.Bash("T=tmp/probe; mkdir -p $T; cat > $T/s.py <<'EOF'\nprint(1)\nEOF", "create")
    assert "REFUSED" not in out and (work / "tmp" / "probe" / "s.py").read_text() == "print(1)\n"
    out = tools.Bash("P=tmp/notes.md; cat >> $P <<'EOF'\ntwo\nEOF", "append")
    assert "REFUSED" not in out and (work / "tmp" / "notes.md").read_text() == "one\ntwo\n"
    out = tools.Bash("P=notes.md; cat >> $P <<'EOF'\ntwo\nEOF", "tracked")
    assert out.startswith(HEREDOC_VERDICT.format(path=work / "notes.md"))
    assert (work / "notes.md").read_text() == "one\n"
