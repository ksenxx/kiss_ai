# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The shell-rewrite conversions are exact or do not happen; compound commands (2026-10-31).

Real ``sed``, ``python3`` and ``bash`` are the oracles: each test runs
the command as written in one directory and compares it with what the
conversion (:func:`heredoc_copy` / :func:`mechanical_edit`, applied
through a real :class:`UsefulTools`) does in another.

* ``sed`` without ``-z`` works line by line: a pattern holding a newline
  is not converted (it never matches); a substitution without ``g``
  replaces the first match of every line, which
  :meth:`MechanicalEdit.edits_for` turns into the exact ``Edit`` for the
  file's text, or into none.
* A ``sed`` script the shell expands (``$A``, a backtick, a glob, ANSI-C
  quoting) is not converted; quoting the shell removes is read as bash
  reads it.
* A python ``.replace()`` after the write, or an I/O keyword that changes
  the bytes (``encoding``, ``errors``, ``newline``), is not converted.
* A here-document followed by another input redirection on its command
  is not converted (bash honours the last one).
* ``[cd DIR &&] <rewrite> [&& | ; | newline] <rest>`` comes back with the
  ``cd`` directory and the rest, and applying the call then running the
  rest leaves the same files as bash; every other shape is not converted.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_VERDICT,
    heredoc_copy,
    heredoc_text,
    mechanical_edit,
    python_edit,
    rewrite_guard,
    sed_edit,
)
from kiss.agents.sorcar.shell_scan import simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.conftest import host_sed_is_gnu, posix_only

pytestmark = posix_only("sed, python3 and bash are the oracles")
needs_gnu_sed = pytest.mark.skipif(not host_sed_is_gnu(), reason="needs GNU sed as the oracle")


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UsefulTools:
    """A tool set on ``tmp_path`` with no detach grace."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    return UsefulTools(work_dir=str(tmp_path))


def _bash(command: str, cwd: Path, **env: str) -> None:
    """Run *command* with bash in *cwd*."""
    subprocess.run(["bash", "-c", command], cwd=cwd, check=True, env={**os.environ, **env})


def _sed(work: Path, content: str, *args: str) -> str:
    """Return the text of ``f.txt`` holding *content* after ``sed -i *args f.txt``."""
    target = work / "f.txt"
    target.write_text(content)
    subprocess.run(["sed", "-i", *args, "f.txt"], cwd=work, check=True)
    return target.read_text()


def _edited(tools: UsefulTools, work: Path, content: str, command: str) -> str | None:
    """Return the text of ``f.txt`` holding *content* after the ``Edit`` *command* stands for.

    The pairs are the ones :meth:`MechanicalEdit.edits_for` settles
    for *content*; ``None`` when there are none, or the ``Edit``
    refuses (the file is then checked to be untouched).
    """
    target = work / "f.txt"
    target.write_text(content)
    edit = mechanical_edit(command, str(work))
    assert edit is not None and edit.path == str(target), command
    pairs = edit.edits_for(content)
    if pairs is None:
        return None
    tools.Read(str(target))
    result = tools.Edit(str(target), edits=json.dumps(
        [{"old_string": old, "new_string": new, "replace_all": every} for old, new, every in pairs]
    ))
    if result.startswith("Error"):
        assert target.read_text() == content
        return None
    return target.read_text()


# --- 1. sed record semantics --------------------------------------------------


@needs_gnu_sed
def test_sed_without_z_converts_only_what_sed_does(tools: UsefulTools, tmp_path: Path) -> None:
    # A pattern holding a newline never matches without -z: sed leaves the file; no conversion.
    assert _sed(tmp_path, "a\nb\n", "s/a\\nb/c/") == "a\nb\n"
    command = "sed -i 's/a\\nb/c/' f.txt"
    assert mechanical_edit(command, str(tmp_path)) is None
    guard = rewrite_guard(command)
    assert guard is not None and 'Edit(file_path="f.txt", old_string=<the exact' in guard
    # With -z the file is one record: the pair is exact.
    assert _sed(tmp_path, "a\nb\n", "-z", "s/a\\nb/c/") == "c\n"
    assert _edited(tools, tmp_path, "a\nb\n", "sed -z -i 's/a\\nb/c/' f.txt") == "c\n"
    # Without g, sed replaces the first match of every line; the Edit follows the text.
    per_line = "sed -i 's/a/b\\n/' f.txt"
    edit = mechanical_edit(per_line, str(tmp_path))
    assert edit is not None and edit.per_line and edit.edits == [("a", "b\n", False)]
    for content in ("a\na\n", "x\na\n", "x\n", "a\naa\n"):
        expected = _sed(tmp_path, content, "s/a/b\\n/")
        converted = _edited(tools, tmp_path, content, per_line)
        assert converted in (expected, None), content
    assert edit.edits_for("a\na\n") == [("a", "b\n", True)]  # two lines, one match each
    assert edit.edits_for("x\na\n") == [("a", "b\n", False)]  # one match
    assert edit.edits_for("x\n") == [("a", "b\n", False)]  # none: the Edit reports it
    assert edit.edits_for("a\naa\n") is None  # a line with two matches: no Edit is exact
    assert _edited(tools, tmp_path, "a\na\n", per_line) == "b\n\nb\n\n"
    assert _edited(tools, tmp_path, "x\n", per_line) is None
    # With g every match goes, as the Edit's replace_all does; -z without g is the first match.
    assert _sed(tmp_path, "a\naa\n", "s/a/b\\n/g") == "b\n\nb\nb\n\n"
    assert _edited(tools, tmp_path, "a\naa\n", "sed -i 's/a/b\\n/g' f.txt") == "b\n\nb\nb\n\n"
    assert _sed(tmp_path, "a\na\n", "-z", "s/a/b/") == "b\na\n"
    assert _edited(tools, tmp_path, "a\na\n", "sed -z -i 's/a/b/' f.txt") is None
    # -z makes NUL the record separator, not the file one record (A08 fable-10): without g
    # the first match of every NUL-separated record goes, as of every line without -z.
    null_data = mechanical_edit("sed -z -i 's/a/b/' f.txt", str(tmp_path))
    assert null_data is not None and null_data.per_line and null_data.separator == "\0"
    assert null_data.edits_for("a\na\n") is None  # one record holding the pattern twice
    assert null_data.edits_for("a\0a") == [("a", "b", True)]  # two records, one match each
    assert _sed(tmp_path, "a\0a", "-z", "s/a/b/") == "b\0b"
    assert _edited(tools, tmp_path, "a\0a", "sed -z -i 's/a/b/' f.txt") == "b\0b"
    assert _sed(tmp_path, "aa\0a", "-z", "s/a/b/") == "ba\0b"
    assert _edited(tools, tmp_path, "aa\0a", "sed -z -i 's/a/b/' f.txt") is None
    # Through the Bash tool: the pairs edits_for settles from the file's text, so two
    # lines with one match each are replaced as sed replaces them; a line holding the
    # pattern twice has no exact Edit and the multi-line sed is refused as before.
    target = tmp_path / "f.txt"
    target.write_text("a\n")
    tools.Read(str(target))
    out = tools.Bash(per_line, "one match")
    assert out.startswith("Applied as Edit(") and target.read_text() == "b\n\n"
    target.write_text("a\na\n")
    out = tools.Bash(per_line, "two matches")
    assert out.startswith("Applied as Edit(") and "replace_all=True" in out, out
    assert target.read_text() == "b\n\nb\n\n"
    target.write_text("a\naa\n")
    out = tools.Bash(per_line, "two matches on one line")
    assert out == rewrite_guard(per_line), out
    assert target.read_text() == "a\naa\n"


# --- 2. shell expansion in the sed script ------------------------------------


@needs_gnu_sed
def test_sed_script_the_shell_expands_is_not_converted(tools: UsefulTools, tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    for command in (
        'sed -i "s/a/$A\\n/" f.txt',
        'sed -i -e "s/a/$A\\n/" f.txt',
        'sed -i --expression="s/a/$A\\n/" f.txt',
        'sed -i -e"s/a/$A\\n/" f.txt',
        'sed -i "s/a/`echo x`\\n/" f.txt',
        "sed -i s/a/b*\\\\n/ f.txt",  # an unquoted glob
        "sed -i $'s/a/b\\\\n/' f.txt",  # ANSI-C quoting
    ):
        target.write_text("a\n")
        _bash(command, tmp_path, A="x")
        assert target.read_text() != "a\n", command  # bash and sed did rewrite the file
        assert mechanical_edit(command, str(tmp_path)) is None, command
        operand, edits = sed_edit(simple_commands(command, None)[0])
        assert operand is not None and operand.text == "f.txt" and edits == [], command
    # Quoting the shell removes is read as bash reads it: "\\n" is sed's \n, unquoted \n is n.
    for command, expected in (
        ('sed -i "s/a/b\\\\n/" f.txt', "b\n\n"),
        ("sed -i s/a/b\\\\n/ f.txt", "b\n\n"),
        ("sed -i s/a/b\\n/ f.txt", "bn\n"),
    ):
        target.write_text("a\n")
        _bash(command, tmp_path)
        assert target.read_text() == expected, command
        assert _edited(tools, tmp_path, "a\n", command) == expected, command


@needs_gnu_sed
def test_sed_escaped_metacharacter_delimiter_is_not_converted(tmp_path: Path) -> None:
    """With a regex metacharacter as its delimiter, sed reads the escaped delimiter of the
    pattern as the metacharacter (``s.a\\.b.Z.`` replaces ``aXb``), not as the text the
    conversion took it for: the ``Edit`` changed other text than sed does."""
    for script, content, expected in (
        ("s.a\\.b.Z.g", "aXb a.b\n", "Z Z\n"),  # any character
        ("s*a\\*b*Z*g", "aab a*b ab\n", "Z a*Z Z\n"),  # a repetition
        ("s^\\^a^Z^g", "^a a\n", "^a a\n"),  # an anchor: the text "^a" stays
        ("s$a\\$$Z$g", "a$ a\n", "a$ Z\n"),
    ):
        assert _sed(tmp_path, content, script) == expected, script
        assert mechanical_edit(f"sed -i '{script}' f.txt", str(tmp_path)) is None, script
    # a delimiter that is no metacharacter, escaped, is the text; so is one in the replacement
    assert _sed(tmp_path, "a|b a.b\n", "s|a\\|b|Z\\||g") == "Z| a.b\n"
    edit = mechanical_edit("sed -i 's|a\\|b|Z\\||g' f.txt", str(tmp_path))
    assert edit is not None and edit.edits == [("a|b", "Z|", True)]
    assert _sed(tmp_path, "a,b aXb\n", "s.a,b.Z\\..g") == "Z. aXb\n"
    edit = mechanical_edit("sed -i 's.a,b.Z\\..g' f.txt", str(tmp_path))
    assert edit is not None and edit.edits == [("a,b", "Z.", True)]


@needs_gnu_sed
def test_sed_command_sed_refuses_is_not_converted(tools: UsefulTools, tmp_path: Path) -> None:
    """A command sed exits 1 on leaves the file: no ``Edit`` stands for it.

    A last ``-e`` / ``--expression`` has no script (it was ignored and the
    script before it applied: review of 2026-11-11, S8 finding 1); an ``s``
    command ends at a newline no backslash precedes; its delimiter is one byte.
    """
    target = tmp_path / "f.txt"
    target.write_text("a\n")
    tools.Read(str(target))  # shown: a command read as an Edit would be applied as it
    for command in (
        "sed -i 's/a/b/' f.txt -e",
        "sed -i 's/a/b/' f.txt --expression",
        "sed -i -e 's/a/b/' f.txt -e",
        "sed -i --expression='s/a/b/' f.txt --expression",
        "sed -i 's/a/b\nc/' f.txt",
        "sed -i -z 's/a\n/b/' f.txt",
        "sed -i 's§a§b§' f.txt",
        "sed -i -z 's→a→b→g' f.txt",
    ):
        run = subprocess.run(["bash", "-c", command], cwd=tmp_path, check=False)
        assert run.returncode == 1 and target.read_text() == "a\n", command
        assert mechanical_edit(command, str(tmp_path)) is None, command
        out = tools.Bash(command, "refused by sed")
        assert out.startswith("Error (exit code 1)") and target.read_text() == "a\n", out
    # The option with its script, in the same place, is the conversion.
    for option in ("-e", "--expression"):
        command = f"sed -i f.txt {option} 's/a/b/'"
        assert _edited(tools, tmp_path, "a\n", command) == "b\n", command
        target.write_text("a\n")
        _bash(command, tmp_path)
        assert target.read_text() == "b\n", command


# --- 3. python replacements after the write ----------------------------------

_READ = "s = open('a.py').read()\n"
_WRITE = "open('a.py', 'w').write(s)\n"


def _python(work: Path, program: str) -> str:
    """Return the text of ``a.py`` (bytes decoded, no newline translation) after bash runs
    *program* as a ``python3 -`` here-document."""
    _bash(f"python3 - <<'EOF'\n{program}EOF", work)
    return (work / "a.py").read_bytes().decode()


def test_python_replacement_after_the_write_is_not_converted(tmp_path: Path) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 1\nq = 2\n")
    program = _READ + "s = s.replace('x', 'y')\n" + _WRITE + "s = s.replace('q', 'r')\n"
    assert _python(tmp_path, program) == "y = 1\nq = 2\n"  # the second never reaches the file
    assert python_edit(program) == ("a.py", [], False)
    assert mechanical_edit(f"python3 - <<'EOF'\n{program}EOF", str(tmp_path)) is None
    source.write_text("x = 1\nq = 2\n")
    program = _READ + _WRITE + "s = s.replace('x', 'y')\n"
    assert _python(tmp_path, program) == "x = 1\nq = 2\n"
    assert python_edit(program) == ("a.py", [], False)
    # The same replacements before the write are the pattern.
    program = _READ + "s = s.replace('x', 'y')\ns = s.replace('q', 'r')\n" + _WRITE
    assert python_edit(program) == ("a.py", [("x", "y", True), ("q", "r", True)], True)
    assert _python(tmp_path, program) == "y = 1\nr = 2\n"


def test_python_no_op_and_empty_replacements_make_no_forbidden_edit(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """A ``replace(x, x)`` yields no pair, ``replace('', x)`` no conversion (astra-13 / fable-11).

    ``Edit`` refuses equal strings and an empty needle, so such a pair
    would fail the batch before the real replacement that follows it.
    """
    source = tmp_path / "a.py"
    source.write_text("a b\n")
    program = _READ + "s = s.replace('a', 'a').replace('b', 'c')\n" + _WRITE
    assert python_edit(program) == ("a.py", [("b", "c", True)], True)
    assert _python(tmp_path, program) == "a c\n"
    heredoc = "python3 - <<'EOF'\n{}EOF"
    assert _edited(tools, tmp_path, "a b\n", heredoc.format(program.replace("a.py", "f.txt"))) == (
        "a c\n"
    )
    # A chain of no-ops alone changes nothing: no Edit at all, not one with equal strings.
    program = _READ + "s = s.replace('a', 'a')\n" + _WRITE
    assert python_edit(program) == ("a.py", [], False)
    assert mechanical_edit(heredoc.format(program), str(tmp_path)) is None
    # An empty needle inserts between every character, which no Edit does.
    program = _READ + "s = s.replace('', 'X').replace('a', 'b')\n" + _WRITE
    source.write_text("a\n")
    assert _python(tmp_path, program) == "XbX\nX"
    assert python_edit(program) == ("a.py", [], False)
    assert mechanical_edit(heredoc.format(program), str(tmp_path)) is None


def test_python_import_rebinding_a_name_is_not_converted(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An import binding the path or text name, or ``open``, makes python fail
    # before or at the write (audit 2026-11-04 S5 F11): the file never holds the
    # Edit's text, so no conversion; the path is still named.  So does a string
    # read as a ``Path``, an assignment or a ``with`` binding ``open`` or ``Path``,
    # and a ``Path`` that was never imported (audit S4-F3).
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    tools.Read(str(source))  # shown: a program read as the pattern would be applied as its Edit
    for program in (
        "p = 'a.py'\nimport os as p\ns = open(p).read()\ns = s.replace('x', 'y')\n"
        "open(p, 'w').write(s)\n",
        _READ + "s = s.replace('x', 'y')\nimport os.path as s\n" + _WRITE,
        "from gzip import open\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE,
        "from os import *\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE,
        "p = 'a.py'\ns = p.read_text()\ns = s.replace('x', 'y')\np.write_text(s)\n",
        "s = 'a.py'.read_text()\ns = s.replace('x', 'y')\n" + _WRITE,
        # python fails at an import of a module or a name that is not there, the file unwritten
        "import no_such_module_b5\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE,
        "import re, no_such_module_b5\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE,
        "from re import no_such_name_b5\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE,
        "p = 'a.py'\nopen = 'a.py'\ns = open(p).read()\ns = s.replace('x', 'y')\n"
        "open(p, 'w').write(s)\n",
        "from pathlib import Path\nPath = 'a.py'\np = Path('a.py')\ns = p.read_text()\n"
        "s = s.replace('x', 'y')\np.write_text(s)\n",
        "with open('a.py') as open:\n    s = open.read()\ns = s.replace('x', 'y')\n" + _WRITE,
        "from pathlib import Path\nwith open('a.py') as f:\n    Path = f.read()\n"
        "s = Path.replace('x', 'y')\nPath('a.py').write_text(s)\n",
        "s = Path('a.py').read_text()\ns = s.replace('x', 'y')\nPath('a.py').write_text(s)\n",
    ):
        source.write_text("x = 1\n")
        out = tools.Bash(f"python3 - <<'EOF'\n{program}EOF", "rebound")
        assert out.startswith("REFUSED: the python heredoc did NOT run") and "a.py" in out, program
        assert source.read_text() == "x = 1\n", program
        run = subprocess.run(["bash", "-c", f"python3 - <<'EOF'\n{program}EOF"], cwd=tmp_path,
                             check=False, capture_output=True)
        assert run.returncode != 0 and source.read_text() != "y = 1\n", program
        assert python_edit(program) == ("a.py", [], False), program
        assert mechanical_edit(f"python3 - <<'EOF'\n{program}EOF", str(tmp_path)) is None, program
    # A module that acts when imported (``this`` prints) is no part of the pattern either.
    program = "import this\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE
    assert python_edit(program) == ("a.py", [], False)
    # An import of a module every python has that does nothing, ``pathlib``'s ``Path``
    # included, is part of the pattern; so is a name assigned a ``Path`` name, a ``Path`` too.
    program = ("import re, os.path\nimport sys as system\nfrom pathlib import Path\n"
               "p = Path('a.py')\nq = p\ns = p.read_text()\n"
               "s = s.replace('x', 'y')\nq.write_text(s)\n")
    source.write_text("x = 1\n")
    assert python_edit(program) == ("a.py", [("x", "y", True)], True)
    assert _python(tmp_path, program) == "y = 1\n"
    # Python read from stdin searches its directory first, then those of PYTHONPATH: a module
    # of that name there is what the import runs (review 2026-11-12 B5), so the program is
    # not applied as an Edit.
    program = "import json\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE
    command = f"python3 - <<'EOF'\n{program}EOF"
    broken = "raise RuntimeError('shadowed import')\n"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("lib", "lib.zip")))
    for module in ("json.py", "json/__init__.py", "json/__init__.so", "lib/json.py", "lib.zip"):
        (tmp_path / module).parent.mkdir(exist_ok=True)
        if module == "lib.zip":  # python imports from an archive on its path
            with zipfile.ZipFile(tmp_path / module, "w") as archive:
                archive.writestr("json.py", broken)
        else:
            (tmp_path / module).write_text(broken)
        source.write_text("x = 1\n")
        out = tools.Bash(command, "shadowed")
        assert out.startswith("REFUSED: the python heredoc did NOT run"), module
        run = subprocess.run(
            ["bash", "-c", command], cwd=tmp_path, check=False, capture_output=True)
        assert run.returncode != 0 and source.read_text() == "x = 1\n", module
        (tmp_path / module).unlink()
    # What only shares the name is no module: python runs the program, and so it is applied.
    for other in ("json.txt", "json.backup.py", "json-notes/__init__.py", "json/notes.py"):
        (tmp_path / other).parent.mkdir(exist_ok=True)
        (tmp_path / other).write_text(broken)
    assert _python(tmp_path, program) == "y = 1\n"
    source.write_text("x = 1\n")
    assert tools.Bash(command, "not shadowed").startswith("Applied as")
    assert source.read_text() == "y = 1\n"
    # The interpreter is asked where it finds the modules (audit 2026-11-12 round 2, cross
    # findings of B7): a reading of the directory and PYTHONPATH missed an archive named as
    # python also takes it, an entry python normalises, a directory python's own startup puts
    # first (a ``sitecustomize``, a ``.pth``) and a module the library's ``json`` itself imports,
    # be it a submodule of the package that a finder of the startup takes (the review's finding).
    with zipfile.ZipFile(tmp_path / "lib.zip", "w") as archive:
        archive.writestr("json.py", broken)
        archive.writestr("pkg/json.py", broken)
    finder = ("import importlib.util, sys\nclass Finder:\n    @staticmethod\n"
              "    def find_spec(name, path, target=None):\n        if name == 'json.decoder':\n"
              "            return importlib.util.spec_from_file_location(name, 'lib/json.py')\n"
              "sys.meta_path.insert(0, Finder)\n")
    startups = {"startup": "import sys\nsys.path.insert(0, 'lib')\n", "finder": finder}
    for directory, module in (("lib", "json.py"), ("deep", "enum.py"),
                              ("startup", "sitecustomize.py"), ("finder", "sitecustomize.py")):
        (tmp_path / directory).mkdir(exist_ok=True)
        (tmp_path / directory / module).write_text(startups.get(directory, broken))
    for pythonpath in ("lib.zip/", "lib.zip/pkg", "missing/../lib", "startup", "deep", "finder"):
        monkeypatch.setenv("PYTHONPATH", pythonpath)
        source.write_text("x = 1\n")
        out = tools.Bash(command, "shadowed")
        assert out.startswith("REFUSED: the python heredoc did NOT run"), pythonpath
        run = subprocess.run(
            ["bash", "-c", command], cwd=tmp_path, check=False, capture_output=True)
        assert run.returncode != 0 and source.read_text() == "x = 1\n", pythonpath


def test_a_rewrite_whose_program_bash_does_not_find_is_not_converted(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bash fails with status 127 and leaves the file alone (cross finding of B7, 2026-11-12,
    # met in a container without the interpreter): no Edit is made in the command's place.
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    assert "x = 1" in tools.Read(str(source))  # shown: only the program is in question
    bash = shutil.which("bash")
    assert bash is not None
    body = " - <<'EOF'\n" + _READ + "s = s.replace('x', 'y')\n" + _WRITE + "EOF"
    monkeypatch.setenv("PATH", str(tmp_path))
    for command in ("python3" + body, "./python3" + body, "sed -i 's/x/y/' a.py"):
        assert not tools.Bash(command, "no such program").startswith("Applied as"), command
        run = subprocess.run([bash, "-c", command], cwd=tmp_path, check=False, capture_output=True)
        assert run.returncode == 127 and source.read_text() == "x = 1\n", command
    # Found on PATH or by its path, the program runs: the rewrite is made as its Edit.
    (tmp_path / "python3").symlink_to(sys.executable)
    for command in ("python3" + body, "./python3" + body):
        source.write_text("x = 1\n")
        assert tools.Bash(command, "found").startswith("Applied as"), command
        assert source.read_text() == "y = 1\n"
    # A relative PATH entry names another directory after the command's ``cd``, where bash
    # looks and this process does not (round 2, cross finding of B7): no Edit either.
    (tmp_path / "bin").mkdir()
    (tmp_path / "python3").rename(tmp_path / "bin" / "python3")
    (tmp_path / "sub").mkdir()
    inner = tmp_path / "sub" / "a.py"
    inner.write_text("x = 2\n")
    assert "x = 2" in tools.Read(str(inner))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "bin")
    command = "cd sub && python3" + body
    assert not tools.Bash(command, "relative PATH").startswith("Applied as")
    run = subprocess.run([bash, "-c", command], cwd=tmp_path, check=False, capture_output=True)
    assert run.returncode == 127 and inner.read_text() == "x = 2\n"


# --- 4. python I/O keywords ---------------------------------------------------


def test_python_io_keywords_convert_only_when_edit_matches_them(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    source = tmp_path / "a.py"
    pair = [("x", "y", True)]
    # The last value says whether the read translates line endings (a newline keyword
    # left out or None): the pairs are then exact only for a file holding no "\r".
    for read, write, universal in (
        ("open('a.py', newline='').read()", "open('a.py', 'w', newline='').write(s)", False),
        ("open('a.py', newline='\\n').read()", "open('a.py', 'w', newline='\\n').write(s)",
         False),
        ("open('a.py', encoding='utf-8').read()", "open('a.py', 'w', encoding='UTF8').write(s)",
         True),
        ("open('a.py', errors='strict').read()", "open('a.py', 'w', encoding=None).write(s)",
         True),
        ("open('a.py', newline=None).read()", "open('a.py', 'w', newline='').write(s)", True),
        ("Path('a.py').read_text(encoding='utf-8')", "Path('a.py').write_text(s)", True),
    ):
        program = f"from pathlib import Path\ns = {read}\ns = s.replace('x', 'y')\n{write}\n"
        assert python_edit(program) == ("a.py", pair, universal), program
        source.write_text("x = 1\n")
        assert _python(tmp_path, program) == "y = 1\n"
        # ...and the Edit writes the same bytes.
        source.write_text("x = 1\n")
        tools.Read(str(source))
        assert tools.Edit(str(source), "x", "y", True).startswith("Successfully")
        assert source.read_bytes() == b"y = 1\n"
    # Keywords that change the bytes read or written: no conversion, the path still named.
    for program, content, expected in (
        (_READ + "s = s.replace('x', 'y')\nopen('a.py', 'w', newline='\\r\\n').write(s)\n",
         b"x\n", b"y\r\n"),
        ("s = open('a.py', errors='ignore').read()\ns = s.replace('x', 'y')\n" + _WRITE,
         b"x\xff\n", b"y\n"),
        ("s = open('a.py', encoding='latin-1').read()\ns = s.replace('\\xe9', 'e')\n"
         "open('a.py', 'w', encoding='latin-1').write(s)\n", b"\xe9\n", b"e\n"),
        ("with open('a.py') as f:\n    s = f.read()\ns = s.replace('x', 'y')\n"
         "with open('a.py', 'w', newline='\\r') as f:\n    f.write(s)\n", b"x\n", b"y\r"),
        ("s = Path('a.py').read_text(errors='replace')\ns = s.replace('x', 'y')\n"
         "Path('a.py').write_text(s)\n", b"x\xff\n", "y\ufffd\n".encode()),
    ):
        source.write_bytes(content)
        assert _python(tmp_path, "from pathlib import Path\n" + program).encode() == expected
        assert python_edit(program) == ("a.py", [], False), program
    for program in (
        _READ.replace("'a.py'", "'a.py', encoding=enc") + "s = s.replace('x', 'y')\n" + _WRITE,
        _READ.replace("'a.py'", "'a.py', encoding='nosuch'") + "s = s.replace('x', 'y')\n" + _WRITE,
        _READ + "s = s.replace('x', 'y')\nopen('a.py', 'w', errors='ignore').write(s)\n",
        _READ + "s = s.replace('x', 'y')\nwith open('a.py', 'a') as f:\n    f.write(s)\n",
        # ``newline`` on Path.read_text (3.13+) / write_text (3.10+): an older python of
        # the environment rejects the program, so it is not read as the Edit.
        "s = Path('a.py').read_text(newline='')\ns = s.replace('x', 'y')\n"
        "Path('a.py').write_text(s)\n",
        "s = Path('a.py').read_text()\ns = s.replace('x', 'y')\n"
        "Path('a.py').write_text(s, newline='')\n",
    ):
        assert python_edit(program) == ("a.py", [], False), program


def test_python_name_python_refuses_is_neither_converted_nor_raised(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """An ``encoding`` or a path holding a null character (a lone surrogate): python raises.

    ``codecs.lookup`` raises ``ValueError`` for such an encoding (``LookupError`` for
    an unknown one), which left ``python_edit`` and the ``Bash`` tool (review of
    2026-11-11, S8 finding 2); a path holding a null character was read as the
    pattern, and its ``Edit`` raised.
    """
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    tools.Read(str(source))  # shown: a program read as the pattern would be applied as its Edit
    replace = "s = s.replace('x', 'y')\n"
    refused = "REFUSED: the python heredoc did NOT run"
    # The path names no file: the heredoc runs as written, to python's own error.
    program = (_READ + replace + _WRITE).replace("a.py", "a.py\\x00")
    cases = [(program, "a.py\x00", "Error (exit code 1)")]
    for encoding in ("utf-8\\x00", "\\x00", "\\ud800"):
        keyword = f", encoding='{encoding}'"
        cases += [
            (_READ.replace("'a.py'", "'a.py'" + keyword) + replace + _WRITE, "a.py", refused),
            (_READ + replace + _WRITE.replace("'w'", "'w'" + keyword), "a.py", refused),
        ]
    for program, path, result in cases:
        command = f"python3 - <<'EOF'\n{program}EOF"
        run = subprocess.run(["bash", "-c", command], cwd=tmp_path, check=False)
        assert run.returncode == 1 and source.read_text() == "x = 1\n", program
        assert python_edit(program) == (path, [], False), program
        assert mechanical_edit(command, str(tmp_path)) is None, program
        out = tools.Bash(command, "refused by python")
        assert out.startswith(result) and source.read_text() == "x = 1\n", out


def test_python_universal_newlines_convert_only_for_a_file_without_cr(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """A read with the default ``newline`` turns ``\\r\\n`` and ``\\r`` into ``\\n``; Edit does not.

    Over a file holding a ``\\r`` the program and the Edit write different
    bytes, so the rewrite is not converted (the python heredoc is refused
    as before, naming the Edit); over a file without one they agree and
    the Edit is applied.  A read with ``newline=''`` keeps the bytes and
    converts over any file.  The oracle is python3 run by bash (review of
    2026-10-31, finding 1).
    """
    source = tmp_path / "a.py"
    universal = _READ + "s = s.replace('old', 'new')\n" + _WRITE
    exact = (
        "s = open('a.py', encoding='utf-8', newline='').read()\ns = s.replace('old', 'new')\n"
        "open('a.py', 'w', encoding='utf-8', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{universal}EOF"
    exact_command = f"python3 - <<'EOF'\n{exact}EOF"
    source.write_bytes(b"old\n")
    tools.Read(str(source))  # shown once: every rewrite below may be applied as its Edit
    for content in (b"old\r\nkept\r\n", b"old\rkept\r", b"old\r\r\nkept\n"):
        source.write_bytes(content)
        translated = _python(tmp_path, universal).encode()
        assert b"\r" not in translated and translated.startswith(b"new\n"), (content, translated)
        edit = mechanical_edit(command, str(tmp_path))
        assert edit is not None and edit.universal_newlines and edit.text_dependent
        assert edit.edits_for(content.decode()) is None
        source.write_bytes(content)
        out = tools.Bash(command, "rewrite")
        assert out == rewrite_guard(command), out
        assert 'Edit(file_path="a.py", old_string="old", new_string="new"' in out
        assert source.read_bytes() == content, "the refused rewrite left the file alone"
        # A read with newline='' keeps the bytes: the Edit writes what the program writes.
        assert _python(tmp_path, exact).encode() == content.replace(b"old", b"new")
        source.write_bytes(content)
        out = tools.Bash(exact_command, "rewrite")
        assert out.startswith("Applied as Edit(") and "Successfully replaced 1" in out, out
        assert source.read_bytes() == content.replace(b"old", b"new")
    # Without a "\r" the universal read is the Edit.
    source.write_bytes(b"old\nkept\n")
    assert _python(tmp_path, universal) == "new\nkept\n"
    source.write_bytes(b"old\nkept\n")
    out = tools.Bash(command, "rewrite")
    assert out.startswith("Applied as Edit(") and source.read_bytes() == b"new\nkept\n", out


def test_a_pair_with_a_bare_newline_is_not_converted_over_a_crlf_file(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """An ``old`` holding a ``\\n`` that a CRLF file lacks: ``Edit`` would retry it with ``\\r\\n``.

    ``s.replace('old\\n', 'new\\n')`` over ``old\\r\\nkept\\r\\n`` matches
    nothing and python leaves the file alone, while the Edit it was
    converted to changed the file through that retry (review of
    2026-10-31, finding 1, second round).  The pair is exact for a file
    holding it, or holding no ``\\r\\n``.
    """
    source = tmp_path / "a.py"
    program = (
        "s = open('a.py', newline='').read()\ns = s.replace('old\\n', 'new\\n')\n"
        "open('a.py', 'w', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{program}EOF"
    edit = mechanical_edit(command, str(tmp_path))
    assert edit is not None and edit.text_dependent and not edit.universal_newlines
    assert edit.edits == [("old\n", "new\n", True)]
    source.write_bytes(b"old\n")
    tools.Read(str(source))
    for content, expected in (
        (b"old\r\nkept\r\n", None),  # the retry would match: no Edit is exact
        (b"old\r\nold\nkept\r\n", [("old\n", "new\n", True)]),  # found as written
        (b"old\nkept\n", [("old\n", "new\n", True)]),
    ):
        assert edit.edits_for(content.decode()) == expected, content
        source.write_bytes(content)
        written = _python(tmp_path, program).encode()
        assert written == content.replace(b"old\n", b"new\n"), content
        source.write_bytes(content)
        out = tools.Bash(command, "rewrite")
        if expected is None:
            assert out == rewrite_guard(command) and source.read_bytes() == content, out
        else:
            assert out.startswith("Applied as Edit(") and source.read_bytes() == written, out
    # Pairs are judged in order, each against the text the earlier ones left, as Edit
    # applies them: the first pair below eats the only bare "a\n", so the second matches
    # nothing, as in python, and is left out (handed on, Edit would retry it with "\r\n").
    chain = (
        "s = open('a.py', newline='').read()\ns = s.replace('a\\n', 'a\\r\\n')\n"
        "s = s.replace('a\\nb', 'X')\nopen('a.py', 'w', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{chain}EOF"
    edit = mechanical_edit(command, str(tmp_path))
    assert edit is not None and edit.edits_for("a\nb\r\n") == [("a\n", "a\r\n", True)]
    assert edit.edits_for("a\nb\n") == [("a\n", "a\r\n", True)]  # the same without CRLF
    source.write_bytes(b"a\nb\r\n")
    assert _python(tmp_path, chain).encode() == b"a\r\nb\r\n"
    source.write_bytes(b"a\nb\r\n")
    out = tools.Bash(command, "rewrite")
    assert out.startswith("Applied as Edit(") and source.read_bytes() == b"a\r\nb\r\n", out


def test_a_pair_matching_nothing_is_left_out_as_str_replace_leaves_the_text(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """A chain whose first pair matches nothing: python skips it, and so must the ``Edit``.

    ``s.replace('absent', 'unused').replace('bar', 'BAR')`` over ``bar``
    wrote ``BAR``, while the Edit it was converted to kept the unmatched
    pair and refused the whole chain as a string not found (audit
    2026-11-05, A06-7).  A pair matching nothing in the text the earlier
    pairs leave is left out; when none matches, the pairs stand and the
    Edit reports the first.
    """
    source = tmp_path / "a.py"
    program = (
        "s = open('a.py', newline='').read()\n"
        "s = s.replace('absent', 'unused').replace('bar', 'BAR')\n"
        "open('a.py', 'w', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{program}EOF"
    edit = mechanical_edit(command, str(tmp_path))
    assert edit is not None and edit.text_dependent  # a chain: read the file to settle it
    assert edit.edits_for("bar\n") == [("bar", "BAR", True)]
    assert edit.edits_for("baz\n") == edit.edits  # nothing matches: the Edit reports it
    source.write_text("bar\n")
    assert _python(tmp_path, program) == "BAR\n"
    source.write_text("bar\n")
    tools.Read(str(source))
    out = tools.Bash(command, "rewrite")
    assert out.startswith("Applied as Edit(") and source.read_text() == "BAR\n", out


@needs_gnu_sed
def test_a_pair_the_file_holds_only_whitespace_aside_is_not_converted(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """A pattern the file holds with other blanks: ``Edit`` would retry it ignoring whitespace.

    ``sed -i 's/a b/X/' f`` over ``a   b`` matches nothing and leaves the
    file alone, while the Edit it was converted to changed the file
    through its whitespace-tolerant retry (audit 2026-11-08, S3/F4).
    No Edit is exact then and the sed runs as written; a pattern the
    file lacks whitespace aside too stays converted, so the Edit
    reports it as not found, as does one replacing every match.
    """
    command = "sed -i 's/a b/X/' f.txt"
    target = tmp_path / "f.txt"
    for content in ("a   b\n", "x\na\tb\ny\n", "a  b\na   b\n"):  # one, one, two near matches
        assert _sed(tmp_path, content, "s/a b/X/") == content
        assert _edited(tools, tmp_path, content, command) is None, content
        tools.Read(str(target))
        out = tools.Bash(command, "rewrite")
        assert not out.startswith(("Applied as", "REFUSED")) and target.read_text() == content, out
    assert _edited(tools, tmp_path, "a\nb\n", command) is None  # not found: the Edit says so
    edit = mechanical_edit(command, str(tmp_path))
    assert edit is not None and edit.edits_for("a\nb\n") == [("a b", "X", False)]
    assert edit.edits_for("a   b\n") is None
    # every match (``g``, or python's ``str.replace``): Edit retries nothing, so the
    # pairs stand and the Edit reports the first as not found
    edit = mechanical_edit("sed -i 's/a b/X/g' f.txt", str(tmp_path))
    assert edit is not None and edit.edits_for("a   b\n") == [("a b", "X", True)]
    assert _edited(tools, tmp_path, "a   b\n", "sed -i 's/a b/X/g' f.txt") is None


@needs_gnu_sed
def test_a_redirected_mechanical_edit_is_applied_verbatim_only(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """The ``Edit`` a rewrite is redirected to matches as sed and ``str.replace`` do: verbatim.

    The pairs are settled from the file's text when the command is
    converted and the agent makes the ``Edit`` later: a writer changing
    the file in between turned a ``sed -i 's/a b/X/'`` matching nothing
    into an Edit landing through the whitespace-tolerant retry, and a
    python ``.replace`` over a file turned CRLF into one landing through
    the CRLF retry (audit 2026-11-08, S3/F4).  The redirect carries
    ``exact``, which the Edit honours for one pair and for a batch,
    while the model's own Edit of the same strings keeps both retries.
    """
    target, oracle = tmp_path / "f.txt", tmp_path / "oracle"
    oracle.mkdir()
    tools.redirect = True
    target.write_text("a b\n")
    tools.Read(str(target))
    with pytest.raises(ToolRedirect) as info:
        tools.Bash("sed -i 's/a b/X/' f.txt", "rewrite")
    assert (info.value.name, info.value.arguments) == ("Edit", {
        "file_path": str(target), "old_string": "a b", "new_string": "X",
        "replace_all": False, "exact": True,
    })
    # Another writer's text lands before the agent makes the call: sed matches nothing now.
    target.write_text("a   b\n")
    assert _sed(oracle, "a   b\n", "s/a b/X/") == "a   b\n"
    result = tools.Edit(**info.value.arguments)
    assert result.startswith("Error: String not found in file."), result
    assert target.read_text() == "a   b\n"
    assert tools.Edit(str(target), "a b", "X").startswith("Successfully")
    assert target.read_text() == "X\n"
    # A batch: two ``.replace`` calls over a file that turns CRLF before the call is made.
    program = (
        "s = open('f.txt', newline='').read()\n"
        "s = s.replace('a\\nb', 'A\\nB').replace('c\\nd', 'C\\nD')\n"
        "open('f.txt', 'w', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{program}EOF"
    target.write_text("a\nb\nc\nd\n")
    with pytest.raises(ToolRedirect) as info:
        tools.Bash(command, "rewrite")
    edits = json.dumps([
        {"old_string": "a\nb", "new_string": "A\nB", "replace_all": True},
        {"old_string": "c\nd", "new_string": "C\nD", "replace_all": True},
    ])
    assert info.value.arguments == {"file_path": str(target), "edits": edits, "exact": True}
    crlf = b"a\r\nb\r\nc\r\nd\r\n"
    target.write_bytes(crlf)
    (oracle / "f.txt").write_bytes(crlf)
    _bash(command, oracle)
    assert (oracle / "f.txt").read_bytes() == crlf
    result = tools.Edit(**info.value.arguments)
    assert result.startswith("Error: edit 1: String not found in file."), result
    assert target.read_bytes() == crlf
    assert tools.Edit(str(target), edits=edits).startswith("Successfully")
    assert target.read_bytes() == b"A\r\nB\r\nC\r\nD\r\n"


def test_a_tilde_target_is_resolved_in_the_home_given(tmp_path: Path) -> None:
    """``cat > ~/x`` names a file in the home of the shell that runs it, not the daemon's.

    The copy's path once came from the daemon's ``HOME`` whatever shell
    ran the command (audit 2026-11-05, A06-9): a container's or a remote
    shell's ``~`` is elsewhere, so the caller names it (``home``), or
    passes ``None`` when it is unknown, which leaves the path to the shell.
    """
    command = "cat > ~/report.txt <<'EOF'\nhello\nEOF"
    home = tmp_path / "home"
    home.mkdir()
    _bash(command, tmp_path, HOME=str(home))
    assert (home / "report.txt").read_text() == "hello\n"
    copy = heredoc_copy(command, str(tmp_path), home=str(home))
    assert copy is not None and copy.path == str(home / "report.txt")
    copy = heredoc_copy(command, str(tmp_path), home=None)
    assert copy is not None and copy.path is None and copy.target == "~/report.txt"
    copy = heredoc_copy(command, str(tmp_path))  # the daemon's own shell by default
    assert copy is not None and copy.path == os.path.expanduser("~/report.txt")


@needs_gnu_sed
def test_a_target_that_can_only_name_a_directory_is_not_converted(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    """``f.txt/`` and ``f.txt/.`` name no file: sed, python and bash fail and leave ``f.txt``.

    The path the conversions resolve drops such an ending, so the ``Edit`` or
    ``Write`` landed in the file the command could not open.
    """
    target = tmp_path / "f.txt"
    target.write_text("a\n")
    tools.Read(str(target))  # shown: a rewrite read as a call would be applied as it
    program = "s = open({0!r}).read()\ns = s.replace('a', 'b')\nopen({0!r}, 'w').write(s)\n"
    for name in ("f.txt/", "f.txt/.", "f.txt//"):
        for command in (
            f"sed -i 's/a/b/' {name}",
            f"python3 - <<'EOF'\n{program.format(name)}EOF",
            f"cat > {name} <<'EOF'\nb\nEOF",
            f"cat >> {name} <<'EOF'\nb\nEOF",
            f"tee {name} <<'EOF'\nb\nEOF",
        ):
            run = subprocess.run(["bash", "-c", command], cwd=tmp_path, check=False)
            assert run.returncode != 0 and target.read_text() == "a\n", command
            assert mechanical_edit(command, str(tmp_path)) is None, command
            assert heredoc_copy(command, str(tmp_path)) is None, command
            out = tools.Bash(command, "no file")
            assert not out.startswith(("Applied", "Appended")), out
            assert target.read_text() == "a\n", command


# --- 5. a here-document followed by another input redirection -----------------


def test_heredoc_with_a_later_input_redirection_is_not_converted(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    target = tmp_path / "f.txt"
    for command, written in (
        ("cat > f.txt <<'EOF' < /dev/null\nnew\nEOF", ""),
        ("cat <<'EOF' < /dev/null > f.txt\nnew\nEOF", ""),
        ("cat <<'EOF' > f.txt 0</dev/null\nnew\nEOF", ""),
        ("cat <<'EOF' > f.txt <> /dev/null\nnew\nEOF", ""),
        ("cat <<'EOF' > f.txt <<< x\nnew\nEOF", "x\n"),
        ("tee f.txt <<'EOF' < /dev/null\nnew\nEOF", ""),
        ("cat <<'EOF' > f.txt <<'EOF2'\nnew\nEOF\nnewer\nEOF2", "newer\n"),
    ):
        target.write_text("old\n")
        _bash(command, tmp_path)
        assert target.read_text() == written, command
        assert heredoc_copy(command, str(tmp_path)) is None, command
        assert heredoc_text(command) is None, command
    # The here-document after an input redirection is the one that counts.
    command = "cat < /dev/null <<'EOF' > f.txt\nnew\nEOF"
    target.write_text("old\n")
    _bash(command, tmp_path)
    copy = heredoc_copy(command, str(tmp_path))
    assert copy is not None and copy.body == target.read_text() == "new\n"
    # A python program fed an empty standard input does nothing: no Edit either.
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    program = f"python3 - <<'EOF' < /dev/null\n{_READ}s = s.replace('x', 'y')\n{_WRITE}EOF"
    _bash(program, tmp_path)
    assert source.read_text() == "x = 1\n" and mechanical_edit(program, str(tmp_path)) is None
    assert heredoc_text("# cat <<EOF\nx\nEOF") is None  # a comment opens nothing
    # Through the Bash tool, the shown file keeps the here-document refusal and its text.
    target.write_text("old\n")
    tools.Read(str(target))
    out = tools.Bash("cat > f.txt <<'EOF' < /dev/null\nnew\nEOF", "empty")
    assert out == HEREDOC_VERDICT.format(path=target) and target.read_text() == "old\n"


# --- compound commands --------------------------------------------------------

_PROGRAM = _READ + "s = s.replace('x = 1', 'x = 10')\n" + _WRITE
_SED = "sed -i 's/x = 1/x = 1\\ny = 2/'"
ACCEPTED = (
    "cd app && cat > f.txt <<'EOF' && cp f.txt g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF'\nbody\nEOF\ncp f.txt g.txt",
    "cat > f.txt <<'EOF'; cp f.txt g.txt\nbody\nEOF\ncp g.txt h.txt",
    "cat > f.txt <<'EOF' &&\nbody\nEOF\ncp f.txt g.txt",
    "cat >> f.txt <<'EOF' && echo more >> g.txt\nbody\nEOF",
    "tee f.txt <<'EOF' > /dev/null && cp f.txt g.txt\nbody\nEOF",
    "cd app && cat > f.txt <<'EOF' 2>&1 && cp f.txt g.txt &> /dev/null\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo \"a && b\" > g.txt && echo 'c\\ || d' > h.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo a\\&b \"\\\"q\\\"\" > g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo a#b > g.txt # a comment && cp f.txt h.txt\nbody\nEOF",
    "cd app && cat > f.txt <<'EOF' # note\nbody\nEOF\ncp f.txt g.txt",
    "cd app && cat > f.txt <<'EOF'\nbody\nEOF",
    f"cd app && python3 - <<'EOF' && cp a.py b.py\n{_PROGRAM}EOF",
    f"python3 - <<'EOF'\n{_PROGRAM}EOF\ncp a.py b.py",
    f"cd app && {_SED} a.py; cp a.py b.py",
    f"{_SED} app/a.py\ncp app/a.py app/b.py",
    f"cd app && {_SED} a.py",
)
"""Shapes of ``[cd DIR &&] <rewrite> [&& | ; | newline] <rest>`` that are converted."""
REFUSED = (
    "cp g.txt f.txt && cat > f.txt <<'EOF'\nbody\nEOF",  # a command before the copy
    "cat > f.txt <<'EOF' || cp f.txt g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' | cat\nbody\nEOF",
    "cat > f.txt <<'EOF' &\nbody\nEOF",
    "cd \"$D\" && cat > f.txt <<'EOF'\nbody\nEOF",  # only the shell knows the directory
    "cd && cat > f.txt <<'EOF'\nbody\nEOF",
    "cd app; cat > f.txt <<'EOF'\nbody\nEOF",
    "cd app && cd sub && cat > f.txt <<'EOF'\nbody\nEOF",
    "cd app <<'EOF' && cat > f.txt\nbody\nEOF",
    "cd app && # nothing\ncat > f.txt <<'EOF'\nbody\nEOF",
    "cat > f.txt <<'EOF' && sed -i 's/a/b/' g.txt\nbody\nEOF",  # a rewrite in the rest
    "cat > f.txt <<'EOF' && cat > g.txt <<'EOF2'\nbody\nEOF\nmore\nEOF2",
    "cat > f.txt <<'EOF'\nbody\nEOF\npython3 - <<'EOF'\nprint(1)\nEOF",
    "(cat > f.txt <<'EOF')\nbody\nEOF",
    "{ cat > f.txt <<'EOF'; }\nbody\nEOF",
    "if true; then cat > f.txt <<'EOF'; fi\nbody\nEOF",
    "! cat > f.txt <<'EOF' && cp f.txt g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo $(date) > g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo \"$(date)\" > g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo \"`date`\" > g.txt\nbody\nEOF",
    "cat > f.txt <<'EOF' && echo \"a > g.txt\nbody\nEOF",  # an unterminated quote
    "cat > f.txt <<'EOF' && cp f.txt g.txt \\\n\nbody\nEOF",  # a continued line
    "X=1 cat > f.txt <<'EOF' && cp f.txt g.txt\nbody\nEOF",
    "> f.txt <<'EOF'\nbody\nEOF",  # no program
    f"! {_SED} app/a.py",  # negated: the shell inverts its status (audit 2026-11-04 S5 F12)
    f"! {_SED} app/a.py; cp app/a.py app/b.py",
    "cat > 'f>>x' <<'EOF'\nbody\nEOF",
    "# nothing",
    f"{_SED} app/a.py && cat <<'EOF' > app/b.py\nbody\nEOF",
)
"""Shapes outside the grammar: neither recogniser converts them."""


def _populate(root: Path) -> None:
    """Create ``f.txt``, ``app/f.txt`` and ``app/a.py`` under *root*."""
    (root / "app").mkdir()
    for directory in (root, root / "app"):
        (directory / "f.txt").write_text("old\n")
        (directory / "a.py").write_text("x = 1\nz = 3\n")


def _tree(root: Path) -> dict[str, str]:
    """Return every file under *root* with its text, by relative path."""
    return {
        str(p.relative_to(root)): p.read_bytes().decode() for p in root.rglob("*") if p.is_file()
    }


def _apply(tools: UsefulTools, work: Path, command: str) -> None:
    """Apply the call *command* stands for through *tools*, then run its remainder with bash."""
    copy = heredoc_copy(command, str(work))
    if copy is not None:
        assert copy.path is not None, command
        content = copy.body
        if copy.append:
            content = Path(copy.path).read_text() + content
        tools.Read(copy.path)
        result = tools.Write(copy.path, content)
        directory, remainder = copy.cwd, copy.remainder
    else:
        edit = mechanical_edit(command, str(work))
        assert edit is not None and edit.path is not None, command
        pairs = edit.edits_for(Path(edit.path).read_text())
        assert pairs is not None, command
        tools.Read(edit.path)
        result = tools.Edit(edit.path, edits=json.dumps(
            [{"old_string": o, "new_string": n, "replace_all": e} for o, n, e in pairs]
        ))
        directory, remainder = edit.cwd, edit.remainder
    assert not result.startswith("Error"), result
    if remainder:
        _bash(f"cd {shlex.quote(directory)} && {remainder}" if directory else remainder, work)


@pytest.mark.parametrize("command", ACCEPTED, ids=[str(index) for index in range(len(ACCEPTED))])
def test_compound_rewrites_leave_the_files_bash_leaves(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    if _SED in command and not host_sed_is_gnu():
        pytest.skip("needs GNU sed as the oracle")
    base, shell, tool = tmp_path / "base", tmp_path / "shell", tmp_path / "tool"
    for root in (base, shell, tool):
        root.mkdir()
        _populate(root)
    _bash(command, shell)
    assert _tree(shell) != _tree(base)  # bash did change something
    monkeypatch.setattr(tools, "work_dir", str(tool))
    _apply(tools, tool, command)
    assert _tree(tool) == _tree(shell)


def test_compound_rewrites_name_the_cd_directory_and_the_rest() -> None:
    # The fields: the cd directory (absolute), the rest (first line's part, then the lines after).
    copy = heredoc_copy(ACCEPTED[0], "/w")
    assert copy is not None and (copy.cwd, copy.remainder) == ("/w/app", "cp f.txt g.txt")
    assert copy.path == "/w/app/f.txt" and not copy.append
    copy = heredoc_copy(ACCEPTED[2], "/w")
    assert copy is not None and (copy.cwd, copy.remainder) == ("", "cp f.txt g.txt\ncp g.txt h.txt")
    copy = heredoc_copy(ACCEPTED[4], "/w")
    assert copy is not None and copy.append and copy.remainder == "echo more >> g.txt"
    copy = heredoc_copy(ACCEPTED[11], "/w")
    assert copy is not None and (copy.cwd, copy.remainder) == ("/w/app", "")
    edit = mechanical_edit(ACCEPTED[12], "/w")
    assert edit is not None and (edit.cwd, edit.remainder) == ("/w/app", "cp a.py b.py")
    assert edit.path == "/w/app/a.py" and edit.edits == [("x = 1", "x = 10", True)]
    # An absolute cd resolves without a start directory.
    edit = mechanical_edit(f"cd /app && python3 - <<'EOF' && cp a.py b.py\n{_PROGRAM}EOF", None)
    assert edit is not None and (edit.cwd, edit.path) == ("/app", "/app/a.py")
    edit = mechanical_edit(ACCEPTED[15], "/w")
    assert edit is not None and (edit.cwd, edit.remainder) == ("", "cp app/a.py app/b.py")
    assert edit.path == "/w/app/a.py" and edit.per_line


def test_compound_shapes_outside_the_grammar_are_not_converted(tmp_path: Path) -> None:
    for command in REFUSED:
        assert heredoc_copy(command, str(tmp_path)) is None, command
        assert mechanical_edit(command, str(tmp_path)) is None, command
    # A relative cd needs a known start directory.
    command = "cd app && cat > f.txt <<'EOF' && cp f.txt g.txt\nbody\nEOF"
    assert heredoc_copy(command, None) is None
    copy = heredoc_copy(command, "/w")
    assert copy is not None and copy.cwd == "/w/app"


@needs_gnu_sed
def test_a_negated_sed_runs_as_written_with_its_status(tools: UsefulTools, tmp_path: Path) -> None:
    # ``! sed -i`` is not applied as an Edit reporting success (audit 2026-11-04 S5
    # F12): it runs as written, and the result is the status the shell inverts.
    target = tmp_path / "f.txt"
    target.write_text("old\n")
    tools.Read(str(target))
    out = tools.Bash("! sed -i 's/old/new/' f.txt", "negated")
    assert out.startswith("Error (exit code 1)") and target.read_text() == "new\n"
