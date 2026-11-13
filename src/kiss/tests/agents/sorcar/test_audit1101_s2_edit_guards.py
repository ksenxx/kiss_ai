# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F5-F9: shell-rewrite conversions that were not exact.

Real bash is the oracle: every command runs as written in one directory
and through the conversion path of a real :class:`UsefulTools`
(:meth:`UsefulTools._rewrite_instead`, the file shown first) in another;
a conversion that happened must leave the same files as bash.

* F5: a python program that rebinds the text to a literal, or a path
  name to the text, is not the read-replace-write pattern.
* F6: a rewrite with an input redirection (bash runs nothing when the
  file is missing), a descriptor duplication (``1>&2`` sends the body
  elsewhere) or a standard-error file (the shell creates it) is not
  converted; ``2>&1`` and ``2> /dev/null`` still are.
* F7: one simple command inside a group (``{ sed -i ...; } > other``)
  or in the background is not the lone rewrite the fast path converts.
* F8: ``>>`` in a comment or in the delimiter does not make an append.
* F9: an unquoted here-document joins a ``\\``-newline before it looks
  for the delimiter, as bash does, so a body line ending in a backslash
  cannot end the document early and hand the rest to the shell.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    heredoc_copy,
    heredoc_text,
    mechanical_edit,
    python_edit,
)
from kiss.agents.sorcar.shell_scan import simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.conftest import host_sed_is_gnu, posix_only

pytestmark = posix_only("bash, sed and python3 are the oracles")

ORIGINAL = "old original\n"
#: Commands the audit showed converted into the wrong files, with whether a
#: conversion is still expected after the fix.
CASES: dict[str, tuple[str, bool]] = {
    "python_text_rebound": (
        "python3 - <<'PY'\ns = open('f').read()\ns = 'new literal old'\n"
        "s = s.replace('old', 'new')\nopen('f', 'w').write(s)\nPY", False),
    "python_path_shadowed": (
        "python3 - <<'PY'\np = 'f'\ns = open(p).read()\np = s.replace('old', 'new')\n"
        "open(p, 'w').write(p)\nPY", False),
    "python_handle_shadows_text": (
        "python3 - <<'PY'\ns = open('f').read()\ns = s.replace('old', 'new')\n"
        "with open('f', 'w') as s:\n    s.write(s)\nPY", False),
    "python_still_converts": (
        "python3 - <<'PY'\np = 'f'\ns = open(p).read()\nt = s.replace('old', 'new')\n"
        "open(p, 'w').write(t)\nPY", True),
    # audit 2026-11-10 (B/S4-F1): programs python refuses before it runs their write
    "python_keyword_repeated": (
        "python3 - <<'PY'\ns = open('f', encoding='utf8', encoding='utf8').read()\n"
        "s = s.replace('old', 'new')\nopen('f', 'w').write(s)\nPY", False),
    "python_late_future_import": (
        "python3 - <<'PY'\np = 'f'\nfrom __future__ import annotations\ns = open(p).read()\n"
        "s = s.replace('old', 'new')\nopen(p, 'w').write(s)\nPY", False),
    "python_relative_pathlib": (
        "python3 - <<'PY'\nfrom .pathlib import Path\np = Path('f')\ns = p.read_text()\n"
        "s = s.replace('old', 'new')\np.write_text(s)\nPY", False),
    "python_relative_import": (
        "python3 - <<'PY'\nfrom . import helpers\ns = open('f').read()\n"
        "s = s.replace('old', 'new')\nopen('f', 'w').write(s)\nPY", False),
    "heredoc_comment_append": ("cat > f <<'EOF' # >> is mentioned here\nnew body\nEOF", True),
    "heredoc_delimiter_append": ("cat > f <<'>>'\nnew body\n>>", True),
    "heredoc_append": ("cat 1>> f <<'EOF'\nnew body\nEOF", True),
    "heredoc_stdout_redirected": ("cat > f <<'EOF' 1>&2\nnew body\nEOF", False),
    "heredoc_stdout_duplicated": ("cat > f <<'EOF' >&2\nnew body\nEOF", False),
    "heredoc_stderr_file": ("cat > f <<'EOF' 2> errors\nnew body\nEOF", False),
    "heredoc_stderr_merged": ("cat > f <<'EOF' 2>&1\nnew body\nEOF", True),
    "heredoc_stderr_dropped": ("cat > f <<'EOF' 2> /dev/null\nnew body\nEOF", True),
    "tee_stderr_file": ("tee f <<'EOF' 2> errors > /dev/null\nnew body\nEOF", False),
    "heredoc_join_delimiter": ("cat > f <<EOF\nold\\\nEOF\nEOF\n", True),
    "heredoc_join_then_commands": (
        "cat > f <<EOF\nold\\\nEOF\nprintf unexpected > created\nEOF\n", True),
    "heredoc_escaped_backslash": ("cat > f <<EOF\nold\\\\\nEOF\ncp f g\n", True),
    "sed_group_output": ("{ sed -i 's/old/new/g' f; } > other", False),
    "sed_subshell": ("( sed -i 's/old/new/g' f ) > other", False),
    "sed_background": ("sed -i 's/old/new/g' f &", False),
    "sed_missing_input": ("sed -i 's/old/new/g' f < missing", False),
    "sed_stderr_file": ("sed -i 's/old/new/g' f 2> errors", False),
    "sed_stdout_to_stderr": ("sed -i 's/old/new/g' f >&2", False),
    "sed_stderr_merged": ("sed -i 's/old/new/g' f 2>&1", True),
    "sed_null_input": ("sed -i 's/old/new/g' f < /dev/null", True),
    "heredoc_after_null_input": ("cat < /dev/null <<'EOF' > f\nnew body\nEOF", True),
    "heredoc_after_missing_input": ("cat < missing <<'EOF' > f\nnew body\nEOF", False),
    "sed_continued_line": ("sed -i 's/old/new/g' \\\n f", True),
    "sed_after_comment_line": ("# the change\nsed -i 's/old/new/g' f", True),
}


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under *root* with its bytes, by relative path."""
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", sorted(CASES))
def test_conversions_leave_the_files_bash_leaves(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, converts = CASES[name]
    if converts and "sed -i" in command and not host_sed_is_gnu():
        pytest.skip("needs GNU sed as the oracle")
    oracle, tools_dir = tmp_path / "oracle", tmp_path / "tools"
    for directory in (oracle, tools_dir):
        directory.mkdir()
        (directory / "f").write_text(ORIGINAL)
    subprocess.run(["bash", "-c", command], cwd=oracle, stdin=subprocess.DEVNULL,
                   capture_output=True, check=False, timeout=10)
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    tools = UsefulTools(work_dir=str(tools_dir))
    tools.Read(str(tools_dir / "f"))
    result = tools._rewrite_instead(command, str(tools_dir), name, 5, 2000)
    if result is None or result.text.startswith("REFUSED"):
        # Not converted: the shell would run it as written (or the model re-issues it).
        assert not converts, (name, result)
        assert _tree(tools_dir) == {"f": ORIGINAL.encode()}, name
    else:
        assert converts, (name, result)
        assert _tree(tools_dir) == _tree(oracle), (name, result)


def test_python_rewrite_names_keep_one_role() -> None:
    for program in (
        "s = open('f').read()\ns = 'x'\ns = s.replace('a', 'b')\nopen('f', 'w').write(s)\n",
        "p = 'f'\ns = open(p).read()\np = s.replace('a', 'b')\nopen(p, 'w').write(p)\n",
        "p = 'f'\ns = open(p).read()\np = open('g').read()\nopen(p, 'w').write(s)\n",
        "s = open('f').read()\ns = s.replace('a', 'b')\n"
        "with open('f', 'w') as s:\n    s.write(s)\n",
        "p = 'f'\ns = open(p).read()\ns = s.replace('a', 'b')\n"
        "with open(p, 'w') as p:\n    p.write(s)\np.write(s)\n",
    ):
        assert python_edit(program)[1] == [], program
    program = "p = 'f'\ns = open(p).read()\nt = s.replace('a', 'b')\nopen(p, 'w').write(t)\n"
    assert python_edit(program) == ("f", [("a", "b", True)], True)


def test_the_lone_rewrite_is_a_top_level_simple_command(tmp_path: Path) -> None:
    cwd = str(tmp_path)
    for command in ("{ sed -i 's/a/b/' f; } > other", "( sed -i 's/a/b/' f ) > other",
                    "{ sed -i 's/a/b/' f; }", "sed -i 's/a/b/' f &", "sed -i 's/a/b/' f &\n"):
        assert mechanical_edit(command, cwd) is None, command
    for command in ("sed -i 's/a/b/' \\\n f", "# note\nsed -i 's/a/b/' f", "sed -i 's/a/b/' f;",
                    "sed -i 's/a/b/' f # trailing & comment"):
        edit = mechanical_edit(command, cwd)
        assert edit is not None and edit.path == f"{cwd}/f", command


def test_append_comes_from_the_redirection_not_the_text(tmp_path: Path) -> None:
    cwd = str(tmp_path)
    for command, append in (
        ("cat > f <<'EOF' # >> is mentioned here\nbody\nEOF", False),
        ("cat > f <<'>>'\nbody\n>>", False),
        ("cat >> f <<'EOF'\nbody\nEOF", True),
        ("cat 1>> f <<'EOF'\nbody\nEOF", True),
        ("cat &>> f <<'EOF'\nbody\nEOF", True),
        ("cat > f <<'EOF' && echo '>>' > g\nbody\nEOF", False),
        ("tee -a f <<'EOF' > /dev/null\nbody\nEOF", True),
    ):
        copy = heredoc_copy(command, cwd)
        assert copy is not None and copy.append is append and copy.body == "body\n", command


def test_unquoted_bodies_join_continuations_before_the_delimiter() -> None:
    # Checked against bash 5.2: ``old\`` + newline + ``EOF`` is one line, ``oldEOF``.
    assert heredoc_text("cat > f <<EOF\nold\\\nEOF\nb\nEOF\n") == "oldEOF\nb\n"
    joined = heredoc_text("cat > f <<EOF\nold\\\nEOF\nprintf x > g\nEOF\n")
    assert joined == "oldEOF\nprintf x > g\n"
    # An escaped backslash before the newline does not join (``\\`` pairs first).
    assert heredoc_text("cat > f <<EOF\na\\\\\nEOF\n") == "a\\\n"
    assert heredoc_text("cat > f <<EOF\na\\\\\\\nEOF\nb\nEOF\n") == "a\\EOF\nb\n"
    # ``<<-`` strips the tabs of the joined line only, not of the continuation.
    assert heredoc_text("cat > f <<-EOF\n\tx\\\n\ty\n\tEOF\n") == "x\ty\n"
    assert heredoc_text("cat > f <<-EOF\n\tx\\\n\t\tEOF\n\tEOF\n") == "x\t\tEOF\n"
    # A quoted delimiter joins nothing.
    assert heredoc_text("cat > f <<'EOF'\nold\\\nEOF\n") == "old\\\n"
    assert heredoc_text("cat > f <<'EOF'\nold\\\nEOF\nb\nEOF\n") is None
    # An unterminated document (the last line continued) is none.
    assert heredoc_text("cat > f <<EOF\nold\\\nEOF\\") is None


def test_only_the_modelled_redirections_convert(tmp_path: Path) -> None:
    cwd = str(tmp_path)
    for command in (
        "cat > f <<'EOF' 1>&2\nbody\nEOF", "cat > f <<'EOF' >&2\nbody\nEOF",
        "cat > f <<'EOF' 2> errors\nbody\nEOF", "cat > f <<'EOF' 2>> errors\nbody\nEOF",
        "cat 3> g > f <<'EOF'\nbody\nEOF", "cat > f 3<<'EOF'\nbody\nEOF",
        "cat > f <<'EOF' >&-\nbody\nEOF", "tee f <<'EOF' 2> errors > /dev/null\nbody\nEOF",
        "python3 - <<'PY' < /dev/null\ns = open('f').read()\n"
        "open('f', 'w').write(s.replace('a', 'b'))\nPY",
    ):
        assert heredoc_copy(command, cwd) is None and mechanical_edit(command, cwd) is None, command
    for command in ("cat > f <<'EOF' 2>&1\nbody\nEOF", "cat > f <<'EOF' 2> /dev/null\nbody\nEOF",
                    "cat 2>&1 > f <<'EOF'\nbody\nEOF", "cat >& f <<'EOF'\nbody\nEOF"):
        copy = heredoc_copy(command, cwd)
        assert copy is not None and copy.path == f"{cwd}/f", command
    for command in ("sed -i 's/a/b/' f < missing", "sed -i 's/a/b/' f <<< x",
                    "sed -i 's/a/b/' f 2> errors", "sed -i 's/a/b/' f >&2",
                    "sed -i 's/a/b/' f 2>&1 > out", "sed -i 's/a/b/' f 3< /dev/null"):
        assert mechanical_edit(command, cwd) is None, command
    # ``/dev/null`` is nothing: as the input (always opens) or for the errors.
    for command in ("sed -i 's/a/b/' f 2>&1", "sed -i 's/a/b/' f 2> /dev/null",
                    "sed -i 's/a/b/' f < /dev/null", "sed -i 's/a/b/' f 0< /dev/null"):
        edit = mechanical_edit(command, cwd)
        assert edit is not None and edit.path == f"{cwd}/f", command
    # The scanner records every redirection, in order, with its descriptor.
    [simple] = simple_commands("cat 2>&1 > f 3< g <<'EOF' >> h\nbody\nEOF", cwd)
    assert [(fd, op, word.text) for fd, op, word in simple.redirections] == [
        ("2", ">&", "1"), ("", ">", "f"), ("3", "<", "g"), ("", "<<", "EOF"), ("", ">>", "h")]
