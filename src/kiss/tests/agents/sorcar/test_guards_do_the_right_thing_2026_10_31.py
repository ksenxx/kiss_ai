# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Proposals 5 and 7 of the 2026-10-31 audit: guards that apply the alternative, free waits.

Through a real :class:`UsefulTools` on files in ``tmp_path``:

* a ``cat``/``tee`` here-document copy into an existing file the model
  was shown is applied as the ``Write`` it stands for (``>>`` appends);
  one into a file not shown is refused naming the ``Read`` to make and
  the ``Write`` with the body; ``~/x`` resolves; ``"$OUT"`` stays refused;
* a python ``.replace`` rewrite and a one-substitution multi-line
  ``sed -i`` are applied as the ``Edit`` they stand for, or refused
  naming that exact call; ``re.sub`` stays refused with the path named;
* a long ``sleep`` waits on the last running background job instead,
  runs as written without one, and is refused only past the call's
  timeout, naming ``bash_job``;
* a foreground call outliving the detach grace goes on as a background
  job; a short one keeps the usual result; the batching hint appears
  once per 20 read-only calls.

Nothing is mocked; ``DEFAULT_CONFIG.bash_detach_seconds`` is set with
``monkeypatch``.  The here-document bodies are checked against what bash
itself writes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    edit_call,
    heredoc_copy,
    heredoc_text,
    mechanical_edit,
    python_edit,
    rewrite_guard,
    sed_edit,
)
from kiss.agents.sorcar.shell_scan import simple_commands
from kiss.agents.sorcar.useful_tools import (
    BATCHING_HINT,
    UV_REWRITTEN_NOTE,
    UsefulTools,
    _is_read_only,
    sleep_alternative,
    sleep_guard,
    uv_in_worktree,
)
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash here-documents and sleeps")


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UsefulTools:
    """A tool set on a work dir holding ``notes.md`` (``old``), with no detach grace."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    (tmp_path / "notes.md").write_text("old\n")
    return UsefulTools(work_dir=str(tmp_path))


def _job_id(started: str) -> str:
    assert started.startswith("Started background job "), started
    return started.split()[3]


# --- proposal 5: here-document copies --------------------------------------


def test_heredoc_into_a_shown_file_is_applied_as_write(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    notes = tmp_path / "notes.md"
    tools.Read(str(notes))
    out = tools.Bash("cat > notes.md <<'EOF'\nline one\nline $two `x`\nEOF", "rewrite")
    applied = f'Applied as Write(file_path="{notes}", content="line one\\nline $two `x`\\n"): '
    assert out.startswith(applied)
    assert "Successfully wrote" in out
    assert notes.read_text() == "line one\nline $two `x`\n"
    # tee, with the body an unquoted delimiter still copies verbatim
    out = tools.Bash("tee notes.md <<EOF > /dev/null\nvia tee\nEOF", "rewrite")
    assert out.startswith("Applied as Write(") and notes.read_text() == "via tee\n"
    # the body a shell view showed counts as a read too
    other = tmp_path / "other.txt"
    other.write_text("other text\n")
    assert tools.Bash("cat other.txt", "view").startswith("other text\n")
    assert tools.Bash("cat > other.txt <<'EOF'\ny\nEOF", "rewrite").startswith("Applied as Write(")
    assert other.read_text() == "y\n"


def test_append_goes_through_write_with_the_old_content(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    notes = tmp_path / "notes.md"
    tools.Read(str(notes))
    out = tools.Bash("cat >> notes.md <<'EOF'\nmore\nEOF", "append")
    assert out.startswith(f'Appended through Write(file_path="{notes}", content="old\\nmore\\n"): ')
    assert notes.read_text() == "old\nmore\n"
    out = tools.Bash("tee -a notes.md <<'EOF'\nagain\nEOF", "append")
    assert out.startswith("Appended through")
    assert notes.read_text() == "old\nmore\nagain\n"
    assert tools.Bash("tee --append notes.md <<'EOF'\n!\nEOF", "append").startswith("Appended")
    assert notes.read_text() == "old\nmore\nagain\n!\n"


def test_heredoc_into_an_unshown_file_names_read_and_write(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    notes = tmp_path / "notes.md"
    body = "x" * 100
    out = tools.Bash(f"cat > notes.md <<'EOF'\n{body}\nEOF", "rewrite")
    assert out.startswith(HEREDOC_VERDICT.format(path=notes))
    assert f'call Read(file_path="{notes}") and then the same command again' in out
    assert f'Write(file_path="{notes}", content="{"x" * 80}…")' in out  # first 80 chars
    assert notes.read_text() == "old\n"
    out = tools.Bash("cat >> notes.md <<'EOF'\nmore\nEOF", "append")
    assert f'Write(file_path="{notes}", content=<the current content followed by "more\\n">)' in out
    # After the Read the same command goes through.
    tools.Read(str(notes))
    assert tools.Bash(f"cat > notes.md <<'EOF'\n{body}\nEOF", "rewrite").startswith("Applied as")
    assert notes.read_text() == body + "\n"


def test_scratch_file_is_written_unread_as_write_allows(tools: UsefulTools, tmp_path: Path) -> None:
    scratch = tmp_path / "tmp" / "notes.md"
    scratch.parent.mkdir()
    scratch.write_text("old\n")
    assert tools.Bash("cat > tmp/notes.md <<'EOF'\nnew\nEOF", "rewrite").startswith("Applied as")
    assert scratch.read_text() == "new\n"
    # an append to a scratch file that is not text cannot be read back
    binary = tmp_path / "tmp" / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00")
    out = tools.Bash("cat >> tmp/blob.bin <<'EOF'\nmore\nEOF", "append")
    assert out.startswith(f"Error: cannot append to {binary}: ")
    assert binary.read_bytes() == b"\xff\xfe\x00"


def test_home_target_resolves_and_new_files_still_run(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # a new file under ~ is created by the shell as before
    out = tools.Bash("cat > ~/fresh.txt <<'EOF'\nfresh\nEOF", "create")
    assert "REFUSED" not in out and (home / "fresh.txt").read_text() == "fresh\n"
    # an existing one is refused until shown, then written
    out = tools.Bash("cat > ~/fresh.txt <<'EOF'\nagain\nEOF", "rewrite")
    assert out.startswith(HEREDOC_VERDICT.format(path=home / "fresh.txt"))
    tools.Read(str(home / "fresh.txt"))
    assert tools.Bash("cat > ~/fresh.txt <<'EOF'\nagain\nEOF", "rewrite").startswith("Applied as")
    assert (home / "fresh.txt").read_text() == "again\n"
    copy = heredoc_copy("cat > ~ <<'EOF'\nx\nEOF", "/w")
    assert copy is not None and copy.path == str(home)
    # a quoted tilde is a directory named ~, also when only the slash after it is quoted
    # (bash expands no tilde-prefix that holds a quoted character); one with an
    # expansion, a brace expansion included, stays unresolved
    work = tmp_path / "w"
    (work / "~").mkdir(parents=True)
    for target in ('"~/x"', '~"/x"', "~'/x'", "~\\/x"):
        command = f"cat > {target} <<'EOF'\nx\nEOF"
        copy = heredoc_copy(command, str(work))
        assert copy is not None and copy.path == f"{work}/~/x", target
        subprocess.run(["bash", "-c", command], cwd=work, check=True)
        assert (work / "~" / "x").read_text() == "x\n" and not (home / "x").exists(), target
        (work / "~" / "x").unlink()
    for target in ("~/$X", "~/*.md", "~user/x", "~/t{a,b}", "~/t{1..2}"):
        copy = heredoc_copy(f"cat > {target} <<'EOF'\nx\nEOF", "/w")
        assert copy is not None and copy.path is None, target
    # bash refuses ``~/t{a,b}`` as an ambiguous redirect: no Write of a file so named
    braces = home / "t{a,b}"
    braces.write_text("old\n")
    tools.Read(str(braces))
    out = tools.Bash("cat > ~/t{a,b} <<'EOF'\nnew\nEOF", "braces")
    assert out == HEREDOC_UNRESOLVED_VERDICT.format(path="~/t{a,b}")
    assert braces.read_text() == "old\n"
    # a ``..`` after a symbolic link is the filesystem's to resolve: bash opens the
    # file beside the directory the link names, not the one beside the link
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (home / "link").symlink_to(outside / "sub")
    (home / "target").write_text("beside the link\n")
    (outside / "target").write_text("beside its directory\n")
    tools.Read(str(home / "target"))
    tools.Read(str(outside / "target"))
    out = tools.Bash("cat > ~/link/../target <<'EOF'\nREPLACED\nEOF", "through a link")
    assert out.startswith("Applied as"), out
    assert (home / "target").read_text() == "beside the link\n"
    assert (outside / "target").read_text() == "REPLACED\n"


def test_variable_target_stays_refused(tools: UsefulTools, tmp_path: Path) -> None:
    out = tools.Bash('cat > "$OUT" <<EOF\nx\nEOF', "dynamic")
    assert out == HEREDOC_UNRESOLVED_VERDICT.format(path="$OUT")
    assert "Write(file_path=<the literal path $OUT names>, content=" in out
    # One the command assigns names the file; the refusal does too.
    out = tools.Bash('OUT=notes.md; cat > "$OUT" <<EOF\nx\nEOF', "assigned")
    assert out == HEREDOC_VERDICT.format(path=tmp_path / "notes.md")
    assert (tmp_path / "notes.md").read_text() == "old\n"


def test_only_a_single_literal_copy_is_applied(tools: UsefulTools, tmp_path: Path) -> None:
    """Expanded bodies, pipelines, wrappers and options keep the plain refusal."""
    notes = tmp_path / "notes.md"
    tools.Read(str(notes))
    for command in (
        "cat > notes.md <<EOF\n$HOME\nEOF",  # the shell would expand the body
        "cat > notes.md <<EOF\n`date`\nEOF",
        "make && cat > notes.md <<'EOF'\nx\nEOF",  # a command before the copy (not a cd)
        "cat <<'EOF' | tee notes.md\nx\nEOF",
        "cat -n <<'EOF' > notes.md\nx\nEOF",  # cat would transform the text
        "env tee notes.md <<'EOF'\nx\nEOF",  # a wrapper
        "tee notes.md other.md <<'EOF'\nx\nEOF",  # two targets
        "tee notes.md <<'EOF' > copy.txt\nx\nEOF",
        "cat <<'EOF' > notes.md > other.md\nx\nEOF",
        "cat > notes.md <<'EOF' \\\n\nx\nEOF",  # a continued operator line
        "cat > notes.md <<'EOF' || echo failed\nx\nEOF",  # not && / ; / newline after the copy
        "# a comment first\ncat > notes.md <<'EOF'\nx\nEOF",
        "cat > notes.md <<'EOF'\nx\n",  # unterminated
    ):
        out = tools.Bash(command, "copy")
        assert out == HEREDOC_VERDICT.format(path=notes), command
    assert notes.read_text() == "old\n"


def test_heredoc_text_reads_bodies_as_bash_does(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    for command in (
        "cat <<'EOF' > OUT\nhello $x `date` \\\\ \\n\nEOF",
        "cat <<EOF > OUT\nhello \\\\ \\n tab\\t\nline2\\\ncontinued\\\\\nEOF",
        "cat <<-EOF > OUT\n\tindented\n\t\tmore\n\tEOF",
        'cat <<"EOF" > OUT\n a \'b\' "c"\nEOF',
        "cat <<\\EOF > OUT\nback\\slash\nEOF",
        "cat <<E'O'F > OUT\n$x\nEOF",
        "tee OUT <<EOF > /dev/null\nvia tee\nEOF",
        "\n\n  cat <<EOF > OUT\nblank lines first\nEOF\n\n",
    ):
        subprocess.run(command.replace("OUT", str(out)), shell=True, check=True, cwd=tmp_path)
        assert heredoc_text(command) == out.read_text(), command
    for command in (
        "cat <<< 'x' > y",
        "echo $((1 << 2))",
        "cat <<EOF > x <<Y\na\nEOF",
        "cat <<'EOF > x\na\nEOF",  # an unterminated quote in the delimiter
        "cat <<\\",
        "cat << > x\na\n",  # no delimiter
    ):
        assert heredoc_text(command) is None, command


# --- proposal 5: python and sed rewrites ------------------------------------

PYTHON_TWO_REPLACES = (
    "python3 - <<'EOF'\n"
    "from pathlib import Path\n"
    "p = Path('a.py')\n"
    "s = p.read_text(encoding='utf-8')\n"
    "s = s.replace('x = 1', 'x = 10').replace('y = 2', 'y = 20')\n"
    "p.write_text(s, encoding='utf-8')\n"
    "EOF"
)


def test_python_replace_rewrite_is_applied_as_edit(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 1\ny = 2\nz = 3\n")
    call = edit_call(str(source), [("x = 1", "x = 10", True), ("y = 2", "y = 20", True)])
    # Not shown: refused with the exact call and the region; the file now counts as read.
    out = tools.Bash(PYTHON_TWO_REPLACES, "rewrite")
    assert out.startswith("REFUSED: the python heredoc did NOT run: it rewrites ")
    assert call in out and "1 | x = 1" in out
    assert source.read_text() == "x = 1\ny = 2\nz = 3\n"
    out = tools.Bash(PYTHON_TWO_REPLACES, "rewrite")
    assert out.startswith(f"Applied as {call}: Successfully replaced 2 occurrence(s)"), out
    assert source.read_text() == "x = 10\ny = 20\nz = 3\n"
    # str.replace replaces every occurrence, so the Edit does too.  The command names
    # ``python``, which not every host has, and the Edit is made only where bash finds
    # the program: one is put on PATH.
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path / "bin"), os.environ["PATH"])))
    source.write_text("a a\n")
    command = (
        "python - <<'EOF'\ns = open('a.py').read()\ns = s.replace('a', 'b')\n"
        "open('a.py', 'w').write(s)\nEOF"
    )
    out = tools.Bash(command, "rewrite")
    assert 'old_string="a", new_string="b", replace_all=True)' in out
    assert source.read_text() == "b b\n"
    # the with-open forms
    command = (
        "python3 - <<'EOF'\nwith open('a.py') as f:\n    s = f.read()\ns = s.replace('b', 'c')\n"
        "with open('a.py', 'w') as f:\n    f.write(s)\nEOF"
    )
    assert tools.Bash(command, "rewrite").startswith("Applied as Edit(")
    assert source.read_text() == "c c\n"
    # a file that does not exist: the Edit's error is reported, not an applied change
    command = (
        "python3 - <<'EOF'\ns = open('gone.py').read()\ns = s.replace('a', 'b')\n"
        "open('gone.py', 'w').write(s)\nEOF"
    )
    out = tools.Bash(command, "rewrite")
    assert out.startswith("Not applied: the command stands for Edit("), out
    assert ", which failed: Error: File not found" in out


def test_python_rewrite_outside_the_pattern_is_refused_naming_the_path(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    tools.Read(str(source))
    command = (
        "python3 - <<'EOF'\nimport re\ns = open('a.py').read()\ns = re.sub('x', 'y', s)\n"
        "open('a.py', 'w').write(s)\nEOF"
    )
    out = tools.Bash(command, "rewrite")
    assert out.startswith("REFUSED: the python heredoc did NOT run: it rewrites a.py with")
    assert 'Edit(file_path="a.py", old_string=..., new_string=...)' in out
    assert source.read_text() == "x = 1\n"
    # a body the shell would expand, or a program with a variable path, names no file
    command = (
        "python3 - <<EOF\ns = open('$F').read()\ns = s.replace('x', 'y')\n"
        "open('$F', 'w').write(s)\nEOF"
    )
    assert "Edit(file_path=<the literal path>, old_string=..." in tools.Bash(command, "rewrite")
    # an unshown file the tool cannot read as text is refused without a region
    blob = tmp_path / "blob.py"
    blob.write_bytes(b"\xff\xfe")
    command = (
        "python3 - <<'EOF'\ns = open('blob.py').read()\ns = s.replace('a', 'b')\n"
        "open('blob.py', 'w').write(s)\nEOF"
    )
    out = tools.Bash(command, "rewrite")
    assert out.startswith("REFUSED: the python heredoc did NOT run") and "Read(file_path=" in out
    assert blob.read_bytes() == b"\xff\xfe"


def test_python_edit_reads_the_pattern_exactly() -> None:
    pair = [("x", "y", True)]
    for program in (
        "s = open('a.py').read()\ns = s.replace('x', 'y')\nopen('a.py', 'w').write(s)\n",
        "s = open('a.py', 'r').read()\nt = s.replace('x', 'y')\nopen('a.py', 'wt').write(t)\n",
        "import os\np = 'a.py'\ns = open(p, encoding='utf-8').read()\ns = s.replace('x', 'y')\n"
        "open(p, 'w', encoding='utf-8').write(s)\n",
        "from pathlib import Path\ns = Path('a.py').read_text()\ns = s.replace('x', 'y')\n"
        "Path('a.py').write_text(s)\n",
        "with open('a.py') as f:\n    s = f.read()\ns = s.replace('x', 'y')\n"
        "with open('a.py', 'w') as f:\n    f.write(s)\n",
    ):
        assert python_edit(program) == ("a.py", pair, True), program
    chain = (
        "s = open('a.py').read()\ns = s.replace('a', 'b').replace('c', 'd')\n"
        "open('a.py', 'w').write(s)\n"
    )
    assert python_edit(chain) == ("a.py", [("a", "b", True), ("c", "d", True)], True)
    # Outside the pattern: the path is still named when it is a literal.
    for program in (
        "s = open('a.py').read()\ns = s.replace('x', 'y', 1)\nopen('a.py', 'w').write(s)\n",
        "s = open('a.py').read()\ns = s.replace(old, 'y')\nopen('a.py', 'w').write(s)\n",
        "s = open('a.py').read()\nopen('a.py', 'w').write(s.replace('x', 'y'))\n",
        "s = open('a.py').read()\ns = s.replace('x', 'y')\nopen('b.py', 'w').write(s)\n",
        "s = open('a.py').read()\ns = s.replace('x', 'y')\n",
        "s = open('a.py').read()\nt = open('b.py').read()\ns = s.replace('x', 'y')\n"
        "open('a.py', 'w').write(s)\n",
        "p = 'a.py'\np = 'b.py'\ns = open(p).read()\ns = s.replace('x', 'y')\n"
        "open(p, 'w').write(s)\n",
        "s = open('a.py').read()\ns = s.replace('x', 'y')\nopen('a.py', 'w').write(s)\nprint(s)\n",
        "s = open('a.py').read()\ns = s.replace('x', 'y')\nopen('a.py', 'w').write(t)\n",
        "s = open('a.py').read()\ns = s.replace('x', 'y')\nopen('a.py', 'w').write(s)\n"
        "open('a.py', 'w').write(s)\n",
        "with open('a.py') as f:\n    s = g.read()\n",
        "with open('a.py', 'w') as f:\n    g.write(s)\n",
        "with open('a.py') as f:\n    s = f.read()\n    t = s\n",
        "with open('a.py') as f, open('b.py') as g:\n    s = f.read()\n",
        "with open('a.py'):\n    pass\n",
        "with open('a.py', 'w') as f:\n    f.write(s)\n",
        "for line in open('a.py'):\n    print(line.replace('x', 'y'))\n",
        "s = Path('a.py').read_text()\ns = s.replace('x', 'y')\n"
        "Path('a.py').write_text(s, 'utf-8')\n",
    ):
        assert python_edit(program) == ("a.py", [], False), program
    for program in (
        "s = open('a.py', 'rb').read()\ns = s.replace(b'x', b'y')\nopen('a.py', 'wb').write(s)\n",
        "s = open(path).read()\ns = s.replace('x', 'y')\nopen(path, 'w').write(s)\n",
        "s = open('a.py', 'r', 'x', 'y').read()\n",
        "s = x.replace('x', 'y')\n",
        "def f(:\n",
    ):
        assert python_edit(program) == (None, [], False), program


SED_MULTILINE = "sed -i 's/y = 20/y = 200\\nw = 1/' a.py"


def test_multiline_sed_substitution_is_applied_as_edit(tools: UsefulTools, tmp_path: Path) -> None:
    source = tmp_path / "a.py"
    source.write_text("x = 10\ny = 20\n")
    call = edit_call(str(source), [("y = 20", "y = 200\nw = 1", False)])
    out = tools.Bash(SED_MULTILINE, "rewrite")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run: it rewrites ") and call in out
    assert source.read_text() == "x = 10\ny = 20\n"
    out = tools.Bash(SED_MULTILINE, "rewrite")
    assert out.startswith(f"Applied as {call}: Successfully replaced 1 occurrence(s)"), out
    assert source.read_text() == "x = 10\ny = 200\nw = 1\n"
    # the g flag replaces every occurrence; -z, -e and another delimiter are read too
    out = tools.Bash("sed -z -i -e 's|w = 1\\n|w = 2\\nv = 3\\n|g' a.py", "rewrite")
    assert "replace_all=True" in out and source.read_text() == "x = 10\ny = 200\nw = 2\nv = 3\n"
    # the shared guard keeps refusing other multi-line sed -i, naming the file
    out = tools.Bash("sed -i 's/w = 2/&\\n/' a.py", "rewrite")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run: it writes a multi-line")
    assert 'Edit(file_path="a.py", old_string=<the exact current text>' in out
    assert source.read_text() == "x = 10\ny = 200\nw = 2\nv = 3\n"


def test_sed_edit_reads_one_literal_substitution() -> None:
    def parse(command: str) -> tuple[str | None, list[tuple[str, str, bool]]]:
        operand, edits = sed_edit(simple_commands(command, None)[0])
        return None if operand is None else operand.text, edits

    assert parse("sed -z -i 's/a\\nb/c/' f") == ("f", [("a\nb", "c", False)])
    assert parse("sed -i 's/a/b\\n/g' f") == ("f", [("a", "b\n", True)])
    assert parse("sed --in-place --expression='s/a/b\\n/' f") == ("f", [("a", "b\n", False)])
    assert parse("sed -i -es/a/b\\\\n/ f") == ("f", [("a", "b\n", False)])
    assert parse("sed -i 's/a\\.b\\*\\[\\]\\^\\$\\//c\\/\\&\\t\\\\/' f") == (
        "f", [("a.b*[]^$/", "c/&\t\\", False)])
    assert parse("sed -i 's/a/b\\n/' f") == ("f", [("a", "b\n", False)])
    for command in (
        "sed -i 's/a\\nb/c/' f",  # without -z a pattern with a newline never matches
        "sed -i 's/a.b/c\\n/' f",  # a bare metacharacter
        "sed -i 's/a/&\\n/' f",  # the matched text
        "sed -i 's/a/\\1\\n/' f",  # a back-reference
        "sed -i 's/a\\+/b\\n/' f",  # an escape that is no literal
        "sed -i 's/a/b\\n/i' f",  # a flag that is no g
        "sed -i 's/a/b\\n' f",  # unterminated
        "sed -i 'y/a/b\\n/' f",
        "sed -i 'ss' f",
        "sed -i 's/a\\n/a\\n/' f",  # old == new
        "sed -i 's//a\\n/' f",  # empty old
        "sed -i -E 's/a/b\\n/' f",
        "sed -i.bak 's/a/b\\n/' f",
        "sed -n -i 's/a/b\\n/' f",
        "sed -i ':a;N;$!ba;s/a\\nb/c/' f",
        "sed 's/a/b\\n/' f",
    ):
        assert parse(command) == ("f", []), command
    assert parse("sed -i 's/a/b\\n/' f g") == (None, [])
    assert parse("sed -i 's/a/b\\n/'") == (None, [])
    assert parse("sed -i 's/a/b\\n/' -e 's/c/d/' f") == (None, [])  # with -e, both are files
    # Since 2026-10-31 the rest of a compound command comes back for the caller to run.
    compound = mechanical_edit("sed -i 's/a/b\\n/' f && cat f", "/w")
    assert compound is not None and (compound.cwd, compound.remainder) == ("", "cat f")
    assert mechanical_edit("sed -i 's/a/b\\n/' f > out", "/w") is None
    assert mechanical_edit("sed -i 's/a/b\\n/' \"$F\"", "/w") is not None
    assert mechanical_edit("sed -i 's/a/b\\n/' \"$F\"", "/w").path is None  # type: ignore[union-attr]
    assert mechanical_edit("echo x", "/w") is None
    # the shared guard names the Edit call when the script is mechanical
    guard = rewrite_guard("sed -i 's/a/b\\n/' f")
    assert guard is not None and 'Edit(file_path="f", old_string="a", new_string="b\\n")' in guard
    guard = rewrite_guard("sed -i 's/a/b\\n/' f g")
    assert guard is not None and "Edit(file_path=<the file>, old_string=<the exact" in guard
    guard = rewrite_guard(
        "python3 - <<'EOF'\ns = open('f').read()\ns = s.replace('a', 'b')\n"
        "open('f', 'w').write(s)\nEOF"
    )
    assert guard is not None
    assert 'Make the same change with Edit(file_path="f", old_string="a"' in guard


def test_rewrites_on_another_machine_are_refused_not_applied(
    tools: UsefulTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    tools.Read(str(source))
    out = tools.Bash(SED_MULTILINE, "remote", background=True, machine="bench")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run")
    assert source.read_text() == "x = 1\n"


# --- proposal 5: sleeps -------------------------------------------------------


def test_long_sleep_waits_on_the_running_job_and_runs_the_rest(tools: UsefulTools) -> None:
    job = _job_id(tools.Bash("sleep 1; echo done", "slow", background=True))
    before = time.monotonic()
    out = tools.Bash("sleep 60; echo after", "poll", timeout_seconds=100)
    assert time.monotonic() - before < 10
    waited = f"Waited on job {job} for up to 60 s instead of sleeping (exited with code 0).\n"
    assert out == "after\n" + waited  # the note after the result, whose head readers parse
    # ...the last started running job, in any spelling the text search finds
    first = _job_id(tools.Bash("sleep 30", "slow", background=True))
    last = _job_id(tools.Bash("sleep 0.5", "quick", background=True))
    command = "echo before && nohup /bin/sleep '1' 1m >/dev/null && echo after"
    out = tools.Bash(command, "poll", timeout_seconds=100)
    waited = f"Waited on job {last} for up to 61 s instead of sleeping (exited with code 0).\n"
    assert out == "before\nafter\n" + waited
    tools.bash_job(first, action="kill")
    report = tools.bash_job(first, action="wait", timeout_seconds=10)
    assert report.startswith(f"Job {first}: killed")
    # on_timeout="background" keeps the note
    assert tools.Bash("sleep 1", "slow", background=True)
    out = tools.Bash("sleep 40; echo x", "poll", timeout_seconds=50, on_timeout="background")
    assert out.startswith("x\n") and "Waited on job " in out


def test_long_sleep_without_a_running_job_runs_as_written(tools: UsefulTools) -> None:
    job = _job_id(tools.Bash("true", "done", background=True))
    tools.bash_job(job, action="wait", timeout_seconds=10)
    before = time.monotonic()
    out = tools.Bash("timeout 1 sleep 60; echo ran", "poll", timeout_seconds=100)
    assert 0.9 < time.monotonic() - before < 10
    # ...and the note names the one-call wait and the adopted job to wait on instead
    assert out == "ran\n" + sleep_alternative(60)
    assert out.startswith("ran\n\n[note: the sleep of 60 s ran as written: no background job")
    assert 'Bash(command, until="<regex>", poll_seconds=N, timeout_seconds=M)' in out
    assert 'bash_job(<id>, "wait")' in out
    # a loop's sleep with no job: as written, with the same note
    command = "for i in 1; do timeout 1 sleep 31; done; echo ran"
    assert tools.Bash(command, "poll", timeout_seconds=100) == "ran\n" + sleep_alternative(31)
    # a sleep the text search cannot take out runs as written, job or not, without the note
    job = _job_id(tools.Bash("sleep 5", "slow", background=True))
    out = tools.Bash("timeout 1 sleep 6'0'; echo ran", "poll", timeout_seconds=100)
    assert out == "ran\n"
    tools.bash_job(job, action="kill")


def test_sleep_pacing_a_loop_runs_as_written_even_with_a_job(tools: UsefulTools) -> None:
    """Swapping a loop's sleep for ``true`` would spin the loop; it paces the loop as written."""
    job = _job_id(tools.Bash("sleep 30", "slow", background=True))
    for command in (
        "for i in 1 2; do timeout 1 sleep 31; done; echo ran",
        "n=0; while [ $n -lt 2 ]; do timeout 1 sleep 31; n=$((n+1)); done; echo ran",
        "n=0; until [ $n -ge 2 ]; do { timeout 1 sleep 31; }; n=$((n+1)); done; echo ran",
    ):
        before = time.monotonic()
        out = tools.Bash(command, "poll", timeout_seconds=100)
        assert out == "ran\n", (command, out)
        assert time.monotonic() - before >= 1.9, command
    tools.bash_job(job, action="kill")
    # a sleep in a conditional, not a loop, is still spent on the job
    job = _job_id(tools.Bash("sleep 0.5", "quick", background=True))
    out = tools.Bash("if true; then sleep 60; fi; echo ran", "poll", timeout_seconds=100)
    assert out.startswith("ran\n") and f"Waited on job {job} for up to 60 s" in out


def test_sleep_longer_than_the_timeout_is_refused_naming_bash_job(
    tools: UsefulTools, tmp_path: Path,
) -> None:
    out = tools.Bash("touch ran; sleep 100", "poll", timeout_seconds=50)
    assert out.startswith(
        "REFUSED: the Bash call did NOT run: a foreground sleep of 100s is a polling turn."
    )
    call = 'bash_job(\'job_id\', action="wait", until="<regex in its log>", timeout_seconds=100)'
    assert call in out
    assert "Bash(<its command>, background=True)" in out
    assert not (tmp_path / "ran").exists()
    job = _job_id(tools.Bash("sleep 5", "slow", background=True))
    out = tools.Bash("sleep infinity", "poll", timeout_seconds=50)
    assert "a foreground sleep of infinity is a polling turn" in out
    call = f'bash_job({job!r}, action="wait", until="<regex in its log>", timeout_seconds=3600)'
    assert call in out
    tools.bash_job(job, action="kill")
    # the parallel runner keeps its own 30 s limit
    assert sleep_guard("sleep 31") is not None and sleep_guard("sleep 30") is None
    report = tools.run_commands_parallel('["sleep 31"]')
    assert "a foreground sleep of 31s" in report


# --- proposal 7: auto-detach ------------------------------------------------


def test_foreground_command_outliving_the_grace_becomes_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 1)
    pieces: list[str] = []
    tools = UsefulTools(stream_callback=pieces.append, work_dir=str(tmp_path))
    before = time.monotonic()
    out = tools.Bash("echo early; sleep 3; echo late", "long", timeout_seconds=10)
    assert time.monotonic() - before < 2.5, out
    assert out.startswith("Still running after 1s; left running as background job j")
    job = out.split("background job ")[1].split()[0]
    assert f'bash_job({job!r}, action="wait", until=<regex>, timeout_seconds=9).' in out
    assert out.endswith("Output so far:\nearly\n") and pieces == ["early\n"]
    report = tools.bash_job(job, action="wait", timeout_seconds=10)
    assert "exited with code 0" in report and report.endswith("early\nlate\n")
    # a command finishing within the grace keeps the usual result, exit code and view
    # bookkeeping, and leaves neither a job entry nor a log behind
    assert tools.Bash("echo quick", "quick", timeout_seconds=10) == "quick\n"
    out = tools.Bash("echo bad; exit 3", "bad", timeout_seconds=10)
    assert out == "Error (exit code 3):\nbad\n"
    assert list(tools._jobs) == [job]
    assert [p.stem for p in (tmp_path / "tmp" / "bash_jobs").glob("*.log")] == [job]
    notes = tmp_path / "notes.md"
    notes.write_text("old\n")
    assert tools.Bash("cat notes.md", "view", timeout_seconds=10).startswith("old\n")
    assert tools.Edit(str(notes), "old", "new").startswith("Successfully")
    # a timeout within the grace, background=True and on_timeout="background" are unchanged
    before = time.monotonic()
    out = tools.Bash("sleep 2; echo late", "short", timeout_seconds=1)
    assert out.startswith("Error: Command execution timeout") and time.monotonic() - before < 2
    out = tools.Bash("sleep 2; echo late", "bg", timeout_seconds=1.5, on_timeout="background")
    assert out.startswith("Still running after 1.5s") and "timeout_seconds=1.5)" in out
    for job in list(tools._jobs):
        tools.bash_job(job, action="kill")


def test_grace_falls_back_to_the_foreground_when_no_job_can_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 1)
    (tmp_path / "tmp").write_text("not a directory\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    before = time.monotonic()
    assert tools.Bash("sleep 1.5; echo late", "long", timeout_seconds=10) == "late\n"
    assert time.monotonic() - before >= 1.4
    assert not tools._jobs
    out = tools.Bash("echo x", "bg", timeout_seconds=10, on_timeout="background")
    assert out.startswith("Error: could not start background job")


def test_tool_panel_stop_during_the_grace_kills_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreground command's Stop kills it (as a foreground shell), on the grace path too."""
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 5)
    tools = UsefulTools(work_dir=str(tmp_path))
    outcome: dict[str, object] = {}

    def run() -> None:
        token = new_tool_call("Bash")
        register_tool_call(token)
        try:
            try:
                outcome["result"] = tools.Bash("sleep 20; echo late", "long", timeout_seconds=30)
                end_tool_call(token)
            except ToolCallInterrupted:
                outcome["interrupted"] = True
        finally:
            unregister_tool_call(token)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not tools._jobs and time.monotonic() < deadline:
        time.sleep(0.05)
    (job,) = tools._jobs.values()
    assert thread.ident is not None
    assert interrupt_tool_call(thread.ident, "Bash") is True
    thread.join(timeout=30)
    assert not thread.is_alive() and outcome == {"interrupted": True}, outcome
    deadline = time.monotonic() + 5
    while job.returncode() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.returncode() is not None and not tools._jobs


def test_stop_ends_the_grace_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 5)
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
    threading.Timer(0.5, stop.set).start()
    before = time.monotonic()
    out = tools.Bash("sleep 20; echo late", "long", timeout_seconds=30)
    assert time.monotonic() - before < 4
    assert "left running as background job" in out or "exited" in out or out.startswith("Error")
    for job in list(tools._jobs):
        tools.bash_job(job, action="kill")


# --- proposal 7: the batching hint -----------------------------------------


def test_batching_hint_once_per_twenty_read_only_calls(tools: UsefulTools) -> None:
    first = tools.Bash("ls", "list")
    assert first.endswith(BATCHING_HINT) and first.startswith("notes.md\n")
    for _ in range(20):
        assert BATCHING_HINT not in tools.Bash("ls", "list")
    assert tools.Bash("grep -n old notes.md", "grep").endswith(BATCHING_HINT)
    # a hint is for Bash calls only, and not for a slow, long or writing one
    assert tools.batching_hint("Read", {}, "x", 0.0) == ""
    assert tools.batching_hint("Bash", {"command": "ls"}, "x", 2.0) == ""
    assert tools.batching_hint("Bash", {"command": "ls"}, "x" * 2000, 0.0) == ""
    assert tools.batching_hint("Bash", {"command": "touch x"}, "", 0.0) == ""
    assert tools.batching_hint("Bash", {}, "", 0.0) == ""


def test_read_only_commands() -> None:
    for command in (
        "grep -rn foo src | head -5", "cat a.py", "ls -la", "sed -n '1,5p' a.py", "tail -n 3 log",
        "git log --oneline -3", "git status --short", "git diff HEAD~1", "git show HEAD:f",
        "find . -name '*.py'", "wc -l a.py; wc -c b.py",
    ):
        assert _is_read_only(command), command
    for command in (
        "", "echo x", "cat a > b", "sed -i 's/a/b/' f", "sed 's/a/b/' f", "sed -n -i p f",
        "git commit -m x", "git", "git -C x log", "find . -delete", "find . -exec rm {} \\;",
        "find . -fprint out", "ls | python3 x.py", "rm -rf build",
    ):
        assert not _is_read_only(command), command


def test_uv_run_in_a_worktree_runs_as_the_shared_environments_binary(tmp_path: Path) -> None:
    """A worktree shares the main checkout's environment: ``uv sync``/``uv run`` there
    would build it one of its own or re-point the shared editable install (the
    2026-11-01 audit's 19 ``check --full`` runs). A ``uv run`` whose rewrite is
    unambiguous runs as the environment's binary on the worktree's sources, the
    result saying so (the 2026-11-03 audit's 6 refusals were each followed by the
    agent copying the suggested command by hand); the rest are refused naming what
    to run."""
    repo = tmp_path / "repo"
    worktree = repo / ".kiss-worktrees" / "kiss_wt-1-abc"
    worktree.mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    for binary in ("python", "check", "pytest"):
        (repo / ".venv" / "bin" / binary).write_text("#!/bin/sh\necho ran $0 \"$@\"\n")
        (repo / ".venv" / "bin" / binary).chmod(0o755)
    sources = f"PYTHONPATH={worktree}/src:{worktree}/projects"
    tools = UsefulTools(work_dir=str(worktree))
    # uv run: rewritten and run, the note after the result (review 2026-11-03: a
    # note before it hid the timeout or error the result opens with)
    rewritten = f"{sources} {repo}/.venv/bin/check --full --fix"
    out = tools.Bash("uv run check --full --fix", "lint")
    assert out == (f"ran {repo}/.venv/bin/check --full --fix\n"
                   + UV_REWRITTEN_NOTE.format(command=rewritten))
    (repo / ".venv" / "bin" / "slow").write_text("#!/bin/sh\nsleep 5\n")
    (repo / ".venv" / "bin" / "slow").chmod(0o755)
    timed_out = tools.Bash("uv run slow", "slow", timeout_seconds=1)
    assert timed_out.startswith("Error: Command") and "time" in timed_out.splitlines()[0]
    assert timed_out.endswith(UV_REWRITTEN_NOTE.format(command=f"{sources} {repo}/.venv/bin/slow"))
    assert tools.Bash(rewritten, "the rewrite itself runs").strip() == (
        f"ran {repo}/.venv/bin/check --full --fix")
    # every uv run of the command is rewritten; the fan-out tool rewrites too
    report = tools.run_commands_parallel(
        '["uv run pytest -q a.py && uv run python -c 1", "uv sync"]', description="both")
    assert (f"{sources} {repo}/.venv/bin/pytest -q a.py && {sources} {repo}/.venv/bin/python -c 1"
            in report)
    assert f"ran {repo}/.venv/bin/pytest -q a.py\nran {repo}/.venv/bin/python -c 1" in report
    assert "REFUSED: `uv sync` did NOT run" in report
    for command, expected in (
        ("cd src && uv run --no-sync pytest -q a.py | tail -3",
         f"cd src && {sources} {repo}/.venv/bin/pytest -q a.py | tail -3"),
        ("UV_PROJECT_ENVIRONMENT=../../.venv uv run --no-project python3 -c 'print(1)'",
         f"UV_PROJECT_ENVIRONMENT=../../.venv {sources} {repo}/.venv/bin/python -c 'print(1)'"),
        # global flags before the subcommand are dropped from the rewrite
        ("uv -q --offline run pytest -q a.py", f"{sources} {repo}/.venv/bin/pytest -q a.py"),
        # 2026-11-04 audit S5 F2: the words of a quoted argument are text, the
        # command after them is the one rewritten
        ("printf '%s\\n' 'uv run python --version'; uv run python -c 'print(19)'",
         f"printf '%s\\n' 'uv run python --version'; {sources} {repo}/.venv/bin/python -c "
         "'print(19)'"),
        # F3: after a wrapper the assignment needs ``env`` to be one
        ("command uv run python -c 'print(23)'",
         f"command env {sources} {repo}/.venv/bin/python -c 'print(23)'"),
        ("timeout 2 uv run python -c 'print(23)'",
         f"timeout 2 env {sources} {repo}/.venv/bin/python -c 'print(23)'"),
        # F4: ``-m`` runs a module of the environment's Python, not a binary of its name
        ("uv run -m pytest --version", f"{sources} {repo}/.venv/bin/python -m pytest --version"),
        ("uv run -q --module pytest -q a.py",
         f"{sources} {repo}/.venv/bin/python -m pytest -q a.py"),
    ):
        assert uv_in_worktree(command, str(worktree)) == (
            None, UV_REWRITTEN_NOTE.format(command=expected), expected)
    assert tools.Bash("timeout 2 uv run python -c 'print(23)'", "wrapped").startswith(
        f"ran {repo}/.venv/bin/python -c print(23)\n")
    # F3: paths with spaces are quoted (the rewrite runs through the real shell)
    spaced = tmp_path / "repo with spaces"
    (spaced / ".kiss-worktrees" / "kiss_wt-2-def").mkdir(parents=True)
    shutil.copytree(repo / ".venv", spaced / ".venv")
    spaced_wt = spaced / ".kiss-worktrees" / "kiss_wt-2-def"
    assert UsefulTools(work_dir=str(spaced_wt)).Bash("uv run python -c 'print(29)'", "spaced") == (
        f"ran {spaced}/.venv/bin/python -c print(29)\n" + UV_REWRITTEN_NOTE.format(
            command=f"PYTHONPATH='{spaced_wt}/src:{spaced_wt}/projects' "
                    f"'{spaced}/.venv/bin/python' -c 'print(29)'"))
    # a program the environment lacks, the sync itself, an option whose value
    # (``--directory .``, before or after ``run``) a rewrite would drop, a flag
    # that changes what the program word means (F4), and a uv run next to one
    # of those: refused as a whole, nothing rewritten
    for command in ("uv run nosuchtool --x", "uv sync", "echo x && uv add requests",
                    "uv --directory . run check --full --fix", "uv --project=. sync",
                    "uv --color never -q run pytest", "uv run pytest -q && uv sync",
                    "uv run --directory /tmp python -c 'import os; print(os.getcwd())'",
                    "uv run --project . python3 -c 1", "uv run -p 3.12 pytest -q",
                    "uv run --with=rich python -c 1", "uv run --script check",
                    "uv run -m 'os;rm' -c 1", "uv run --help pytest"):
        verdict, note, unchanged = uv_in_worktree(command, str(worktree))
        assert verdict is not None and "bin/check --full" in verdict, command
        assert (note, unchanged) == ("", command)
        assert tools.Bash(command, "refused") == verdict
    # ``uv`` as text (a heredoc body, a quoted word) or outside a worktree runs
    for command in ("cat > s.sh <<'EOF'\nuv run pytest\nEOF\n", "echo 'uv sync'", "uv lock",
                    "uv --version", "uv -q pip list", "uv tool run ruff check ."):
        assert uv_in_worktree(command, str(worktree)) == (None, "", command), command
    assert uv_in_worktree("uv run pytest", str(repo)) == (None, "", "uv run pytest")
