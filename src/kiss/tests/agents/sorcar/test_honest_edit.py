# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The honest edit is the easy edit (``edit_guards.py`` and the Edit / Bash tools).

A file counts as read when any tool showed it (Read, Write, a Bash
``cat`` / ``sed -n`` / ``grep -n`` of that one file), at either of its
checkouts (main or worktree).  An Edit of an unread file returns the
region around ``old_string`` instead of a bare refusal; an Edit takes a
list of pairs and reports the lines around each replacement; shell
rewrites that dodge Edit are refused.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    output_shows_file,
    region_around_string,
    repo_read_key,
    rewrite_guard,
    shell_view_target,
)
from kiss.agents.sorcar.useful_tools import UsefulTools

SOURCE = "".join(f"line {i}\n" for i in range(1, 31))


@pytest.fixture
def work(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    return tmp_path


def test_unread_edit_is_not_applied_but_shows_the_region_and_counts_as_read(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    out = tools.Edit(str(work / "a.py"), "line 12\n", "line twelve\n")
    assert out.startswith("Error:") and "edit was NOT applied" in out
    assert "    2 | line 2" in out and "   12 | line 12" in out and "   22 | line 22" in out
    assert "    1 |" not in out and "   23 |" not in out
    assert (work / "a.py").read_text() == SOURCE
    again = tools.Edit(str(work / "a.py"), "line 12\n", "line twelve\n")
    assert again.startswith("Successfully replaced")
    assert "line twelve" in (work / "a.py").read_text()


def test_edit_reports_three_lines_of_context_around_the_replacement(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    tools.Read(str(work / "a.py"))
    out = tools.Edit(str(work / "a.py"), "line 10\n", "line ten\nline ten-b\n")
    assert out.startswith("Successfully replaced 1 occurrence(s) in ")
    assert "\nlines 7-14 after the edit:\n" in out
    assert "    7 | line 7" in out and "   10 | line ten" in out
    assert "   11 | line ten-b" in out and "   14 | line 13" in out
    assert "   15 |" not in out


def test_edit_applies_several_pairs_in_order_or_none(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    tools.Read(str(work / "a.py"))
    edits = json.dumps([
        {"old_string": "line 1\n", "new_string": "one\n"},
        {"old_string": "line 2", "new_string": "two", "replace_all": True},
    ])
    out = tools.Edit(str(work / "a.py"), edits=edits)
    assert out.startswith("Successfully replaced 12 occurrence(s) in ")
    assert "edit 1 (1 occurrence(s)); lines 1-4 after the edit:" in out
    assert "edit 2 (11 occurrence(s)); lines 1-5 after the edit:" in out
    text = (work / "a.py").read_text()
    assert text.startswith("one\ntwo\n") and "two0\n" in text and "line 2" not in text
    # A failing pair leaves the file untouched.
    before = text
    bad = json.dumps([
        {"old_string": "one\n", "new_string": "uno\n"},
        {"old_string": "missing", "new_string": "x"},
    ])
    out = tools.Edit(str(work / "a.py"), edits=bad)
    assert out == "Error: edit 2: String not found in file"
    assert (work / "a.py").read_text() == before
    assert tools.Edit(str(work / "a.py"), edits="[").startswith("Error: edits must be a JSON list")
    assert tools.Edit(str(work / "a.py"), edits="[]").startswith("Error: edits must be a non-empty")
    assert tools.Edit(str(work / "a.py"), edits='[{"old_string": 1}]').startswith(
        "Error: edit 1 must be an object"
    )


def test_bash_view_of_one_file_counts_as_read(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    assert "line 3" in tools.Bash("sed -n 1,5p a.py", "view")
    assert tools.Edit(str(work / "a.py"), "line 3\n", "line three\n").startswith("Successfully")
    # Forward slashes: the Bash tool is bash on every OS (Git bash on Windows).
    assert "b = 1" in tools.Bash(f"cd {work.as_posix()} && cat b.py", "view b")
    assert tools.Edit(str(work / "b.py"), "b = 1", "b = 2").startswith("Successfully")
    # Two operands, a failing command, or a grep over a directory are not reads.
    (work / "c.py").write_text("c = 1\n")
    (work / "d.py").write_text("d = 1\n")
    tools.Bash("cat c.py d.py", "two files")
    assert tools.Edit(str(work / "c.py"), "c = 1", "c = 2").startswith("Error:")
    tools.Bash("grep -n nothing-here d.py", "grep miss (exit 1)")
    assert tools.Edit(str(work / "d.py"), "d = 1", "d = 2").startswith("Error:")
    assert "d = 1" in tools.Bash("grep -n 'd = ' d.py", "grep hit")
    assert tools.Edit(str(work / "d.py"), "d = 1", "d = 2").startswith("Successfully")


def test_shell_view_target_parsing(work: Path) -> None:
    a = (work / "a.py").resolve()
    assert shell_view_target("cat a.py", str(work)) == a
    assert shell_view_target("sed -n '10,20p' a.py", str(work)) == a
    assert shell_view_target("grep -n 'line 1' a.py", str(work)) == a
    assert shell_view_target("grep -n -e 'line 1' a.py", str(work)) == a
    assert shell_view_target("head -n 20 a.py", str(work)) == a
    assert shell_view_target("nl a.py | head", str(work)) == a
    assert shell_view_target(f"cd {work.as_posix()} && sed -n 1,3p a.py", "/") == a
    assert shell_view_target("cat a.py b.py", str(work)) is None
    assert shell_view_target("grep -rn x .", str(work)) is None
    assert shell_view_target("ls a.py", str(work)) is None
    assert shell_view_target("cat missing.py", str(work)) is None
    assert shell_view_target("cat 'unterminated", str(work)) is None


def test_shell_rewrites_are_refused_with_the_edit_call_named(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    tools.Read(str(work / "a.py"))
    out = tools.Bash("sed -i 's/line 1$/line 1\\nextra/' a.py", "sed rewrite")
    assert out.startswith("REFUSED: the `sed -i` call did NOT run") and "Edit(file_path" in out
    heredoc = (
        "python3 - <<'EOF'\np='a.py'\ns=open(p).read()\n"
        "open(p,'w').write(s.replace('line 1','x'))\nEOF"
    )
    out = tools.Bash(heredoc, "python rewrite")
    assert out.startswith("REFUSED: the python heredoc did NOT run") and "edits=[" in out
    assert (work / "a.py").read_text() == SOURCE
    # A single-line sed -i and a python heredoc that only reads are allowed.
    assert rewrite_guard("sed -i 's/a/b/' a.py") is None
    assert rewrite_guard("python3 - <<'EOF'\nprint(open('a.py').read())\nEOF") is None
    # ``-i.bak`` (suffix attached) is the one in-place spelling GNU and BSD sed share.
    assert tools.Bash("sed -i.bak 's/line 30/line thirty/' a.py", "one-line sed") == ""
    assert (work / "a.py").read_text().endswith("line thirty\n")
    # A heredoc that replaces text but only prints it (an analysis) is allowed too;
    # one that writes through write_text or a mode "a" open is not.
    analysis = "python3 - <<'EOF'\nimport re\nprint(re.sub('x', 'y', open('a.py').read()))\nEOF"
    assert rewrite_guard(analysis) is None
    to_stdout = (
        "python3 - <<'EOF'\nimport sys\n"
        "sys.stdout.write(open('a.py').read().replace('a', 'b'))\nEOF"
    )
    assert rewrite_guard(to_stdout) is None
    writers = (
        "python3 - <<'EOF'\nfrom pathlib import Path\np=Path('a.py')\n"
        "p.write_text(p.read_text().replace('a','b'))\nEOF",
        "python3 - <<'EOF'\ns=open('a.py').read()\nopen('a.py','a').write(s.replace('a','b'))\nEOF",
    )
    for command in writers:
        refusal = rewrite_guard(command)
        assert refusal is not None and refusal.startswith("REFUSED: the python heredoc"), command


def test_output_shows_file_recognises_numbered_views(work: Path) -> None:
    a = work / "a.py"
    assert output_shows_file("line 3\n", a)
    assert output_shows_file("3:line 3\n", a), "grep -n"
    assert output_shows_file("     3\tline 3\n", a), "nl"
    assert output_shows_file("2-line 2\n3:line 3\n", a), "grep -n with context"
    assert not output_shows_file("label\n", a)
    assert not output_shows_file("\n\n", a)
    assert not output_shows_file("l\n", a), "a one-character line proves nothing"
    assert not output_shows_file("line 3\n", work / "missing.py")


def test_a_shell_view_counts_as_read_only_when_it_showed_something(work: Path) -> None:
    """A failing or silent view leaves the file unread (an unread Edit refusal then reads it)."""
    # A fresh instance per case: the refused Edit itself shows the region and counts as a read.
    silent_views = (
        "sed -n 999p a.py",
        "cat a.py; exit 2",
        "grep -q 'line 1' a.py",
        "echo label; true || cat a.py",  # prints, but nothing of a.py
    )
    for command in silent_views:
        tools = UsefulTools(work_dir=str(work))
        tools.Bash(command, "view that shows nothing usable")
        assert not tools._is_read((work / "a.py").resolve()), command
        assert tools.Edit(str(work / "a.py"), "line 1\n", "one\n").startswith("Error:"), command
        assert (work / "a.py").read_text() == SOURCE
    # run_commands_parallel views count the same way.
    tools = UsefulTools(work_dir=str(work))
    tools.run_commands_parallel(
        json.dumps(["cat a.py", "grep -q line b.py"]), description="views"
    )
    assert tools.Edit(str(work / "a.py"), "line 1\n", "one\n").startswith("Successfully")
    assert tools.Edit(str(work / "b.py"), "b = 1", "b = 2").startswith("Error:")


def test_read_key_is_shared_between_checkout_and_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "src").mkdir()
    (repo / "src" / "m.py").write_text("v = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-q", str(wt), "-b", "side"], cwd=repo, check=True)
    main_file = (repo / "src" / "m.py").resolve()
    wt_file = (wt / "src" / "m.py").resolve()
    assert repo_read_key(main_file) == repo_read_key(wt_file)
    assert repo_read_key(main_file).endswith("::src/m.py")
    other = tmp_path / "loose.py"
    other.write_text("")
    assert repo_read_key(other.resolve()) == str(other.resolve())
    # Read at the main checkout, edit at the worktree path.
    tools = UsefulTools(work_dir=str(wt))
    tools.Read(str(main_file))
    assert tools.Edit(str(wt_file), "v = 1", "v = 2").startswith("Successfully")
    assert wt_file.read_text() == "v = 2\n" and main_file.read_text() == "v = 1\n"


def test_region_around_string_falls_back_to_the_first_line_or_the_head() -> None:
    text = "".join(f"l{i}\n" for i in range(1, 41))
    exact = region_around_string(text, "l20\nl21\n", context=2)
    assert exact.splitlines()[0] == "   18 | l18" and exact.splitlines()[-1] == "   23 | l23"
    partial = region_around_string(text, "l30\nnot there\n", context=1)
    assert "   30 | l30" in partial
    head = region_around_string(text, "zzz", context=2)
    assert head.splitlines()[0] == "    1 | l1" and len(head.splitlines()) == 4
