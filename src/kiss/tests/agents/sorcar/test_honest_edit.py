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
import sys
import time
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


def test_edit_report_names_only_the_lines_the_file_has(work: Path) -> None:
    """The report's header ends at the file's last line; an emptied file has none to name.

    Review 2026-11-11 S4r3: the header named three lines after the replacement
    whether the file had them or not (``lines 1-4`` over the one line of a
    one-line file).
    """
    tools = UsefulTools(work_dir=str(work))
    path = work / "short.txt"
    path.write_text("1\n2\n3\n4\n5\nlast = 1\n")
    tools.Read(str(path))
    done = f"Successfully replaced 1 occurrence(s) in {path}\n"
    assert tools.Edit(str(path), "last = 1", "last = 2") == (
        done + "lines 3-6 after the edit:\n    3 | 3\n    4 | 4\n    5 | 5\n    6 | last = 2"
    )
    # The last line deleted: the lines before where it was.
    assert tools.Edit(str(path), "last = 2\n", "") == (
        done + "lines 3-5 after the edit:\n    3 | 3\n    4 | 4\n    5 | 5"
    )
    one_line = done + "lines 1-1 after the edit:\n    1 | 5"
    assert tools.Edit(str(path), "1\n2\n3\n4\n", "") == one_line
    assert tools.Edit(str(path), "5\n", "") == done + "The file is empty after the edit."
    assert path.read_text() == ""


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
    assert out.startswith("Error: edit 2: String not found in file. The closest region is lines ")
    assert (work / "a.py").read_text() == before
    # "[" * 100_000 is nested past the decoder's depth: a RecursionError, no ValueError.
    for undecodable in ("[", "[" * 100_000):
        out = tools.Edit(str(work / "a.py"), edits=undecodable)
        assert out.startswith("Error: edits must be a JSON list"), out
    assert tools.Edit(str(work / "a.py"), edits="[]").startswith("Error: edits must be a non-empty")
    assert tools.Edit(str(work / "a.py"), edits='[{"old_string": 1}]').startswith(
        "Error: edit 1 must be an object"
    )
    # A replace_all that is no JSON boolean is refused, not read by its truth ("false" is truthy).
    for value in ("false", 0, None, ""):
        quoted = json.dumps([{"old_string": "two", "new_string": "2", "replace_all": value}])
        assert tools.Edit(str(work / "a.py"), edits=quoted) == (
            "Error: edit 1: replace_all must be true or false"
        ), value
    assert (work / "a.py").read_text() == before


CODE = "def f():\n    x = 1\n    y = 2\n    return x + y\n"


def test_edit_tolerates_whitespace_differences_and_keeps_the_files_indentation(
    work: Path,
) -> None:
    tools = UsefulTools(work_dir=str(work))
    (work / "c.py").write_text(CODE)
    tools.Read(str(work / "c.py"))
    # Two-space indentation and trailing blanks in old_string: the file's four-space
    # lines are replaced, and new_string is indented as they were.
    out = tools.Edit(str(work / "c.py"), "  x = 1  \n  y = 2\n", "  x = 10\n  y = 20\n")
    assert out.startswith("Successfully replaced 1 occurrence(s) in ")
    assert "    2 |     x = 10" in out and "    3 |     y = 20" in out
    assert (work / "c.py").read_text() == "def f():\n    x = 10\n    y = 20\n    return x + y\n"
    # A CRLF file: the retry with CRLF line endings feeds the tolerant match too.
    (work / "d.py").write_text(CODE.replace("\n", "\r\n"), newline="")
    tools.Read(str(work / "d.py"))
    out = tools.Edit(str(work / "d.py"), "  return x +  y\n", "  return x - y\n")
    assert out.startswith("Successfully replaced 1 occurrence(s) in ")
    assert (work / "d.py").read_bytes().endswith(b"    y = 2\r\n    return x - y\r\n")
    # replace_all keeps the exact-match semantics: whitespace differences are not found.
    out = tools.Edit(str(work / "c.py"), "    x =  10\n", "    x = 11\n", replace_all=True)
    assert out.startswith("Error: String not found in file. The closest region is lines 2-2:")
    assert "    x = 10" in (work / "c.py").read_text()
    assert tools.Edit(str(work / "c.py"), "    x =  10\n", "    x = 11\n").startswith("Success")


def test_edit_with_an_ambiguous_whitespace_match_names_the_count(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    (work / "c.py").write_text("a\n  b\nc\na\n  b\nd\n")
    tools.Read(str(work / "c.py"))
    out = tools.Edit(str(work / "c.py"), "a\n b\n", "A\n")
    assert out == (
        "Error: String not found verbatim; ignoring whitespace it matches 2 places "
        "(not unique). Include more surrounding lines in old_string."
    )
    assert (work / "c.py").read_text() == "a\n  b\nc\na\n  b\nd\n"
    # More context makes it unique; the second match is the one edited, by position.
    out = tools.Edit(str(work / "c.py"), "a\n b\nd\n", "A\nd\n")
    assert out.startswith("Successfully replaced 1 occurrence(s) in ")
    assert (work / "c.py").read_text() == "a\n  b\nc\nA\nd\n"


def test_edit_not_found_in_a_long_file_names_the_nearest_region_at_once(work: Path) -> None:
    """A 60-line old_string not found in a 20,000-line file is answered in seconds.

    Every window of the file was compared with the old string character
    by character (110 s for this file), inside the lock that every Write
    and Edit of the daemon takes.
    """
    lines = [f"value_{i} = compute({i}, offset={i * 7})\n" for i in range(1, 20001)]
    (work / "long.py").write_text("".join(lines))
    tools = UsefulTools(work_dir=str(work))
    tools.Read(str(work / "long.py"), max_lines=1)
    old = "".join(lines[15000:15060]).replace("offset=105210)", "offset=0)")
    started = time.monotonic()
    out = tools.Edit(str(work / "long.py"), old, "pass\n")
    assert time.monotonic() - started < 15
    assert out.startswith(
        "Error: String not found in file. The closest region is lines 15001-15060:\n"
        "15001 | value_15001 = compute(15001, offset=105007)\n"
    )
    # Review 2026-11-11 S4-1: an old string longer than a file of 20,000 equal lines hung in
    # the search for the windows sharing a line with it (400 million pairs of equal lines).
    braces = "}\n" * 20_000
    (work / "braces.txt").write_text(braces)
    tools.Read(str(work / "braces.txt"), max_lines=1)
    started = time.monotonic()
    out = tools.Edit(str(work / "braces.txt"), braces + "changed\n", "pass\n")
    assert time.monotonic() - started < 15
    assert out.startswith(
        "Error: String not found in file. The closest region is lines 1-20000:\n    1 | }\n"
    )


def test_edit_not_found_names_the_nearest_region_for_a_read_free_retry(work: Path) -> None:
    tools = UsefulTools(work_dir=str(work))
    (work / "c.py").write_text(CODE)
    tools.Read(str(work / "c.py"))
    out = tools.Edit(str(work / "c.py"), "    y = 3\n    return x * y\n", "    return y\n")
    assert out == (
        "Error: String not found in file. The closest region is lines 3-4:\n"
        "    3 |     y = 2\n"
        "    4 |     return x + y\n"
        "Re-issue the Edit with the exact text of those lines as old_string."
    )
    assert (work / "c.py").read_text() == CODE
    out = tools.Edit(str(work / "c.py"), "    y = 2\n    return x + y\n", "    return y\n")
    assert out.startswith("Successfully replaced 1 occurrence(s) in ")
    assert (work / "c.py").read_text() == "def f():\n    x = 1\n    return y\n"
    # An empty file has no region to name.
    (work / "e.py").write_text("")
    tools.Read(str(work / "e.py"))
    assert tools.Edit(str(work / "e.py"), "x", "y") == "Error: String not found in file."
    # The region is numbered by newlines, as a listing is: a form feed or a CRLF
    # terminator inside the file does not shift what is shown.
    (work / "f.py").write_text("header\x0cpart\r\nTARGET = 1\r\n", newline="")
    tools.Read(str(work / "f.py"))
    out = tools.Edit(str(work / "f.py"), "TARGET = 2\n", "TARGET = 3\n")
    assert out == (
        "Error: String not found in file. The closest region is lines 2-2:\n"
        "    2 | TARGET = 1\n"
        "Re-issue the Edit with the exact text of those lines as old_string."
    )
    # Only the ``\r`` of a ``\r\n`` goes with the line's end, as in what Read shows
    # (review 2026-11-11 S4r3: every trailing ``\r`` was stripped).
    (work / "g.py").write_text("TARGET = 1\r\r\nlast\r", newline="")
    assert tools.Read(str(work / "g.py")).startswith("TARGET = 1\r\nlast\r")
    out = tools.Edit(str(work / "g.py"), "TARGET = 2\nlost", "x")
    assert out == (
        "Error: String not found in file. The closest region is lines 1-2:\n"
        "    1 | TARGET = 1\r\n"
        "    2 | last\r\n"
        "Re-issue the Edit with the exact text of those lines as old_string."
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


def _sed_numbered(path: Path, first: int, last: int) -> str:
    """Return lines *first*..*last* of *path* as ``sed -n`` ends them, numbered as a report is."""
    lines = path.read_bytes().decode().split("\n")
    if lines[-1] == "":
        lines.pop()
    listing = "".join(f"{line}\n" for line in lines[first - 1:last])
    listing = listing.replace("\r\n", "\n").removesuffix("\n")
    return "\n".join(f"{n:5d} | {line}" for n, line in enumerate(listing.split("\n"), first))


def test_a_numbered_report_ends_a_line_at_a_newline_only(work: Path) -> None:
    """The line numbers of an Edit report or refusal are those of ``sed -n`` and of Read.

    ``str.splitlines`` also ends a line at a form feed, a lone ``\\r`` and seven more
    characters: every line after one of them was numbered too high, against the
    "lines X-Y" header above it, and the old string's own lines were counted so too.
    """
    path = work / "odd.txt"
    odd = "".join(f"a{c}b\n" for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029\r")
    path.write_text(odd + "TARGET = 1\r\n" + SOURCE + "end", encoding="utf-8", newline="")
    tools = UsefulTools(work_dir=str(work))
    # Unread, and the two-line old string is not there: ten lines each side of where its
    # first line is, line 9.
    refused = tools.Edit(str(path), "a\rb\nmissing", "x")
    assert refused.endswith("exact current text.\n" + _sed_numbered(path, 1, 20))
    applied = tools.Edit(str(path), "a\rb\nTARGET = 1", "a\rb\nTARGET = 2")
    assert applied.endswith("\nlines 6-13 after the edit:\n" + _sed_numbered(path, 6, 13))
    assert tools.Read(str(path), start_line=10, max_lines=1).startswith("TARGET = 2\n")
    rewrite_command = (
        "sed -i '' 's/TARGET = 2/X/' odd.txt"
        if sys.platform == "darwin" else "sed -i 's/TARGET = 2/X/' odd.txt"
    )
    rewrite = UsefulTools(work_dir=str(work)).Bash(rewrite_command, "rewrite")
    assert rewrite.startswith("REFUSED:")
    assert rewrite.endswith("goes through.\n" + _sed_numbered(path, 1, 20))
    # The last line, line 41, is shown without a final newline, and no line 42 after one.
    for old, new in (("end", "END"), ("END", "end\n")):
        ended = tools.Edit(str(path), old, new)
        assert ended.endswith("\nlines 38-41 after the edit:\n" + _sed_numbered(path, 38, 41))
