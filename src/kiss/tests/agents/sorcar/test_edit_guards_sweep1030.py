# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-30, slice S7: the shell scanner and the edit guards.

* F7: the files of two submodules shared one read-before-modify key
  (``_main_checkout`` took the parent of any ``.git`` path segment, so a
  submodule's ``gitdir: ../.git/modules/x`` named the superproject).
* F9: ``$((1 << 2))`` and ``((x << 3))`` opened a fake here-document
  that swallowed the rest of the command, hiding later writes and sleeps.
* F10: a wrapper option's value (``sudo -u root``) was taken for the
  program, and a pipe into a group fed only the group's first command.
* F11: ``shell_view_target`` split the command on ``&&``/``|``/``;``
  inside quotes and ignored comments and subshells.
* F12: ``rewrite_guard`` missed ``sed -E -i`` and ``sed -e ... -i`` and
  refused a one-line ``sed -i`` followed by any ``\\n`` in the command.
* F15: ``output_shows_file`` read the whole file before its cap and
  searched only the head, so a ``tail`` of a big log never counted.
"""

from __future__ import annotations

import os
import subprocess
import tracemalloc
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
    output_shows_file,
    repo_read_key,
    rewrite_guard,
    shell_view_target,
)
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard
from kiss.tests.conftest import posix_only

SOURCE = "".join(f"line {i}\n" for i in range(1, 31))


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A work dir holding ``a.py``, ``notes.md`` and ``sub/log.txt``."""
    (tmp_path / "a.py").write_text(SOURCE)
    (tmp_path / "notes.md").write_text("old\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "log.txt").write_text("one\n")
    return tmp_path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "protocol.file.allow=always",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True,
    )


def _repo(path: Path, readme: str) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "README.md").write_text(readme)
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "init")
    return path


def test_files_of_two_submodules_have_their_own_read_keys(tmp_path: Path) -> None:
    """F7: reading ``sub-one/README.md`` does not make ``sub-two/README.md`` editable."""
    source = _repo(tmp_path / "source", "module readme\n")
    superproject = _repo(tmp_path / "super", "super readme\n")
    for name in ("sub-one", "sub-two"):
        _git(superproject, "submodule", "add", "-q", str(source), name)
    one = (superproject / "sub-one" / "README.md").resolve()
    two = (superproject / "sub-two" / "README.md").resolve()
    top = (superproject / "README.md").resolve()
    assert (superproject / "sub-one" / ".git").is_file()
    assert len({repo_read_key(one), repo_read_key(two), repo_read_key(top)}) == 3
    assert repo_read_key(one) == f"{one.parent}::README.md"
    tools = UsefulTools(work_dir=str(superproject))
    tools.Read(str(one))
    assert tools.Edit(str(two), "module readme", "changed").startswith("Error:")
    assert two.read_text() == "module readme\n"
    assert tools.Edit(str(top), "super readme", "changed").startswith("Error:")
    assert tools.Edit(str(one), "module readme", "changed").startswith("Successfully")


def test_a_worktree_still_shares_the_main_checkout_key(tmp_path: Path) -> None:
    """F7: the linked-worktree layout (``<main>/.git/worktrees/<name>``) keeps one key."""
    repo = _repo(tmp_path / "repo", "v = 1\n")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "side")
    assert repo_read_key((wt / "README.md").resolve()) == (
        f"{repo.resolve()}::README.md") == repo_read_key((repo / "README.md").resolve())


@posix_only("bash arithmetic and here-documents")
def test_arithmetic_shift_does_not_open_a_here_document(work: Path) -> None:
    """F9: what follows an arithmetic ``<<`` is still read as commands."""
    cwd, notes = str(work), work / "notes.md"
    refusal = HEREDOC_VERDICT.format(path=notes)
    for command in (
        "echo $((1<<2))\ncat > notes.md <<EOF\nx\nEOF",
        "x=$(( (1 + 2) << 3 )); cat > notes.md <<EOF\nx\nEOF",
        "((x = 1 << 3))\ncat > notes.md <<EOF\nx\nEOF",
        "for ((i = 0; i << 1; i++)); do cat > notes.md <<EOF\nx\nEOF\ndone",
        "cat <<EOF > notes.md\n$((1<<2))\nEOF",
    ):
        assert heredoc_guard(command, cwd, os.path.lexists) == refusal, command
    assert sleep_guard("y=$((1<<4))\nsleep 60") is not None
    # A $( ( ... ) ) that does not close as `))` is a command substitution.
    assert heredoc_guard("x=$((cat <<EOF > notes.md\nx\nEOF\n) )", cwd, os.path.lexists) == (
        refusal)
    # One that never closes (a syntax error for bash) is read the same way.
    assert heredoc_guard("x=$((cat <<EOF > notes.md\nx\nEOF\n", cwd, os.path.lexists) == refusal
    assert heredoc_guard("echo $((1 + 2)) > notes.md; ((n++))", cwd, os.path.lexists) is None
    tools = UsefulTools(work_dir=str(work))
    assert tools.Bash("echo $((1<<2))\ncat > notes.md <<EOF\nx\nEOF", "rewrite") == refusal
    assert notes.read_text() == "old\n"
    assert tools.Bash("echo $((1<<2)) $(( (2 + 3) * 4 ))", "arithmetic") == "4 20\n"


@posix_only("bash here-documents")
def test_wrapper_option_values_and_pipes_into_groups(work: Path) -> None:
    """F10: ``sudo -u root tee`` is tee; a group's ``cat`` copies the pipe it reads."""
    cwd, notes = str(work), work / "notes.md"
    refusal = HEREDOC_VERDICT.format(path=notes)
    for command in (
        "sudo -u root tee notes.md <<EOF\nx\nEOF",
        "timeout -s KILL 5 cat <<EOF > notes.md\nx\nEOF",
        "stdbuf -o L cat <<EOF > notes.md\nx\nEOF",
        "nice -n +5 cat <<EOF > notes.md\nx\nEOF",
        "cat <<EOF | (cd sub && cat > ../notes.md)\nx\nEOF",
        "cat <<EOF | { cd sub; cat > ../notes.md; }\nx\nEOF",
        "cat <<EOF | while true; do cat > notes.md; break; done\nx\nEOF",
        "cat <<EOF | (cat) > notes.md\nx\nEOF",
        "cat <<EOF | ( (cat) ) | tee notes.md\nx\nEOF",
    ):
        assert heredoc_guard(command, cwd, os.path.lexists) == refusal, command
    for command in (
        # A group command that is no copier writes output of its own.
        "cat <<EOF | (grep x) > notes.md\nx\nEOF",
        "cat <<EOF | { read l; echo \"$l\"; } > notes.md\nx\nEOF",
        # ...and one fed by another pipe inside the group does not read the heredoc.
        "cat <<EOF | (grep x | cat > notes.md)\nx\nEOF",
        # The shell opens a redirection before `env -C` moves the program:
        # this writes ../notes.md of the work dir, a new file.
        "env -C sub cat <<EOF > ../notes.md\nx\nEOF",
        # A pipe into an arithmetic command reaches no later command.
        "cat <<EOF | ((x)); cat > notes.md\nx\nEOF",
    ):
        assert heredoc_guard(command, cwd, os.path.lexists) is None, command
    assert heredoc_targets("cat <<EOF | (grep x | sort) > out.txt\nx\nEOF") == ["out.txt"]
    tools = UsefulTools(work_dir=str(work))
    out = tools.Bash("cat <<EOF | (cd sub && cat > ../notes.md)\nnew\nEOF", "rewrite")
    assert out == refusal
    assert notes.read_text() == "old\n"


@posix_only("the host Bash tool runs bash")
def test_shell_view_target_reads_the_command_as_bash_does(work: Path) -> None:
    """F11: quotes, comments and subshells do not hide the one file shown."""
    a, log = (work / "a.py").resolve(), (work / "sub" / "log.txt").resolve()
    cwd = str(work)
    assert shell_view_target("grep -n 'line 1 && x' a.py", cwd) == a
    assert shell_view_target("grep -n 'x | y; z' a.py", cwd) == a
    assert shell_view_target("cat a.py # the module", cwd) == a
    assert shell_view_target("(cd sub && cat log.txt)", cwd) == log
    assert shell_view_target("/bin/cat a.py", cwd) == a
    assert shell_view_target("cd sub; cat ../a.py", cwd) == a
    # Text, a here-document body and an unresolvable operand show no file.
    assert shell_view_target("echo 'cat a.py'", cwd) is None
    assert shell_view_target("cat > x.md <<'EOF'\ncat a.py\nEOF", cwd) is None
    assert shell_view_target('cat "$F"', cwd) is None
    assert shell_view_target("cat sub", cwd) is None
    tools = UsefulTools(work_dir=str(work))
    assert "line 1" in tools.Bash("grep -n 'line 1 && x\\|line 1' a.py", "view")
    assert tools.Edit(str(a), "line 2\n", "line two\n").startswith("Successfully")
    assert "one" in tools.Bash("(cd sub && cat log.txt) # look", "view")
    assert tools.Edit(str(log), "one", "two").startswith("Successfully")


def test_sed_in_place_is_judged_per_command() -> None:
    """F12: any in-place spelling with a newline escape in that sed's own words."""
    for command in (
        "sed -E -i 's/a\\nb/c/' README.md",
        "sed -e 's/a\\nb/c/' -i f",
        "sed -i.bak -e 's/a\\nb/c/' f",
        "sed --in-place 's/a\\nb/c/' f",
        "sed --in-place=.bak 's/a\\nb/c/' f",
        "/usr/bin/sed -ni 's/a\\nb/c/p' f",
        "cd d && sed -i \"s/a\\nb/c/\" f",
    ):
        refusal = rewrite_guard(command)
        assert refusal is not None and refusal.startswith("REFUSED: the `sed -i`"), command
    for command in (
        "sed -i 's/x/y/' f.txt; printf 'done\\n'",
        "sed 's/a\\nb/c/' f",
        "sed -e 's/a\\nb/c/' f > g",
        "cat > notes.md <<'EOF'\nsed -i 's/a\\nb/c/' f\nEOF",
        "echo \"sed -i 's/a\\nb/c/' f\"",
    ):
        assert rewrite_guard(command) is None, command


def test_output_shows_file_reads_only_the_ends_of_a_big_file(tmp_path: Path) -> None:
    """F15: a ``tail`` of a big file counts, and the read stays bounded."""
    big = tmp_path / "big.log"
    with big.open("w") as file:
        for n in range(350_000):
            file.write(f"filler line number {n:08d}\n")
        file.write("the very last line\n")
    assert big.stat().st_size > 8_000_000
    assert output_shows_file("the very last line\n", big)
    assert output_shows_file("filler line number 00000001\n", big)
    assert not output_shows_file("filler line number 00175000\n", big)  # the middle is not read
    sparse = tmp_path / "sparse.log"
    with sparse.open("wb") as file:
        file.write(b"first line\n")
        file.truncate(64_000_000)
        file.seek(0, os.SEEK_END)
        file.write(b"\nlast line\n")
    tracemalloc.start()
    try:
        assert output_shows_file("last line\n", sparse)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 40_000_000, peak


@posix_only("the host Bash tool runs bash")
def test_a_tail_of_a_big_file_counts_as_a_read(tmp_path: Path) -> None:
    """F15 end to end: ``tail -n 1`` of a 4 MB+ file lets the next Edit through."""
    big = tmp_path / "big.log"
    with big.open("w") as file:
        for n in range(150_000):
            file.write(f"filler line number {n:08d}\n")
        file.write("the very last line\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Bash("tail -n 1 big.log", "tail") == "the very last line\n"
    out = tools.Edit(str(big), "the very last line", "edited last line")
    assert out.startswith("Successfully"), out[:200]
    assert big.read_text().endswith("edited last line\n")
