"""End-to-end tests of the here-document rewrite guard of the shell tools.

``edit_guards.heredoc_guard`` (shared by the host tools, the Docker
tools and the coding SEA's container harness) refuses a ``cat``/``tee``
here-document that writes or appends to an existing file, or to a path
only the shell can resolve, and lets one that creates a new file run.
It reads the command with :mod:`kiss.agents.sorcar.shell_scan`.  The
tests drive the real ``UsefulTools.Bash`` / ``run_commands_parallel`` on
files in a temporary directory and check what is on disk afterwards;
commands that must never run (an append to ``~/.bashrc``) go to the
guard function itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
)
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash here-documents")


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A work dir holding ``notes.md``, ``my notes.md``, ``sub/log.txt`` and ``sub dir/x.md``."""
    (tmp_path / "notes.md").write_text("old\n")
    (tmp_path / "my notes.md").write_text("old\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "log.txt").write_text("one\n")
    (tmp_path / "sub dir").mkdir()
    (tmp_path / "sub dir" / "x.md").write_text("old\n")
    return tmp_path


def _refusal(path: Path | str) -> str:
    return HEREDOC_VERDICT.format(path=path)


def _unresolved(path: str) -> str:
    return HEREDOC_UNRESOLVED_VERDICT.format(path=path)


def test_heredoc_may_create_but_not_rewrite_a_file(work: Path) -> None:
    """A new target is written; an existing one is refused, whatever the syntax."""
    tools = UsefulTools(work_dir=str(work))
    out = tools.Bash("cat > new.txt <<'EOF'\nfresh\nEOF", "create")
    assert "REFUSED" not in out
    assert (work / "new.txt").read_text() == "fresh\n"
    notes = work / "notes.md"
    for command, target in (
        ("cat > notes.md <<'EOF'\nnew\nEOF", notes),
        ("cat >> notes.md <<'EOF'\nmore\nEOF", notes),
        ("cat 1>notes.md <<EOF\nnew\nEOF", notes),
        ("cat <<EOF >| notes.md\nnew\nEOF", notes),
        ("cat <<EOF &>> notes.md\nnew\nEOF", notes),
        ("cat <<EOF >& notes.md\nnew\nEOF", notes),
        ("cat > \"my notes.md\" <<EOF\nnew\nEOF", work / "my notes.md"),
        ("cat > my\\ notes.md <<EOF\nnew\nEOF", work / "my notes.md"),
        ("cat <<EOF | tee notes.md\nnew\nEOF", notes),
        ("cat <<EOF | tee -a notes.md > /dev/null\nmore\nEOF", notes),
        # every tee operand is a target, not only the first
        ("tee fresh.txt notes.md <<EOF\nnew\nEOF", notes),
        ("cat <<EOF | tee --append notes.md\nnew\nEOF", notes),
        # a line continuation joins the redirection to the heredoc line
        ("cat <<'EOF' \\\n> notes.md\nnew\nEOF", notes),
        # once written, the new file is an existing file too
        ("cat > new.txt <<'EOF'\nagain\nEOF", work / "new.txt"),
        # a heredoc copied inside a command substitution writes its file too
        ("x=$(cat <<'EOF' > notes.md\nnew\nEOF\n)", notes),
        ("x=`cat > notes.md <<EOF\nnew\nEOF\n`", notes),
        # the substitutions in an unquoted heredoc body run
        ("cat <<OUTER\n$(cat > notes.md <<'INNER'\nnew\nINNER\n)\nOUTER", notes),
        # a program named by its path, or behind wrappers, is the same program
        ("/bin/cat > notes.md <<EOF\nnew\nEOF", notes),
        ("timeout 5 /usr/bin/env tee notes.md <<EOF\nnew\nEOF", notes),
        # a redirection or a pipe after a group applies to the commands in it
        ("(cat <<EOF\nnew\nEOF\n) > notes.md", notes),
        ("{ cat <<EOF\nnew\nEOF\n} >> notes.md", notes),
        ("for i in 1; do cat <<EOF\nnew\nEOF\ndone > notes.md", notes),
        ("{ cat; } <<EOF > notes.md\nnew\nEOF", notes),
        ("f() { cat > notes.md <<EOF\nnew\nEOF\n}; f", notes),
        ("case a in a) cat > notes.md <<EOF\nnew\nEOF\n;; esac", notes),
        # a pipe at the end of the heredoc line goes on after the body
        ("cat <<EOF |\nnew\nEOF\ntee notes.md", notes),
    ):
        assert tools.Bash(command, "rewrite") == _refusal(target), command
    assert notes.read_text() == "old\n"
    assert (work / "my notes.md").read_text() == "old\n"
    assert (work / "new.txt").read_text() == "fresh\n"
    assert not (work / "fresh.txt").exists()  # nothing of a refused command ran


def test_cd_moves_where_relative_targets_resolve(work: Path) -> None:
    """``cd`` is followed in the shell that runs it; a subshell's ``cd`` ends with it."""
    tools = UsefulTools(work_dir=str(work))
    log = work / "sub" / "log.txt"
    assert tools.Bash("cd sub && cat >> log.txt <<'EOF'\ntwo\nEOF", "append") == _refusal(log)
    assert tools.Bash("cd sub\ncat >> log.txt <<'EOF'\ntwo\nEOF", "append") == _refusal(log)
    assert tools.Bash("{ cd sub; cat >> log.txt <<'EOF'\ntwo\nEOF\n}", "group") == (
        _refusal(log))
    assert tools.Bash(f"cd {work}/sub; cat > ../notes.md <<'EOF'\nx\nEOF", "rewrite") == (
        _refusal(work / "notes.md"))
    assert tools.Bash('cd "sub dir" && cat > x.md <<EOF\nnew\nEOF', "quoted") == (
        _refusal(work / "sub dir" / "x.md"))
    # A cd in a subshell, a pipeline or the background leaves this shell where it was.
    for command in ("(cd sub && pwd)\ncat > notes.md <<EOF\nnew\nEOF",
                    "cd sub | true; cat > notes.md <<EOF\nnew\nEOF",
                    "{ cd sub; } | true; cat > notes.md <<EOF\nnew\nEOF",
                    "cd sub && true & cat > notes.md <<EOF\nnew\nEOF",
                    # a group's redirection opens before its cd runs
                    "{ cd sub; cat <<EOF; } > notes.md\nnew\nEOF"):
        assert tools.Bash(command, "scoped") == _refusal(work / "notes.md"), command
    assert tools.Bash("x=$(cd sub)\ncat > notes.md <<EOF\nnew\nEOF", "substitution") == (
        _refusal(work / "notes.md"))
    # A cd in a pipeline stage holds for the rest of that stage; `|| exit` stops the script.
    assert tools.Bash("true | { cd sub; cat >> log.txt <<'EOF'\ntwo\nEOF\n}", "stage") == (
        _refusal(log))
    assert tools.Bash("cd sub || exit 1\ncat >> log.txt <<'EOF'\ntwo\nEOF", "or exit") == (
        _refusal(log))
    # Relative to the work dir, log.txt does not exist: the shell creates it.
    out = tools.Bash("cat > log.txt <<'EOF'\nroot log\nEOF", "create")
    assert "REFUSED" not in out and (work / "log.txt").read_text() == "root log\n"
    # After a cd only the shell can resolve, a relative target cannot be checked.
    for command in ('D=sub; cd "$D" && cat >> log.txt <<\'EOF\'\ntwo\nEOF',
                    "cd sub; cd -; cat >> log.txt <<'EOF'\ntwo\nEOF",
                    "pushd sub; popd; cat >> log.txt <<'EOF'\ntwo\nEOF",
                    # a cd that may not run: after ||, or in an if, a loop or a case
                    "cd sub || cd other; cat >> log.txt <<'EOF'\ntwo\nEOF",
                    "true ||\ncd sub; cat >> log.txt <<'EOF'\ntwo\nEOF",
                    "false || { true; cd sub; }; cat >> log.txt <<'EOF'\ntwo\nEOF",
                    "if true; then cd sub; fi; cat >> log.txt <<'EOF'\ntwo\nEOF"):
        assert tools.Bash(command, "unknown dir") == _unresolved("log.txt"), command
    # ...but an absolute one still can.
    out = tools.Bash(f'cd "$HOME"; cat >> {log} <<EOF\ntwo\nEOF', "absolute")
    assert out == _refusal(log)
    assert log.read_text() == "one\n"


def test_only_a_copy_of_the_heredoc_is_a_rewrite(work: Path) -> None:
    """Program output, quoted text, other descriptors and heredoc bodies are not rewrites."""
    tools = UsefulTools(work_dir=str(work))
    (work / "cat.json").write_text("{}\n")
    for command in (
        # an interpreter writes its own output, whatever the file is called
        "python3 - <<'EOF' > notes.md\nprint('generated')\nEOF",
        "python3 - <<'EOF' > cat.json; cat notes.md\nprint('generated')\nEOF",
        # a pipeline stops copying at its first other command
        "cat <<'EOF' | python3 - > cat.json\nprint('piped')\nEOF",
        # quoted text is not a command (and there is no heredoc here at all)
        "printf '%s\\n' 'cat <<EOF > notes.md'",
        # descriptor duplications, devices and standard error are not the copy
        "cat >&2 <<'EOF'\nw\nEOF\ncat 1>&2 <<'EOF'\nw\nEOF\ncat > /dev/stderr <<'EOF'\nw\nEOF",
        "cat <<'EOF' 2>> notes.md\nx\nEOF",
        # the classic commit-message substitution writes no file
        "echo \"$(cat <<'EOF'\nfix: a ) and a 'quote'\nEOF\n)\"",
        # a quoted heredoc's body is text, a substitution in it too
        "cat <<'OUTER'\n$(cat > notes.md <<'INNER'\nnew\nINNER\n)\nOUTER",
        "cat <<OUTER\n\\$(cat > notes.md <<'INNER'\nINNER\n)\nOUTER",
    ):
        assert "REFUSED" not in tools.Bash(command, "allowed"), command
    assert (work / "notes.md").read_text() == "generated\n"
    assert (work / "cat.json").read_text() == "piped\n"
    # A heredoc in a substitution is a word of the outer command, not what it pipes on.
    log = work / "log.md"
    log.write_text("old\n")
    assert "REFUSED" not in tools.Bash("git init -q", "init")
    out = tools.Bash(
        "git -c user.name=t -c user.email=t@example.com -c commit.gpgsign=false commit "
        "--allow-empty -m \"$(cat <<'EOF'\nfix: message\nEOF\n)\" | tee -a log.md", "commit")
    assert "REFUSED" not in out, out
    assert log.read_text().startswith("old\n") and "fix: message" in log.read_text()
    # Any delimiter ends its body, and a body line is text, never a command.
    out = tools.Bash("cat > script.sh <<'END-SCRIPT'\ncat <<X > notes.md\nX\nEND-SCRIPT",
                     "script")
    assert "REFUSED" not in out
    assert (work / "script.sh").read_text() == "cat <<X > notes.md\nX\n"
    out = tools.Bash("cat > tabs.sh <<-EOF.\n\tcat > notes.md <<X\n\tEOF.\necho done", "tabs")
    assert "REFUSED" not in out and "done" in out
    assert (work / "tabs.sh").read_text() == "cat > notes.md <<X\n"


def test_a_target_only_the_shell_resolves_is_refused(work: Path) -> None:
    """A variable, a command substitution or a glob may name an existing file."""
    tools = UsefulTools(work_dir=str(work))
    for command, target in (
        ('OUT=x; cat > "$OUT" <<EOF\nx\nEOF', "$OUT"),
        ('cat > "`pwd`/notes.md" <<EOF\nx\nEOF', "`pwd`/notes.md"),
        ('cat > "$(pwd)/notes.md" <<EOF\nx\nEOF', "$(pwd)/notes.md"),
        ("cat > ${OUT:-notes}.md <<EOF\nx\nEOF", "${OUT:-notes}.md"),
        ("cat > *.md <<EOF\nx\nEOF", "*.md"),
        ("cat <<EOF > >(cat > notes.md)\nx\nEOF", ">(cat > notes.md)"),
    ):
        assert tools.Bash(command, "dynamic") == _unresolved(target), command
    assert sorted(path.name for path in work.iterdir()) == [
        "my notes.md", "notes.md", "sub", "sub dir"]
    assert (work / "notes.md").read_text() == "old\n"


def test_parallel_and_background_commands_are_guarded(work: Path) -> None:
    """``run_commands_parallel`` and ``Bash(background=True)`` refuse the same rewrite."""
    tools = UsefulTools(work_dir=str(work))
    report = tools.run_commands_parallel(
        '["cat > made.txt <<EOF\\nm\\nEOF", "cat >> notes.md <<EOF\\nm\\nEOF"]',
    )
    assert report.startswith("2 commands: 1 succeeded, 1 failed"), report
    assert _refusal(work / "notes.md") in report
    assert (work / "made.txt").read_text() == "m\n"
    out = tools.Bash("cat >> notes.md <<EOF\nm\nEOF", "bg", background=True)
    assert out == _refusal(work / "notes.md")
    assert (work / "notes.md").read_text() == "old\n"


def test_guard_reads_commands_as_bash_does(tmp_path: Path) -> None:
    """Commands that must not run go to the guard itself; odd syntax is read as bash reads it."""
    cwd, notes = str(tmp_path), tmp_path / "notes.md"
    notes.write_text("old\n")
    (tmp_path / "sub").mkdir()
    refused = {
        "cat >> ~/.bashrc <<'EOF'\nx\nEOF": _unresolved("~/.bashrc"),
        "cat > $'notes.md' <<EOF\nx\nEOF": _unresolved("$'notes.md'"),
        "cat <<EOF | sudo -E tee notes.md\nx\nEOF": _refusal(notes),
        "A=1 env B=2 tee notes.md <<EOF\nx\nEOF": _refusal(notes),
        "if true; then cat > notes.md <<EOF\nx\nEOF\nfi": _refusal(notes),
        "cat 0<<EOF > notes.md\nx\nEOF": _refusal(notes),
        'cat > "no\\"tes.md" <<EOF\nx\nEOF\ncat > "no\\\ntes.md" <<EOF\nx\nEOF': _refusal(notes),
        "cd -P sub && cat > ../notes.md <<EOF\nx\nEOF": _refusal(notes),
        "cd && cat > notes.md <<EOF\nx\nEOF": _unresolved("notes.md"),
        "cat > notes.md <<EOF\nno terminator": _refusal(notes),
        "(cat <<EOF) | tee notes.md\nx\nEOF": _refusal(notes),
    }
    for command, verdict in refused.items():
        assert heredoc_guard(command, cwd, os.path.lexists) == verdict, command
    for command in (
        "ls",
        "cat <<< 'x' > notes.md",  # a here-string is no here-document
        "# cat > notes.md <<EOF\nls # cat > notes.md <<EOF",
        "cat 3<<EOF > notes.md\nx\nEOF",  # the heredoc is not cat's input
        "cat <<EOF 3<> notes.md >&-\nx\nEOF",
        "<<EOF > notes.md\nx\nEOF",  # no command copies the heredoc
        "diff <(cat <<EOF\na\nEOF\n) notes.md",
        "echo a ) b; echo cost $ 5 'unterminated",
        'echo "a\\b `date` unterminated',
        "echo $((1 + 2)) \\",
    ):
        assert heredoc_guard(command, cwd, os.path.lexists) is None, command
    # The coding SEA's ledger gate counts every heredoc write, program output too.
    assert heredoc_targets(
        "python3 - <<'EOF' > out.json\nprint(1)\nEOF\ncat <<EOF | tee -a a.txt\nx\nEOF\n"
        "cat <<EOF | sed s/a/b/ > 'b c.txt'\nx\nEOF\necho > not-a-heredoc.txt",
    ) == ["out.json", "a.txt", "b c.txt"]
