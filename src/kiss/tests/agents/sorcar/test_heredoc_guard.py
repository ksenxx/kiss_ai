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

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    heredoc_guard,
    heredoc_targets,
)
from kiss.agents.sorcar.shell_scan import TooDeepError, simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import host_sed_is_gnu, posix_only

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
        # piped into the text a shell or ``eval`` runs, the copy reaches what the commands of
        # that text write, or pass on (audit 2026-11-10 S10-F1, the direction left open then)
        ("cat <<EOF | bash -c 'tee notes.md' > /dev/null\nnew\nEOF", notes),
        ("cat <<EOF | eval 'tee notes.md' > /dev/null\nnew\nEOF", notes),
        ("cat <<EOF | sh -c 'cat' | tee notes.md\nnew\nEOF", notes),
        # the operator spelt across two quoted strings of that text: the command as written
        # holds no ``<<`` (audit 2026-11-21 B7-O2: the host tool has no gate on the text)
        ("bash -c 'cat > notes.md <''<EOF\nnew\nEOF'", notes),
        ("eval 'cat > notes.md <''<EOF\nnew\nEOF'", notes),
        # that text as an ANSI-C string, which bash decodes before the shell or ``eval`` reads
        # it (audit 2026-11-22 LH3: the scan took it for a value only the shell knows)
        ("bash -c $'cat > notes.md <<EOF\\nnew\\nEOF'", notes),
        ("eval $'cat > notes.md <<EOF\\nnew\\nEOF'", notes),
        ("X=$'cat > notes.md <<EOF\\nnew\\nEOF'; eval \"$X\"", notes),
    ):
        # Since 2026-10-31 a single copy into a file not shown names the Read
        # and Write to make after the verdict; the file is not touched.
        assert tools.Bash(command, "rewrite").startswith(_refusal(target)), command
    assert notes.read_text() == "old\n"
    assert (work / "my notes.md").read_text() == "old\n"
    assert (work / "new.txt").read_text() == "fresh\n"
    assert not (work / "fresh.txt").exists()  # nothing of a refused command ran
    out = tools.Bash("cat <<EOF | bash -c 'tee piped.txt' > /dev/null\nfresh\nEOF", "create")
    assert "REFUSED" not in out and (work / "piped.txt").read_text() == "fresh\n"
    out = tools.Bash("bash -c $'cat > decoded.txt <<EOF\\nfresh\\nEOF'", "create")
    assert "REFUSED" not in out and (work / "decoded.txt").read_text() == "fresh\n"


@pytest.mark.parametrize("redirect", [
    "<\\\n<'EOF'", "<<\\\n'EOF'", "<\\\n<\\\n-'EOF'", "<<-E\\\nOF",
    "<<'EOF' >\\\n>",
])
def test_continued_heredoc_tokens_cannot_hide_a_rewrite(work: Path, redirect: str) -> None:
    """Bash joins physical lines before reading operators and delimiter words (LH4/O1)."""
    command = (f"cat {redirect} notes.md\nnew\nEOF" if redirect.endswith(">")
               else f"cat > notes.md {redirect}\n\tnew\nEOF")
    result = UsefulTools(work_dir=str(work)).Bash(command, "continued rewrite")
    assert result.startswith("REFUSED"), (command, result)
    assert (work / "notes.md").read_text() == "old\n"


def test_continued_tokens_keep_literal_data_and_new_files(work: Path) -> None:
    """Continued operators are tokens; quoted backslash-newlines and bodies stay data."""
    tools = UsefulTools(work_dir=str(work))
    for name, command, expected in (
        ("doc", "cat > doc <\\\n<'EOF'\na\\\nb\nEOF", "a\\\nb\n"),
        ("expanded", "LH4=fresh; cat > expanded <<E\\\nOF\n$LH4\nEOF", "fresh\n"),
        ("tabs", "cat > tabs <\\\n<\\\n-\\\n 'EOF'\n\tfresh\nEOF", "fresh\n"),
        ("string", "cat > string <\\\n<\\\n< 'fresh'", "fresh\n"),
        ("append", "cat >\\\n> append <<'EOF'\nfresh\nEOF", "fresh\n"),
        ("fd", "cat 1\\\n> fd <<'EOF'\nfresh\nEOF", "fresh\n"),
        ("and", "true &\\\n& cat > and <<'EOF'\nfresh\nEOF", "fresh\n"),
        ("stream", "cat <\\\n(echo fresh) > stream", "fresh\n"),
        ("stdin", "cat > stdin < <\\\n(echo fresh)", "fresh\n"),
        ("quoted", "printf %s '<\\\n<' > quoted", "<\\\n<"),
    ):
        result = tools.Bash(command, "continued creation")
        assert "REFUSED" not in result, (command, result)
        assert (work / name).read_text() == expected, command
    # A script file's contents are outside T2; unrelated literal data is still data.
    (work / "harmless.sh").write_text(":\n")
    # Hidden script-looking words are ordinary data in a known non-evaluating command.
    for command in ("printf %s 'cat > notes.md <<EOF\nnew\nEOF'",
                    "bash harmless.sh; printf %s 'cat > notes.md <<EOF\nnew\nEOF'",
                    "printf '%s\\n' bash 'cat > notes.md <<EOF\nnew\nEOF' | grep bash",
                    "X=('cat > notes.md <<EOF\nnew\nEOF'); printf %s \"${X[0]}\""):
        assert "REFUSED" not in tools.Bash(command, "quoted data"), command
    assert (work / "notes.md").read_text() == "old\n"


@pytest.mark.parametrize("ansi", [False, True])
@pytest.mark.parametrize("form", [
    "declare X={}; eval \"$X\"", "readonly X={}; eval \"$X\"",
    "typeset X={}; eval \"$X\"", "X={} eval 'eval \"$X\"'",
    "X=({}); eval \"${{X[0]}}\"", "X=({}); eval \"$X\"",
    "set -- {}; eval \"$1\"", "bash -c 'eval \"$0\"' {}",
    "bash -c 'eval \"$1\"' _ {}", ": ${{X:={}}}; eval \"$X\"",
    "X=${{Y:-{}}}; eval \"$X\"", "for X in {}; do eval \"$X\"; done",
    "find . -maxdepth 0 -exec sh -c {} \\;", "setsid -w bash -c {}",
    "source /dev/stdin <<< {}", ". /dev/stdin <<< {}", "bash <<< {}", "bash -s <<< {}",
    "echo {} | bash", "printf '%s\\n' {} | sh", "cat <<< {} | bash",
    "bash < <(echo {})", "X={}; Y=$X; eval \"$Y\"", "X={}; eval \"${{X:-}}\"",
    "X={}; eval \"${{X%z}}\"", "X=a; X+={}; eval \"${{X#a}}\"",
    "printf -v X %s {}; eval \"$X\"", "read -r -d '' X <<< {}; eval \"$X\"",
    "IFS= read -r -d '' X < <(printf %s {}); eval \"$X\"",
])
def test_unread_shell_text_cannot_hide_a_heredoc(work: Path, form: str, ansi: bool) -> None:
    """All 29 LH3 forms contain the executed text, inside T2; unread values fail closed."""
    text = ("$'cat > notes.md <<EOF\\nnew\\nEOF'" if ansi
            else "'cat > notes.md <<EOF\nnew\nEOF'")
    command = "unset X Y; " + form.format(text)
    result = UsefulTools(work_dir=str(work)).Bash(command, "indirect rewrite")
    assert result.startswith("REFUSED"), (command, result)
    assert (work / "notes.md").read_text() == "old\n"


@pytest.mark.parametrize("command", [
    'echo "$\\\n(cat > notes.md <<EOF\nnew\nEOF\n)"',
    "eval $\\\n'cat > notes.md <<EOF\\nnew\\nEOF'",
    "find . -maxdepth 0 -exec env sh -c 'cat > notes.md <<EOF\nnew\nEOF' \\;",
    "flock lockfile setsid -w bash -c 'cat > notes.md <<EOF\nnew\nEOF'",
    "printf 'cat > notes.md <<EOF\\nnew\\nEOF\\n' | bash",
    "printf -v X 'cat > notes.md <<EOF\\nnew\\nEOF\\n'; eval \"$X\"",
    "X='cat > notes.md <'; X+='<EOF\nnew\nEOF'; eval \"$X\"",
    "X=('cat > notes.md <''<EOF\nnew\nEOF'); eval \"$X\"",
])
def test_unread_script_fragments_fail_closed(work: Path, command: str) -> None:
    """LH4 review: continued expansion openers, nested runners and assembled script text."""
    assert heredoc_guard(command, str(work), os.path.lexists) is not None, command
    result = UsefulTools(work_dir=str(work)).Bash(command, "hidden script fragments")
    assert result.startswith("REFUSED"), (command, result)
    assert (work / "notes.md").read_text() == "old\n"


def test_long_quoted_data_does_not_trigger_quadratic_hidden_text_search(work: Path) -> None:
    """A run of '<' without a newline is data, not 60,000 failed regex suffix searches."""
    command = "printf %s '" + "<" * 60_000 + "' > long-data"
    start = time.monotonic()
    assert heredoc_guard(command, str(work), os.path.lexists) is None
    assert time.monotonic() - start < 0.1
    assert "REFUSED" not in UsefulTools(work_dir=str(work)).Bash(command, "long literal data")
    assert (work / "long-data").read_text() == "<" * 60_000


@pytest.mark.parametrize("command", [
    # a backslash-newline is gone before bash reads a word, an operator or a name (J)
    "i\\\nf false; then cd sub; fi; cat > notes.md <<EOF\nnew\nEOF",
    "false &&\\\n\ncd sub\ncat > notes.md <<EOF\nnew\nEOF",
    "function f\\\n { cat > notes.md; }; f <<EOF\nnew\nEOF",
    "T=notes.md; cat > $\\\n{T} <<EOF\nnew\nEOF",
    "eval 'cat > notes.md <\\\n<EOF\nnew\nEOF'",
    # one ``<`` in each of two pieces, escapes that spell one, a function's or a pipe's text (H)
    "A='<'; printf -v X 'cat > notes.md %s%sEOF\\nnew\\nEOF\\n' \"$A\" \"$A\"; eval \"$X\"",
    "X=$(printf 'cat > notes.md \\74\\x3cEOF\\nnew\\nEOF\\n'); eval \"$X\"",
    "f() { eval \"$1\"; }; f 'cat > notes.md <<EOF\nnew\nEOF'",
    "echo 'cat > notes.md <<EOF\nnew\nEOF' | { IFS= read -r -d '' l; eval \"$l\"; }",
    "awk 'BEGIN{print \"cat > notes.md <<EOF\\nnew\\nEOF\"}' | bash",
    "source <(sed 's/^/cat > notes.md <<EOF\\nnew\\nEOF/' <<< '')",
    # a runner that puts its input into the script of the shell it runs
    "echo '<' | xargs -I@ sh -c 'cat > notes.md @@EOF\nnew\nEOF\n'",
    # the script of a shell that runners or an expansion name is read as one
    "K=sh; env -u NOPE setsid -w nohup sh -c '\"$0\" -c \"cat > notes.md <<EOF\nnew\nEOF\n\"' $K",
])
def test_joined_lines_and_text_for_an_unread_shell_fail_closed(work: Path, command: str) -> None:
    """LH4b: lines are joined before any token is judged; a ``<`` that may reach a shell
    the scan did not read is refused however the text is put together."""
    result = UsefulTools(work_dir=str(work)).Bash(command, "joined or assembled rewrite")
    assert result.startswith("REFUSED"), (command, result)
    assert (work / "notes.md").read_text() == "old\n"


def test_kept_continuations_and_plain_data_beside_an_unread_shell_run(work: Path) -> None:
    """LH4b: bash keeps a backslash-newline in quotes, comments and quoted bodies; a ``<``
    in the operands of a program is data, whatever else the command runs."""
    tools = UsefulTools(work_dir=str(work))
    (work / "harmless.sh").write_text(":\n")
    for name, command, expected in (
        ("tabsq", "cat > tabsq <<-'EOF'\n\ta\\\n\tb\n\tEOF", "a\\\nb\n"),
        ("open", "cat > open <<'EOF'\na\\\nb", "a\\\nb\n"),
        ("later", "cat > later <<'EOF'\nx\nEOF\necho a \\\n >> later", "x\na\n"),
        ("comment", "echo one > comment # c \\\necho two >> comment", "one\ntwo\n"),
        ("ansi", "printf %s $'a\\\nb' > ansi", "a\\\nb"),
        ("plain", "unset Y; eval \"$Y\"; echo '<b>' > plain", "<b>\n"),
        ("count", "n=$(grep -c '<old' notes.md); echo $((n+1)) > count", "1\n"),
        ("group", "unset Y; { echo '<a>'; echo '<b>'; } > group; eval \"$Y\"", "<a>\n<b>\n"),
        ("fed", "unset Y; X=1 <<< '<a>'; eval \"$Y\"; echo ok > fed", "ok\n"),
        ("sourced", ". ./harmless.sh; grep -c '<' notes.md > sourced; eval \"$Y\"", "0\n"),
        # a script the scan can read is judged by what it does, under runners too: the
        # shape of the script that starts a background job in a container
        ("nested", "setsid -w sh -c 'tr a-z A-Z < notes.md > nested'; eval \"$Y\"", "OLD\n"),
        ("job", "env -u NOPE setsid -w nohup sh -c 'k=bash; command -v bash >/dev/null || k=sh; "
                "\"$k\" -c '\\''tr a-z A-Z < notes.md > job'\\'' < /dev/null' < /dev/null",
         "OLD\n"),
    ):
        assert heredoc_guard(command, str(work), os.path.lexists) is None, command
        if "setsid " in command and shutil.which("setsid") is None:
            continue  # util-linux's: stock macOS has none to run the script with
        result = tools.Bash(command, "data")
        assert "REFUSED" not in result, (command, result)
        assert (work / name).read_text() == expected, command
    assert (work / "notes.md").read_text() == "old\n"


def test_more_kept_continuations_than_the_scan_restarts_for_are_refused(work: Path) -> None:
    """LH4b: the scan starts over once per quoted string holding a backslash-newline, and
    refuses the command past its limit; a gate reads what it found.

    (Through ``run_commands_parallel``: ``Bash`` does not catch the scan's error for a
    command the destructive guard does not read, reported as LH4b-N1.)"""
    tools = UsefulTools(work_dir=str(work))
    command = "echo " + " ".join(["'a\\\nb'"] * 70) + " > many"
    with pytest.raises(TooDeepError, match="backslash-newline"):
        heredoc_guard(command, str(work), os.path.lexists)
    report = tools.run_commands_parallel(json.dumps([command]))
    assert "1 failed" in report and "backslash-newline" in report, report
    assert not (work / "many").exists()
    assert simple_commands(command, str(work), optimistic=True)
    few = "echo " + " ".join(["'a\\\nb'"] * 3) + " > few"
    assert "REFUSED" not in tools.Bash(few, "kept pairs")
    assert (work / "few").read_text() == " ".join(["a\\\nb"] * 3) + "\n"


def test_cd_moves_where_relative_targets_resolve(work: Path) -> None:
    """``cd`` is followed in the shell that runs it; a subshell's ``cd`` ends with it."""
    tools = UsefulTools(work_dir=str(work))
    log = work / "sub" / "log.txt"
    # Since 2026-10-31 a ``cd DIR &&`` copy is read as the copy itself (not shown: the
    # refusal names the Read and Write to make); a ``cd`` on its own line is not.
    assert tools.Bash("cd sub && cat >> log.txt <<'EOF'\ntwo\nEOF", "append").startswith(
        _refusal(log))
    assert tools.Bash("cd sub\ncat >> log.txt <<'EOF'\ntwo\nEOF", "append") == _refusal(log)
    assert tools.Bash("{ cd sub; cat >> log.txt <<'EOF'\ntwo\nEOF\n}", "group") == (
        _refusal(log))
    # the ``..`` is the filesystem's to resolve (audit 2026-11-10 S10-F6): the path keeps it
    assert tools.Bash(f"cd {work}/sub; cat > ../notes.md <<'EOF'\nx\nEOF", "rewrite") == (
        _refusal(f"{work}/sub/../notes.md"))
    assert tools.Bash('cd "sub dir" && cat > x.md <<EOF\nnew\nEOF', "quoted").startswith(
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
    # A variable the command assigned a literal value is followed into its cd...
    out = tools.Bash('D=sub; cd "$D" && cat >> log.txt <<\'EOF\'\ntwo\nEOF', "assigned")
    assert out == _refusal(log)
    # ...but after a cd only the shell can resolve, a relative target cannot be checked.
    for command in ('cd "$D" && cat >> log.txt <<\'EOF\'\ntwo\nEOF',
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
        ('cat > "$OUT" <<EOF\nx\nEOF', "$OUT"),  # the environment's, not the command's
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
    # A variable the command assigns names its file: an existing one is refused as
    # such, a new one is created.
    assert tools.Bash('OUT=notes.md; cat > "$OUT" <<EOF\nx\nEOF', "assigned") == (
        _refusal(work / "notes.md"))
    assert "REFUSED" not in tools.Bash('OUT=x; cat > "$OUT" <<EOF\nx\nEOF', "assigned")
    assert (work / "x").read_text() == "x\n"


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
    assert out.startswith(_refusal(work / "notes.md"))
    assert (work / "notes.md").read_text() == "old\n"


def test_an_exec_sink_is_the_file_its_shell_opened_where_it_opened_it(work: Path) -> None:
    """Audit 2026-11-08 S8/F1-F2: ``exec > file`` opens the file in the directory of the
    ``exec``, in the shell that runs it: a later ``cd`` does not move it, an ``exec`` in a
    pipeline stage or in the background is a subshell's, and one that may not run (after
    ``&&``, in an ``if``, in a function body) leaves the earlier file possible."""
    tools = UsefulTools(work_dir=str(work))
    notes = work / "notes.md"
    for command, verdict in (
        # the file opened in the work dir, whatever directory the copy runs in
        ("exec > notes.md; cd sub; cat <<'EOF'\nnew\nEOF", _refusal(notes)),
        ("cd sub; exec > ../notes.md; cd ..; cat <<'EOF'\nnew\nEOF",
         _refusal(f"{work}/sub/../notes.md")),
        # a sink opened in a directory only the shell knows stays unknown after a known cd
        (f'cd "$D"; exec > notes.md; cd {work}/sub; cat <<\'EOF\'\nnew\nEOF',
         _unresolved("notes.md")),
        # an exec in a pipeline stage or in the background is a subshell's
        ("exec > notes.md; exec > sub/log2 | cat; cat <<'EOF'\nnew\nEOF", _refusal(notes)),
        ("exec > notes.md; exec > sub/log2 & cat <<'EOF'\nnew\nEOF", _refusal(notes)),
        # an exec that may not run leaves the earlier file possible
        ("exec > notes.md; false && exec > sub/log2; cat <<'EOF'\nnew\nEOF", _refusal(notes)),
        ("exec > notes.md; if false; then exec > sub/log2; fi; cat <<'EOF'\nnew\nEOF",
         _refusal(notes)),
        ("exec > notes.md; f() { exec > sub/log2; }; cat <<'EOF'\nnew\nEOF", _refusal(notes)),
    ):
        assert tools.Bash(command, "exec") == verdict, command
    assert notes.read_text() == "old\n"
    assert not (work / "sub" / "log2").exists()
    # A new file is created where the exec opened it; an exec that may not run
    # into a new file refuses nothing.
    out = tools.Bash("exec > made.log; cd sub; cat <<'EOF'\nnew\nEOF", "create")
    assert "REFUSED" not in out and (work / "made.log").read_text() == "new\n"
    out = tools.Bash("[ -n \"$NOPE\" ] && exec > sub/log2; cat <<'EOF'\nshown\nEOF", "maybe")
    assert "REFUSED" not in out and "shown" in out and not (work / "sub" / "log2").exists()


def test_a_child_shell_is_read_with_its_own_variables_and_options(work: Path) -> None:
    """Audit 2026-11-08 S8/F3-F4: a child shell's ``-c`` text sees the parent's environment
    (its exported variables, as the assignments before the command set them, none after
    ``env -i``), not its unexported variables, and the shell reads its options to their end
    (``--``, a later ``-e``, ``-co posix``, a valued ``-o`` / ``--rcfile``) before the text.
    (FX45: after ``-o posix`` bash runs the text otherwise, an assignment before ``:`` stays,
    so only the shell knows the file.)
    Review 2026-11-08 R7 S8/F3: an ``export`` that may not run (after ``&&`` once its list
    ends, after ``||``, in an ``if``) leaves the child's value unknown, as does an ``export
    -n`` that may not run, and ``unset`` drops the export attribute with the value."""
    tools = UsefulTools(work_dir=str(work))
    notes = work / "notes.md"
    for command, verdict in (
        ("T=sub/; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'", _unresolved("${T}notes.md")),
        ("export T=sub/; T= bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'", _refusal(notes)),
        ("export T=sub/; env -i bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("export T=sub/; export -n T; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("T=sub/; false && export T; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("T=sub/; if false; then export T; fi; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("T=sub/; true || export T; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("export T=sub/; unset T; T=sub/; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("export T=sub/; false && export -n T; bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("T=sub/; (export T); bash -c 'cat > \"${T}notes.md\" <<EOF\nnew\nEOF'",
         _unresolved("${T}notes.md")),
        ("bash -c -- 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("bash -c -e 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("bash -co posix 'cat > notes.md <<EOF\nnew\nEOF'", _unresolved("notes.md")),
        ("bash -o pipefail -c 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("bash --rcfile /dev/null -c 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("bash -rcfile /dev/null -c 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("sh -c -- 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
        ("eval 'cat > notes.md <<EOF\nnew\nEOF'", _refusal(notes)),
    ):
        assert tools.Bash(command, "child") == verdict, command
    assert notes.read_text() == "old\n"
    # The words after the text are the child's arguments; an exported variable is the
    # child's too; eval runs in this shell, which knows its own variables.
    out = tools.Bash("bash -ec 'cat > made.txt <<EOF\nnew\nEOF' -x", "arguments")
    assert "REFUSED" not in out and (work / "made.txt").read_text() == "new\n"
    out = tools.Bash("T=sub/; export T; bash -c 'cat > \"${T}made.txt\" <<EOF\nnew\nEOF'", "export")
    assert "REFUSED" not in out and (work / "sub" / "made.txt").read_text() == "new\n"
    # An export after ``&&`` holds for the rest of its list, as an assignment does; an
    # export that may not run adds nothing to a variable exported before it.
    out = tools.Bash("T=sub/ && export T && bash -c 'cat > \"${T}made3.txt\" <<EOF\nnew\nEOF'",
                     "and-list export")
    assert "REFUSED" not in out and (work / "sub" / "made3.txt").read_text() == "new\n"
    out = tools.Bash("export T=sub/; false && export T; bash -c 'cat > \"${T}made4.txt\" <<EOF\n"
                     "new\nEOF'", "exported twice")
    assert "REFUSED" not in out and (work / "sub" / "made4.txt").read_text() == "new\n"
    out = tools.Bash("T=sub/; eval 'cat > \"${T}made2.txt\" <<EOF\nnew\nEOF'", "eval")
    assert "REFUSED" not in out and (work / "sub" / "made2.txt").read_text() == "new\n"


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
        # a physical cd ends where the symbolic links lead (audit 2026-11-10 S10-F7)
        "cd -P sub && cat > ../notes.md <<EOF\nx\nEOF": _unresolved("../notes.md"),
        "cd && cat > notes.md <<EOF\nx\nEOF": _unresolved("notes.md"),
        "cat > notes.md <<EOF\nno terminator": _refusal(notes),
        "(cat <<EOF) | tee notes.md\nx\nEOF": _refusal(notes),
        # the substitutions of an array value run
        "values=($(cat > notes.md <<'EOF'\nx\nEOF\n))": _refusal(notes),
        'values=(a "$(cat > notes.md <<EOF\nx\nEOF\n)" b)': _refusal(notes),
        # an unquoted body joins a line ending in a backslash with the next
        # before the delimiter is looked for: the first heredoc ends at ``\``, ``EOF``
        "cat <<EOF\n\\\nEOF\ncat > notes.md <<'END'\nx\nEND": _refusal(notes),
        # a delimiter is not expanded: ``<<$D`` ends at a ``$D`` line, and is unquoted
        "D=EOF; cat > sub/new <<$D\nbody\n$D\ncat > notes.md <<'END'\nx\nEND": _refusal(notes),
        "D=EOF; cat <<$D\n$(cat > notes.md <<X\nx\nX\n)\n$D": _refusal(notes),
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
        # joined, ``x\`` and ``EOF`` are no delimiter: the rest is body
        "cat <<EOF\nx\\\nEOF\ncat > notes.md <<'END'\nx\nEND",
        "D=EOF; cat <<$D\nx\nEOF\ncat > notes.md <<'END'\nx\nEND",
    ):
        assert heredoc_guard(command, cwd, os.path.lexists) is None, command
    # The coding SEA's ledger gate counts every heredoc write, program output too.
    assert heredoc_targets(
        "python3 - <<'EOF' > out.json\nprint(1)\nEOF\ncat <<EOF | tee -a a.txt\nx\nEOF\n"
        "cat <<EOF | sed s/a/b/ > 'b c.txt'\nx\nEOF\necho > not-a-heredoc.txt",
    ) == ["out.json", "a.txt", "b c.txt"]


def _tree(root: Path) -> dict[str, str]:
    """Every file under *root* with its text, by relative path."""
    return {str(path.relative_to(root)): path.read_text()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_rewrite_after_a_cd_is_judged_where_the_inherited_cdpath_leads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-21 B7-O1: the host tool's shell inherits this process's ``CDPATH``; with
    ``sub`` there bash takes ``cd d`` to ``sub/d``.  The conversions and the rewrite guard
    read that ``cd`` without it, so ``cd d && cat > victim <<EOF`` ran as the creation of
    ``d/victim`` and overwrote ``sub/d/victim``, was applied as a ``Write``/``Edit`` of
    ``d/victim`` where that existed, and a multi-line ``sed -i`` or a python rewrite of
    ``sub/d/victim`` passed as one of a file that is not there.  Bash is the oracle."""
    python = ("python3 - <<'EOF'\ns = open('victim').read()\n"
              "open('victim', 'w').write(s.replace('ORIGINAL', 'owned'))\nEOF")
    monkeypatch.setenv("CDPATH", "sub")
    for number, (command, both, refusal) in enumerate((
        ("cd d && cat > victim <<'EOF'\nowned\nEOF", False, _unresolved("victim")),
        ("cd d && cat >> victim <<'EOF'\nowned\nEOF", True, _unresolved("victim")),
        ("cd d && tee victim <<'EOF'\nowned\nEOF", True, _unresolved("victim")),
        ("cd d && sed -i ':a;N;$!ba;s/ORIGINAL\\nsecond/owned/' victim", False,
         "REFUSED: the `sed -i` call did NOT run"),
        (f"cd d && {python}", False, "REFUSED: the python heredoc did NOT run"),
        # A one-substitution ``sed -i`` is no ``Edit`` of ``d/victim`` here, and what bash
        # does not look up in ``CDPATH`` is where the text says: both run as written.
        ("cd d && sed -i 's/ORIGINAL/owned/' victim", True, None),
        ("cd ./d && cat > fresh.txt <<'EOF'\nowned\nEOF", True, None),
        ("CDPATH= cd d && cat > fresh.txt <<'EOF'\nowned\nEOF", True, None),
    )):
        if "sed -i" in command and not host_sed_is_gnu():
            continue  # BSD sed takes the script for the backup suffix
        oracle, work = tmp_path / f"bash{number}", tmp_path / f"tool{number}"
        for root in (oracle, work):
            (root / "sub" / "d").mkdir(parents=True)
            (root / "d").mkdir()
            (root / "sub" / "d" / "victim").write_text("ORIGINAL\nsecond\n")
            if both:  # the file the text spells exists too
                (root / "d" / "victim").write_text("ORIGINAL\nsecond\n")
        seeded = _tree(work)
        subprocess.run(["bash", "-c", command], cwd=oracle, stdin=subprocess.DEVNULL,
                       capture_output=True, check=False, timeout=30)
        tools = UsefulTools(work_dir=str(work))
        for victim in sorted(work.rglob("victim")):
            tools.Read(str(victim))  # shown: a rewrite read as one of it would be applied
        out = tools.Bash(command, "rewrite")
        if refusal is None:
            assert _tree(work) == _tree(oracle) != seeded, (command, out)
        else:
            assert out.startswith(refusal), (command, out)
            assert _tree(work) == seeded != _tree(oracle), command


#: Audit 2026-11-21 B7-O3: copies of the here-document into ``victim`` through a descriptor
#: the command itself opened on the file, or through the name of one (``/dev/stdin``).
_DESCRIPTOR_COPIES = (
    "cat 3>>victim >&3 <<'EOF'\nowned\nEOF\n",
    "cat 3<>victim >&3 <<'EOF'\nowned\nEOF\n",
    "cat <>victim >&0 <<'EOF'\nowned\nEOF\n",
    "cat 3>victim 1>&3- <<'EOF'\nowned\nEOF\n",
    "cat 3>victim 4>&3 >&4 <<'EOF'\nowned\nEOF\n",
    "cat 2>victim >&2 <<'EOF'\nowned\nEOF\n",
    "tee 3>victim >&3 <<'EOF'\nowned\nEOF\n",
    "{ cat; } 3>victim >&3 <<'EOF'\nowned\nEOF\n",
    "cat <<'EOF' | cat 3>victim >&3\nowned\nEOF\n",
    "cat <(cat <<'EOF'\nowned\nEOF\n) 3>victim >&3\n",
    "cat <>victim >/dev/stdin <<'EOF'\nowned\nEOF\n",
)


def _bash_copies(victim: Path, command: str) -> bool:
    """Whether bash, run in the directory of *victim*, copies ``owned`` into that file."""
    victim.write_text("ORIGINAL\n")
    subprocess.run(["bash", "-c", command], cwd=victim.parent, stdin=subprocess.DEVNULL,
                   capture_output=True, check=False, timeout=30)
    return "owned" in victim.read_text()


def test_a_copy_through_a_descriptor_into_a_file_only_the_shell_knows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-21 B7-O3: ``cat 3>>victim >&3 <<EOF`` copies the text into ``victim``
    through descriptor 3.  The guard named the file only where it could resolve it: in a
    directory only the shell knows (an exec bridge without one, after ``cd "$D"``), or
    opened by a word only the shell expands (``3>"$X"``), the copy passed; so did one
    through ``/dev/stdin``, the name of descriptor 0, in any directory.  Bash is the oracle."""
    victim, cwd = tmp_path / "victim", str(tmp_path)
    monkeypatch.setenv("X", str(victim))
    monkeypatch.setenv("D", cwd)
    for command in _DESCRIPTOR_COPIES:
        assert _bash_copies(victim, command), command
        assert heredoc_guard(command, None, bool) == _unresolved("victim"), command
        assert heredoc_guard(command, cwd, os.path.lexists) == _refusal(victim), command
    # The coding SEA's ledger gate counts that write too.
    assert heredoc_targets(_DESCRIPTOR_COPIES[0]) == ["victim"]
    # Through the tool, whose shell inherits ``X`` and ``D``: nothing ran.
    tools = UsefulTools(work_dir=cwd)
    for command, verdict in (
        ("cat 3>\"$X\" >&3 <<'EOF'\nowned\nEOF", _unresolved("$X")),
        ("tee 3>\"$X\" >&3 <<'EOF'\nowned\nEOF", _unresolved("$X")),
        ("cat 2>\"$X\" >&2 <<'EOF'\nowned\nEOF", _unresolved("$X")),
        ("cd \"$D\" && cat 3>>victim >&3 <<'EOF'\nowned\nEOF", _unresolved("victim")),
        ("cat <>victim >/dev/stdin <<'EOF'\nowned\nEOF", _refusal(victim)),
    ):
        assert _bash_copies(victim, command), command
        victim.write_text("ORIGINAL\n")
        assert tools.Bash(command, "copy") == verdict, command
        assert victim.read_text() == "ORIGINAL\n", command
    # No copy of the text: the descriptor is opened and not written, or the output ends
    # elsewhere.
    for command in ("cat 3>\"$X\" <<'EOF'\nowned\nEOF",
                    "cat 3>\"$X\" >&3 >/dev/null <<'EOF'\nowned\nEOF",
                    "cat >&2 3>\"$X\" <<'EOF'\nowned\nEOF"):
        assert not _bash_copies(victim, command), command
        assert heredoc_guard(command, cwd, os.path.lexists) is None, command


def test_a_bare_cd_is_pinned_whatever_cdpath_the_developer_exports(tmp_path: Path) -> None:
    """Leftover H1 (audit 2026-11-21): the guards judge a bare ``cd`` by the ``CDPATH`` the
    tool's shell inherits, so a pytest started from a shell that exports one failed the tests
    that pin where such a ``cd`` leads.  ``conftest.py`` drops it for the session: the test
    of ``cd`` above passes in a nested run started with ``..`` exported."""
    tmp = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{__file__}::test_cd_moves_where_relative_targets_resolve"],
        env=dict(os.environ, TMPDIR=tmp, TMP=tmp, TEMP=tmp, CDPATH=".."),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
