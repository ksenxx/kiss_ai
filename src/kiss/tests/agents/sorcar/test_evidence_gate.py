# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests of the file-change readers of :mod:`kiss.agents.sorcar.evidence_gate`.

The coding gate and the coding SEA read what a shell command changes
(:func:`~kiss.agents.sorcar.evidence_gate.writes_files`,
:func:`~kiss.agents.sorcar.evidence_gate.file_changes`) to arm the gate on a
code file a command wrote, to count the changes since the agent's own passing
verifier run, and to re-review only after a change: a program counts in
command position only, a package or build tool writes through its mutating
subcommands only (``uv run pytest`` is a check, ``uv sync`` a write), and a
file under a ``tmp`` directory is the run's scratch, not a change of the work.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar import evidence_gate as eg
from kiss.tests.conftest import posix_only


def test_writing_commands_and_scratch_paths_are_recognised() -> None:
    for path in ("tmp/x", "./tmp/a.log", "/u/repo/tmp/probe/out", "sub/tmp/notes", "tmp",
                 "/var/folders/T/kiss-sub-ab12cd34/probe.py"):  # a sub-agent's private folder
        assert eg.is_scratch_path(path), path
    for path in ("tmpfile", "mytmp/x", "tmp/../a.py", "a.tmp", "src/a.py", "src/kiss-sub-x.py",
                 "/var/folders/T/kiss-sub-ab12cd34/../a.py"):
        assert not eg.is_scratch_path(path), path
    for command in ("echo x > f.txt", "cat a >> log", "tee out.txt", "cp a b", "mv a b",
                    "sed -i s/a/b/ f", "sed --in-place s/a/b/ f", "git apply p.diff",
                    "pip install x", "make -j8", "touch f", "cat <<EOF > f\nx\nEOF", "mkdir d",
                    "a -> b", "rm hello.py", "rm -rf build", "ls | xargs rm", "rmdir d",
                    "unlink f", "git rm -q a.py", "git mv a.py b.py", "git restore f",
                    "git clean -fd", "git switch main",
                    # a quoted '>' next to a real redirection, a quoted here-document body,
                    # an apostrophe in a comment, a nested shell's redirection
                    "grep '>' f > out.txt", "cat <<'EOF' > notes.txt\ndon't\nEOF",
                    "# don't change the input\nprintf x > result.txt",
                    'bash -c "rm -rf build"', 'bash -c "printf x > result.txt"',
                    "sh -ec 'echo a > b'", "timeout 60 bash -o pipefail -c 'make > log'",
                    # audit 2026-11-04 (S8): -c first in its cluster, then more options
                    "bash -ce 'mv a b'", "sh -ceu 'echo a > b'",
                    "bash -c \"echo it's\"; printf x > f", "/usr/bin/env bash -c 'echo a > b'",
                    "cd x && sh -c 'echo a > b'", "FOO=1 sudo -u me bash -lc 'echo a > b'",
                    "nohup nice -n 5 bash -c 'echo a > b' &", "ls | xargs -0 sh -c 'echo a > b'",
                    "timeout -k 5 60 bash -c 'echo a > b'", "timeout -v 1.5s bash -c 'echo a > b'",
                    "find . -name '*.py' -exec sh -c 'echo > {}' \\;",
                    "echo \\\\; bash -c 'echo a > b'",
                    # audit 2026-11-08 (S3/F5): a nested shell whose text runs no command (a
                    # comment, blanks) still truncates the file its redirection names
                    "bash -c '# no command' > src/main.py", "sh -c ' ' > a.py",
                    "bash -c '#x' >> a.py", "eval '# x' > a.py", "eval '' > a.py",
                    # S3/F6: ``--reference=RFILE`` supplies the mode or owner, so every
                    # operand is a file, the first too
                    "chmod --reference=tmp/ref src/main.py tmp/log",
                    "chmod --reference tmp/ref a.py", "chown --reference=tmp/ref a.py",
                    "chgrp --reference=tmp/ref a.py tmp/x", "chmod -R --reference=tmp/ref src",
                    # a program in command position after a wrapper, a keyword, in a
                    # group or a substitution; an in-place sed with the option anywhere;
                    # a here-document read by a program that writes
                    "sudo rm -rf build", "xargs -0 rm <list", "xargs -I {} rm {} <list",
                    "if cp a b; then echo ok; fi", "for f in *.o; do rm $f; done",
                    "x=$(rm -rf y)", "(cd pkg && touch x)", "{ mv a b; } 2>/dev/null",
                    "sed -n -i s/a/b/ f", "cmd &> out.txt", "tee f <<EOF\nx\nEOF",
                    "patch -p1 <<EOF\n--- a\nEOF", "patch -p1 < fix.diff",
                    "cp a.py tmp/x/ && cp tmp/x/a.py b.py", "chmod 755 bin/run",
                    "python -m pip install x", "mv tmp/x/a.py src/", "rm -rf tmp/x build",
                    # review 2026-11-03: a copy's destination named by -t, a scratch
                    # path that leaves tmp/ through .., a tool after a python -m
                    "cp -t src tmp/a.py", "cp --target-directory=src tmp/a.py",
                    "mv --target-directory src tmp/a.py", "install -t bin tmp/x",
                    "cp tmp/a.py tmp/../a.py", "touch tmp/x/../../a.py", "cp a tmp/..",
                    "mv -t tmp/x a.py", "python3 -m pip install x", "cp -tsrc tmp/a.py",
                    "cp -rt src tmp/a.py", "mv --target-directory src tmp/a.py tmp/b.py",
                    "cp -rtsrc/x tmp/a.py",
                    "sed -i -e s/a/b/ tmp/x/f src/a.py", "sed -i s/a/b/",
                    # review round 3: an attached or clustered script option must not
                    # make the first file the script
                    "sed -i -es/a/b/ src/a.py tmp/x/f", "sed -i -fscript.sed src/a.py tmp/x/f",
                    "sed -i -ne s/a/b/ src/a.py tmp/x/f", "sed -i --line-length 5 s/a/b/ src/a.py",
                    "sed -i -E -e s/a/b/ -e s/c/d/ a.py",
                    # audit 2026-11-10 (B/S4-F4): GNU sed takes a script option after its
                    # files, an abbreviated long one too, and only files after ``--``
                    "sed -i src/a.py tmp/x/f -e s/a/b/", "sed -i src/a.py tmp/x/f --expr s/a/b/",
                    "sed -i -e s/a/b/ tmp/x/f -- -a.py", "sed -i --exp=s/a/b/ src/a.py tmp/x/f",
                    "sed -i src/a.py --fi=x.sed tmp/x/f",
                    # ... and an abbreviated ``--in-place`` (real sed edits with all three)
                    "sed --in s/a/b/ src/a.py", "sed --i s/a/b/ f", "sed --in-pl=.bak s/a/b/ f",
                    # a tool's subcommand that is a tool, a launcher's program (full-suite
                    # sweep 2026-11-03: `uv pip install -e .` had passed only because `pip`
                    # matched anywhere)
                    "uv pip install -e .", "uv run pip install x", "uv run --with x pip install y",
                    "uv run python -m pip install x", "python -m uv pip install x",
                    "uv run --directory /tmp pip uninstall -y x", "uv run check --full --fix",
                    "uv run --no-sync --with x pip install y", "uv run --with=x pip install y",
                    "uv run --no-sync -p 3.12 --with x pip install y",
                    "uv --no-cache pip install x"):
        assert eg.writes_files({"command": command}), command
    for command in ("pytest -q", "ls -la", "cmd 2>&1 | head", "cmd > /dev/null", "cmd &> /dev/null",
                    "grep -n x f", "git status", "git diff", "docker run --rm img pytest",
                    "perform -x",
                    # a test or probe output saved under a scratch directory is a check
                    "pytest -q > tmp/out.txt 2>&1", "pytest -q >> ./tmp/4aa/log.txt",
                    "python3 probe.py > /home/u/repo/tmp/probe/result.json",
                    'cmd > "tmp/x y.txt"', "cmd > /tmp/out.txt", "cmd >tmp/out",
                    # a quoted, escaped or commented '>' / '<<' is text to the shell, not a
                    # redirection, in a nested shell's command too; a shell name that is
                    # an argument (echo bash -c ..., even after a wrapper) runs nothing
                    "python -c \"print(cfg.get('model', '<unset>'))\"",
                    "grep -n '<unset>' src/kiss/agents/sorcar/model_families.py",
                    'grep -c ">>" f.py', "grep -n '<<' f", "echo a \\> b",
                    "git commit -m 'rename a -> b'", "# note: output > file\nls",
                    "bash -c 'grep \"<unset>\" f'", "python -c 'print(1 > 0)'",
                    "echo bash -c 'a > b'", "bash -x; echo -c 'a > b'", "env echo bash -c 'a > b'",
                    "timeout 60 echo bash -c 'a > b'", "find . -exec echo bash -c 'a > b' \\;",
                    "docker exec c sh -c 'echo a > b'",
                    "echo \\;bash -c 'a > b'", "echo \\(bash -c 'a > b'",
                    # a program's name as an argument or in text is no command
                    # (install.sh, a .patch, a commit message, a python program)
                    "cat install.sh", "bash -n scripts/install.sh", "ls x.patch",
                    "python tests/test_install.py", 'git commit -m "fix(install): rm stale; cp"',
                    "python -c 'import shutil; cp = shutil.copy'", "docker run --rm img make",
                    "grep -rn 'tee|mkdir' src", "echo rm -rf build", "which rm",
                    "echo npm install", "echo git commit -m x", "git log pip install",
                    "echo -m npm install", "echo -m git commit -m x", "cp -rttmp/x a.py",
                    "uv run pytest -q", "uv run check --full", "echo uv pip install x",
                    "uv run echo pip install", "uv pip list", "uv run git log pip install",
                    # review round 4: a launcher option without a value (--no-sync) must not
                    # let the word after the program regain command position
                    "uv run --no-sync echo pip install", "uv run -q --locked pytest -q",
                    "uv run -p 3.12 pytest",
                    "cp -t tmp/x a.py b.py", "mv -t tmp/x tmp/a.py", "cp a.py ./tmp/../tmp/x/",
                    # scratch files are the run's own: made, copied, removed, saved under tmp/
                    "rm -rf tmp/9e43/parent", "mkdir -p tmp/x", "cp tmp/a.log tmp/b.log",
                    "cp -r logs/ tmp/x/", "verify.sh | tee tmp/x/log", "touch ./tmp/x/.keep",
                    "S=tmp/x; cp a $S/b", "mkdir -p tmp/x && T=tmp/x && mv tmp/old $T/new",
                    "cd /repo && S=tmp/x && gh api pages > $S/pages.json", "rm -rf /tmp/probe",
                    "chmod +x tmp/x/verify.sh", "dd if=/dev/zero of=tmp/x/blob bs=1k count=1",
                    # a mode that starts with a dash is chmod's first operand, not an option
                    "chmod -x tmp/x/verify.sh", "chmod -R -w tmp/x", "chmod -v -rw,o=r tmp/x/k",
                    "bash -c '# no command' > tmp/out", "sh -c ' ' > /dev/null",
                    "chmod --reference=tmp/ref tmp/log", "chown --reference tmp/ref tmp/log",
                    "chmod --reference=src/a.py tmp/log",  # the reference file is only read
                    # an in-place sed on scratch files (replay 2026-11-03, task 65132d04)
                    "sed -i s/a/b/ tmp/x/f", "sed -n -i.bak -e s/a/b/ --expression=p tmp/x/f tmp/y",
                    "sed -i -es/a/b/ tmp/x/f", "sed -nie s/a/b/ tmp/x/f", "sed -in s/a/b/ tmp/x/f",
                    "sed -i -l5 s/a/b/ tmp/x/f", "sed -i -E -e s/a/b/ -e s/c/d/ tmp/x/f",
                    "sed -i tmp/x/f -e s/a/b/", "sed -i --exp=s/a/b/ tmp/x/f -- tmp/y",
                    "sed --in s/a/b/ tmp/x/f",
                    "T=tmp/x && sed -i 's/^W = .*/W = 1/' $T/recompute.py && grep -n W $T/r.py",
                    # the command read as it was meant to run (audit 2026-11-03: a variable
                    # assigned after && used past the ; that ends its list)
                    "cd /w && T=tmp/x && tail -2 $T/a.log; cp $T/a.log ../../$T/",
                    "cd /w && S=tmp/x && echo '=== PAGES ==='; gh api pages > $S/pages.json",
                    "cd /w || exit 1; T=tmp/x || true; cp a $T/b",
                    # a here-document is input to its reader, not a write
                    "cat <<EOF\nx\nEOF", "python3 - <<'EOF'\nimport os; cp = 1\nprint(1 > 0)\nEOF",
                    "node - <<EOF\nrm = 1\nEOF", "sqlite3 db.sqlite <<EOF\nselect 1;\nEOF",
                    "python3 - <<EOF > tmp/out.json\nimport json\nEOF",
                    # a remote command's text is not this host's command
                    'ssh host "cd $DIR && git -C \\"my repo\\" pull"',
                    # a path is a scratch path in the directory the scanner's cd put the
                    # command in (held-out replay 2026-11-03, task f26100f4: a probe
                    # written into the scratch dir after a cd into it)
                    "mkdir -p /u/s10s/tmp/f2 && cd /u/s10s/tmp/f2 && cat > dbstats.py <<'EOF'\n"
                    "import x\nEOF",
                    "cd /u/s10s/tmp/f2 && cp a.py b.py && sed -i s/a/b/ c.py && "
                    "bash -c 'echo 1 > d.py'",
                    "cd /w/tmp/x && OUT=b.py && cp a $OUT",
                    "pushd /w/tmp/x && cp a b", "cd /w/tmp/x; pushd -n /w/src; cp a b",
                    # find's matches are under its start paths: scratch when they are
                    "cd /w/tmp/x && find . -name '*.pyc' -exec rm {} +",
                    "cd /w/tmp/x && find . -type f -execdir sh -c 'echo x > b.py' \\;",
                    "cd /w/tmp/x && find . -name '*.pyc' -delete",
                    "find tmp/x -name '*.pyc' -exec rm {} +"):
        assert not eg.writes_files({"command": command}), command
    # ...while a cd out of the scratch dir, one only the shell knows (a pushd that
    # rotates the stack, a ``cd "$D"``), a target the shell still expands, or find's
    # matches outside it keep the write (review round 6)
    for command in ("cd /u/s10s/tmp/f2 && cd /u/s10s && cat > dbstats.py <<'EOF'\nx\nEOF",
                    "cd /u/s10s/tmp/f2 && cd ../.. && cp a.py b.py", 'cd "$D" && cp a.py b.py',
                    "cd /u/s10s/tmp/f2 && cp a.py /u/s10s/b.py",
                    'cd /w/tmp/x && cp a "$OUT"', 'cd /w/tmp/x && echo x > "$OUT"',
                    'cd /w/tmp/x && sed -i s/a/b/ "$OUT"', "cd /w/tmp/x && cp a ~/b.py",
                    "cd /w/src; pushd -n /w/tmp/x; cp a b",
                    "pushd /w/src; pushd /w/tmp/x; pushd +1; cp a b",
                    "pushd /w/tmp/x; pushd; cp a b",
                    "cd /w/tmp/x && find /w/src -type f -exec rm {} +",
                    "cd /w/tmp/x && find /w/src -type f -execdir sh -c 'echo x > b.py' \\;",
                    "cd /w/tmp/x && find /w/src -name x -delete"):
        assert eg.writes_files({"command": command}), command
    # ...but not one into a file whose name only starts with tmp, or one tmp file among others
    for command in ("cmd > tmpfile", "cmd > mytmp/x", "cmd > tmp/a > out.txt",
                    "cmd > tmp/a; cmd > b"):
        assert eg.writes_files({"command": command}), command
    assert eg.writes_files({"commands": '["ls", "echo 1 > f"]'})
    assert not eg.writes_files({"commands": '["ls", "pwd"]'})
    assert eg.writes_files({"commands": "echo 1 > f"})  # not JSON: one command
    # not a list: the text itself, where a quoted word is text, not a command
    assert not eg.writes_files({"commands": '{"a": "rm f"}'})
    assert not eg.writes_files({"commands": '{"a": "echo 1 > f"}'})
    # the commands are the ones the tool runs, decoded as it decodes them (audit 2026-11-11,
    # N1 and N4): a lone backslash (grep's ``\|``) and a raw newline in a string are no
    # error, and text nested past the decoder's depth is one command like any other text
    lone = r'''["grep -c 'a\|b' notes.txt > src/app.py", "git checkout main"]'''
    assert eg.file_changes({"commands": lone}) == ["src/app.py", "git", "checkout", "main"]
    assert eg.file_changes({"commands": '["cat <<EOF > a.py\nx = 1\nEOF", "ls"]'}) == ["a.py"]
    assert eg.shell_commands({"commands": "[" * 100_000}) == ["[" * 100_000]
    # the files a call changes, as its commands name them (the program first)
    assert eg.file_changes({"command": "cp a.py tmp/x/; cp tmp/x/a.py b.py > log 2>&1"}) == [
        "log", "cp", "b.py"]
    assert eg.file_changes({"command": "make test; make"}) == ["make"]
    assert eg.file_changes({"command": "bash -c 'rm -rf build; mkdir out'"}) == [
        "rm", "build", "mkdir", "out"]
    # a nested shell's text that runs no command leaves the redirection this command's
    # (S3/F5); a reference option keeps every file operand (S3/F6)
    assert eg.file_changes({"command": "bash -c '# no command' > src/main.py"}) == [
        "src/main.py"]
    # ...while one that runs commands hands them its redirection, as a subshell's
    assert eg.file_changes({"command": "bash -c 'echo 1 > a; echo 2' > b"}) == ["a", "b", "b"]
    assert eg.file_changes({"command": "(echo 1 > a; echo 2) > b"}) == ["a", "b", "b"]
    assert eg.file_changes({"command": "chmod --reference=tmp/ref src/main.py tmp/log"}) == [
        "chmod", "src/main.py", "tmp/log"]
    assert eg.file_changes({"command": "chown --reference tmp/ref a.py"}) == ["chown", "a.py"]
    assert eg.file_changes({"command": "chmod 755 bin/run"}) == ["chmod", "bin/run"]
    # a mode that starts with a dash was dropped as an option and the file with it
    assert eg.file_changes({"command": "chmod -x bin/run"}) == ["chmod", "bin/run"]
    assert eg.file_changes({"command": "chmod -fR -w src"}) == ["chmod", "src"]
    # ``--`` ends a writing program's options: a file after it may start with a dash
    # (audit 2026-11-12: ``-dest.py`` was dropped, or its ``t`` read as ``-t``)
    assert eg.file_changes({"command": "cp -- tmp/source.py -dest.py"}) == ["cp", "-dest.py"]
    assert eg.file_changes({"command": "rm -f -- -dest.py"}) == ["rm", "-dest.py"]
    assert eg.file_changes({"command": "chmod -- -x -run.py"}) == ["chmod", "-run.py"]
    # a copy into a directory writes a file of each source's name there, which was not
    # listed: a directory by the words (``-t``, several sources, a trailing ``/``) or,
    # the words not telling, a destination without a suffix
    assert eg.file_changes({"command": "cp tmp/source.py src/"}) == [
        "cp", "src/", "src/source.py"]
    assert eg.file_changes({"command": "cp -t pkg.d tmp/a.py tmp/b.py"}) == [
        "cp", "pkg.d", "pkg.d/a.py", "pkg.d/b.py"]
    assert eg.file_changes({"command": "install tmp/a.py tmp/b.py pkg.d"}) == [
        "install", "pkg.d", "pkg.d/a.py", "pkg.d/b.py"]
    assert eg.file_changes({"command": "cp tmp/source.py src"}) == ["cp", "src", "src/source.py"]
    assert eg.file_changes({"command": "cp src/a.py src/a.py.orig"}) == ["cp", "src/a.py.orig"]
    # find's ``{}`` are its matches, under its start paths
    assert eg.file_changes({"command": "find . -name '*.pyc' -exec rm {} +"}) == ["rm", "."]
    assert eg.file_changes({"command": "find src lib -delete"}) == ["find", "src", "lib"]
    # every -exec clause counts, each read up to its own ``;`` / ``+`` (audit A06-11: the
    # first clause once swallowed the rest), and -execdir's directory is that clause's alone
    assert eg.file_changes({
        "command": "find data -type f -exec true {} \\; -exec touch changed.py \\;",
    }) == ["touch", "changed.py"]
    assert eg.file_changes({
        "command": "cd /w/tmp/x && find . -type f -execdir cat {} + -exec touch /w/src/x.py \\;",
    }) == ["touch", "/w/src/x.py"]
    # -delete does not hide a writing clause of the same find (audit 2026-11-07, A08
    # astra-10 / fable-6), and an -execdir command that may leave the scratch tree (an
    # operand with ``..``; a start whose own parent is not scratch, since the start itself
    # is a match and its command runs in that parent) is judged as written (astra-11)
    assert eg.file_changes({"command": "find tmp/x -delete -exec touch src/x.py \\;"}) == [
        "touch", "src/x.py"]
    assert eg.file_changes({"command": "find src -delete -exec touch tmp/x.py \\;"}) == [
        "find", "src"]
    assert eg.file_changes({
        "command": "cd /w && find tmp/a -execdir cp {} ../src/x.py \\;",
    }) == ["cp", "../src/x.py"]
    assert eg.file_changes({
        "command": "cd /w && find tmp -maxdepth 0 -execdir touch out.py \\;",
    }) == ["touch", "out.py"]
    assert eg.file_changes({"command": "cd /w && find tmp/a -execdir touch done.txt \\;"}) == []
    assert eg.file_changes({"command": "uv run check --full --fix"}) == ["uv", "run", "check"]
    # the command ``uv run`` launches is read as itself (A08 fable-7: only package
    # subcommands were), after uv's and run's options (valued or not)
    assert eg.file_changes({
        "command": "uv run --no-project --no-sync sed -i s/old/new/ source.py",
    }) == ["sed", "s/old/new/", "source.py"]
    # a script option after the files, and a file ``--`` protects, keep the files (B/S4-F4)
    assert eg.file_changes({"command": "sed -i source.py tmp/log -e s/old/new/g"}) == [
        "sed", "source.py", "tmp/log", "s/old/new/g"]
    assert eg.file_changes({"command": "sed -i -e s/a/b/ tmp/log -- -source.py"}) == [
        "sed", "s/a/b/", "tmp/log", "-source.py"]
    assert eg.file_changes({"command": "uv -q --directory /w run -p 3.12 cp a.py b.py"}) == [
        "cp", "b.py"]
    assert eg.file_changes({"command": "uv run bash -c 'echo 1 > f'"}) == ["f"]
    for command in ("uv run --with x cp a.py tmp/b.py", "uv run -q pytest -q", "uv run",
                    "uv --directory /w run --with"):
        assert eg.file_changes({"command": command}) == [], command
    assert eg.file_changes({"command": "(" * 20000 + "rm x"}) == []  # nested too deep to read
    assert not eg.writes_files({"commands": 7})
    assert not eg.writes_files({})


@posix_only("bash and chmod are the oracles")
def test_a_commandless_nested_shell_and_a_reference_mode_change_the_files_named(
    tmp_path: Path,
) -> None:
    """What the shell does is what the reader says: a ``bash -c '# comment' > f`` empties
    ``f``, and ``chmod --reference=RFILE`` changes every file operand.

    The reader once dropped the redirection of a nested shell whose
    text yields no command, and the first file operand of a ``chmod``
    with a reference option as if it were the mode, so the coding gate
    kept its cached verifier run current after the source was emptied or
    made non-executable (audit 2026-11-08, S3/F5 and F6).
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "tmp").mkdir()
    source = tmp_path / "src" / "main.py"
    for command in ("bash -c '# no command' > src/main.py", "sh -c ' ' > src/main.py",
                    "eval '# x' > src/main.py"):
        source.write_text("print(42)\n")
        subprocess.run(["bash", "-c", command], cwd=tmp_path, check=True)
        assert source.read_text() == "", command
        assert eg.file_changes({"command": command}) == ["src/main.py"], command
    source.write_text("#!/usr/bin/env python3\nprint(42)\n")
    source.chmod(0o755)
    (tmp_path / "tmp" / "ref").write_text("ref")
    (tmp_path / "tmp" / "ref").chmod(0o644)
    (tmp_path / "tmp" / "log").write_text("log")
    command = "chmod --reference=tmp/ref src/main.py tmp/log"
    done = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True)
    if done.returncode == 0:  # GNU chmod; the BSD one has no reference option
        assert stat.S_IMODE(source.stat().st_mode) == 0o644
    assert eg.file_changes({"command": command}) == ["chmod", "src/main.py", "tmp/log"]


#: Running the tests or a lint through a package or build tool is a check, not a
#: write (sweep 2026-10-30: the bare tool names once matched); review 1, R4
#: finding 2: a global option (``npm --prefix web run build``, ``uv --project .
#: sync``), a Cargo toolchain (``cargo +stable fmt``), a later Make target
#: (``make test build``) and ``--fix`` do not hide the mutating subcommand.
CHECKS = (
    "uv run pytest -q", "uv run check --full", "npm test", "npm run test", "yarn test",
    "pnpm test", "cargo test", "cargo clippy", "make test", "make check", "ninja test",
    "python -m pytest -q", "uv --project . run pytest -q", "npm --prefix src/kiss/webapp test",
    "cargo +stable test", "make test V=1", "make -k check", "make test check",
    "make test 2>&1 | tail -5", "make check && echo ok", "git -C repo diff", "pytest --fixtures",
    # review 2, Q2 finding 2: a quoted or escaped option value with blanks is one
    # word, so the subcommand after it is still found; a remote shell's command
    # string is text to this host's shell
    'npm --prefix "src/web app" test', "npm --prefix src/web\\ app run lint",
    'git -C "my repo" diff', "uv --project 'my project' run pytest -q",
    'cargo --manifest-path "a b/Cargo.toml" test', "bash -c \"npm --prefix 'my app' test\"",
    "git -C repo\ncheckout main", 'ssh host "cd $DIR && git -C \\"my repo\\" pull"',
)
WRITES = (
    "pip install x", "pip3 uninstall -y x", "uv pip install -e .", "uv add httpx", "uv sync",
    "uv lock", "uv build", "npm install", "npm ci", "npm i lodash", "npm run build",
    "yarn add left-pad", "pnpm remove x", "cargo build --release", "cargo install ripgrep",
    "cargo fmt", "apt-get install -y curl", "apt remove vim", "make", "make -j8",
    "make test && make install", "cmake -S . -B build", "ninja -C build",
    "npm --prefix src/kiss/webapp run build", "uv --project . sync", "cargo +stable fmt",
    "make test build", "npm run lint -- --fix", "uv run ruff check --fix .",
    "git -C repo checkout main", "pip --quiet install x", "make -k test install", "make V=1",
    'npm --prefix "src/web app" run build', "npm --prefix src/web\\ app run build",
    "yarn --cwd 'my app' add left-pad", 'pnpm --dir "my app" remove x',
    'uv --project "my project" sync', "uv --directory 'a b' add httpx",
    'git -C "my repo" checkout main', "git -C my\\ repo stash", "git -c 'user.name=A B' pull",
    'cargo --manifest-path "a b/Cargo.toml" build', "cargo --config 'a b' fmt",
    'pip --log "a b.log" uninstall -y x', 'apt-get -o "Debug::X=a b" purge vim',
    'apt -o "Debug::X=a b" upgrade', "git -C repo \\\n  checkout main",
    "bash -c \"npm --prefix 'my app' ci\"", "echo ok && sudo git -C 'my repo' reset --hard",
    "(cd x; npm --prefix \"a b\" i)", "x=$(git -C 'a b' rebase main)",
)


@pytest.mark.parametrize("command", CHECKS)
def test_a_package_or_build_tools_check_is_no_write(command: str) -> None:
    assert not eg.writes_files({"command": command})


@pytest.mark.parametrize("command", WRITES)
def test_a_package_or_build_tools_mutating_subcommand_is_a_write(command: str) -> None:
    assert eg.writes_files({"command": command})


# Runs writes_files on the commands read from stdin (JSON) while a thread ticks
# every 10 ms; prints the results, the time taken and the longest pause of
# the ticking thread (a regex that runs for seconds holds the GIL that long).
_PROBE = r"""
import json, sys, threading, time
from kiss.agents.sorcar.evidence_gate import writes_files

beats = []
done = threading.Event()


def beat():
    while not done.is_set():
        beats.append(time.monotonic())
        time.sleep(0.01)


thread = threading.Thread(target=beat)
thread.start()
commands = json.load(sys.stdin)
start = time.monotonic()
results = [writes_files({"command": command}) for command in commands]
elapsed = time.monotonic() - start
done.set()
thread.join()
gaps = [later - earlier for earlier, later in zip(beats, beats[1:])]
print(json.dumps({"results": results, "elapsed": elapsed, "max_gap": max(gaps, default=0.0)}))
"""

#: Review 2, Q2 finding 1: the global options before a subcommand were one
#: repeated regex in which a ``+a`` word was both a possible value of the
#: option before it and an option itself, so ``npm --prefix +a ... test`` had
#: 2**n readings: a 300-byte command held the GIL, and with it the whole
#: daemon, for seconds; a long run of options with tool names in it was
#: quadratic.  The nested-shell scan had the same shapes.
ADVERSARIAL = {
    "npm " + "--prefix +a " * 2000 + "test": False,
    "npm " + "--prefix +a " * 2000 + "ci": True,
    "npm -a " * 20000 + "test": False,
    "uv " + "-a pip " * 20000 + "test": False,
    "git " + '-c "a b" ' * 20000 + "status": False,
    "cargo " + "+a " * 20000 + "fmt": True,
    # the nested-shell scan: shells with options and no -c, shells with -c, quotes,
    # command starts with assignments or paths
    "bash -a " * 20000 + "status": False,
    "bash -c " * 20000 + "'x'": False,
    # bash stops at its first operand (the second ``bash``, a script file): the ``-c``
    # after it is that script's argument (audit 2026-11-08, S8: shell_script_index)
    "bash -a " * 20000 + "-c 'x > y'": False,
    "'" * 20000 + " > f": True,
    ";x=1" * 20000 + " > f": True,
    "(x=" * 20000: False,
    "(" * 20000 + "/x/" * 10000 + "sh": False,
    "sudo " * 20000 + "echo": False,
    "find x " * 10000 + "-exec sh -c 'x > y'": True,
}


def test_repeated_options_are_read_in_linear_time_without_holding_the_gil() -> None:
    src = str(Path(eg.__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    commands = list(ADVERSARIAL)
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], input=json.dumps(commands),
        capture_output=True, text=True, timeout=120, env=env, check=True,
    )
    report = json.loads(done.stdout)
    assert report["results"] == list(ADVERSARIAL.values())
    assert report["elapsed"] < 30, report
    assert report["max_gap"] < 2, report
    # short runs of the same shape still find the subcommand
    assert not eg.writes_files({"command": "npm " + "--prefix +a " * 25 + "test"})
    assert eg.writes_files({"command": "npm " + "--prefix +a " * 25 + "ci"})
    assert eg.writes_files({"command": "npm --prefix +a ci"})
    assert not eg.writes_files({"command": "npm --prefix +a test"})
