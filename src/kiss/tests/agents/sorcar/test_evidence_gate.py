# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests of the evidence gate (:mod:`kiss.agents.sorcar.evidence_gate`).

The gate refuses, once per task, a ``finish(success=True)`` whose summary
hedges or that follows a file change no shell command checked; the Sorcar
agent installs it in its tool-call guard (:meth:`SorcarAgent._guard_tool_call`)
for every run that has the edit tools.
"""

from __future__ import annotations

import tempfile

from kiss.agents.sorcar import evidence_gate as eg
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent


def test_hedges_and_writing_commands_are_recognised() -> None:
    assert eg.hedges_in("All checks pass; see the table.") == []
    assert eg.hedges_in("This is my best guess; the output is UNVERIFIED.") == [
        "best guess", "unverified"]
    # quoted and coded text is cited, not claimed: a UI label, a command, the
    # gate's own wording; an apostrophe opens no quotation; "not verified" is
    # a label (iOS, macOS profiles) and no hedge on its own
    assert eg.hedges_in('The profile shows "Not Verified" until signed; it is now verified.') == []
    assert eg.hedges_in("The gate refuses a summary saying 'could not verify' or `untested`.") == []
    assert eg.hedges_in("<p>Refused phrases: <code>should work</code>, "
                        "<pre>\nmy guess\n</pre></p>") == []
    assert eg.hedges_in("Pages marked &quot;untested&quot; are listed; "
                        "the agent’s run is “unverified”.") == []
    assert eg.hedges_in("I couldn't verify it; it's an educated guess.") == [
        "educated guess", "couldn't verify"]
    assert eg.hedges_in("The output is Not Verified, so I did not test it.") == ["did not test"]
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
                    "bash -c \"echo it's\"; printf x > f", "/usr/bin/env bash -c 'echo a > b'",
                    "cd x && sh -c 'echo a > b'", "FOO=1 sudo -u me bash -lc 'echo a > b'",
                    "nohup nice -n 5 bash -c 'echo a > b' &", "ls | xargs -0 sh -c 'echo a > b'",
                    "timeout -k 5 60 bash -c 'echo a > b'", "timeout -v 1.5s bash -c 'echo a > b'",
                    "find . -name '*.py' -exec sh -c 'echo > {}' \\;",
                    "echo \\\\; bash -c 'echo a > b'",
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
                    # an in-place sed on scratch files (replay 2026-11-03, task 65132d04)
                    "sed -i s/a/b/ tmp/x/f", "sed -n -i.bak -e s/a/b/ --expression=p tmp/x/f tmp/y",
                    "sed -i -es/a/b/ tmp/x/f", "sed -nie s/a/b/ tmp/x/f", "sed -in s/a/b/ tmp/x/f",
                    "sed -i -l5 s/a/b/ tmp/x/f", "sed -i -E -e s/a/b/ -e s/c/d/ tmp/x/f",
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
    # the files a call changes, as its commands name them (the program first)
    assert eg.file_changes({"command": "cp a.py tmp/x/; cp tmp/x/a.py b.py > log 2>&1"}) == [
        "log", "cp", "b.py"]
    assert eg.file_changes({"command": "make test; make"}) == ["make"]
    assert eg.file_changes({"command": "bash -c 'rm -rf build; mkdir out'"}) == [
        "rm", "build", "mkdir", "out"]
    # find's ``{}`` are its matches, under its start paths
    assert eg.file_changes({"command": "find . -name '*.pyc' -exec rm {} +"}) == ["rm", "."]
    assert eg.file_changes({"command": "find src lib -delete"}) == ["find", "src", "lib"]
    assert eg.file_changes({"command": "uv run check --full --fix"}) == ["uv", "run", "check"]
    assert eg.file_changes({"command": "(" * 20000 + "rm x"}) == []  # nested too deep to read
    assert not eg.writes_files({"commands": 7})
    assert not eg.writes_files({})


def test_gate_refuses_once_with_the_reason_then_passes() -> None:
    gate = eg.EvidenceGate(track_changes=True)
    done = {"success": True, "summary_in_html": "<p>Done, tests pass.</p>"}
    # nothing changed: a clean claim passes
    assert gate.verdict(done) is None and not gate.used
    # an edit with no check after it
    gate.observe("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"})
    assert gate.unchecked_change
    assert gate.verdict({"success": False, "summary_in_html": "gave up"}) is None
    assert gate.verdict({}) is None  # the implicit finish
    verdict = gate.verdict(done)
    assert verdict is not None and verdict.startswith("Not finished: finish(success=True)")
    assert "1. files were changed after the last shell command" in verdict
    assert gate.used
    # the second finish passes whatever it says
    assert gate.verdict(done) is None
    # a check after the change clears it; a writing command re-marks it
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Write", {"file_path": "a.py", "content": "x"})
    gate.observe("Bash", {"command": "pytest -q"})
    assert not gate.unchecked_change
    gate.observe("Bash", {"command": "echo fix >> a.py"})
    assert gate.unchecked_change
    gate.observe("run_commands_parallel", {"commands": '["pytest a", "pytest b"]'})
    assert not gate.unchecked_change
    gate.observe("Read", {"file_path": "a.py"})  # neither a change nor a check
    assert not gate.unchecked_change
    assert gate.verdict(done) is None
    # waiting on a background check reads its outcome: it clears the change; a tail does not
    gate.observe("Edit", {"file_path": "a.py"})
    gate.observe("bash_job", {"job_id": "j1", "action": "tail"})
    assert gate.unchecked_change
    gate.observe("bash_job", {"job_id": "j1", "action": "wait", "timeout_seconds": 600})
    assert not gate.unchecked_change
    # a hedged summary is refused even without a change; both reasons are listed together
    hedged = {"success": "true", "summary_in_html": "It should work but I could not test it."}
    verdict = gate.verdict(hedged)
    assert verdict is not None and '"could not test"' in verdict and '"should work"' in verdict
    assert "2." not in verdict
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {"file_path": "a.py"})
    verdict = gate.verdict(hedged)
    assert verdict is not None and "1. the summary itself says" in verdict
    assert "2. files were changed" in verdict
    # without change tracking (the interactive agent) only the hedge counts
    gate = eg.EvidenceGate()
    gate.observe("Edit", {"file_path": "a.py"})
    assert gate.unchecked_change and gate.verdict(done) is None
    verdict = gate.verdict(hedged)
    assert verdict is not None and "files were changed" not in verdict


def test_sorcar_agent_guard_installs_the_gate_for_runs_with_edit_tools() -> None:
    agent = ChatSorcarAgent("evidence")
    with tempfile.TemporaryDirectory() as work_dir:
        agent.work_dir = work_dir
        agent._begin_task(None)
        done = {"success": True, "summary_in_html": "Done."}
        hedged = {"success": True, "summary_in_html": "Done, but untested."}
        # a read-only run (a reviewer) is never gated, hedged or not
        agent._run_tool_names = frozenset({"Bash", "Read", "finish"})
        assert agent._guard_tool_call("Bash", {"command": "echo x > f"}) is None
        assert agent._guard_tool_call("finish", hedged) is None
        # a full run that writes prose: an unchecked write passes (the run
        # changed no code, so only the hedge check applies), a hedged
        # success is refused once
        agent._run_tool_names = frozenset({"Bash", "Read", "Edit", "Write", "finish"})
        write_args = {"file_path": f"{work_dir}/notes.md", "content": "x"}
        assert agent._guard_tool_call("Write", write_args) is None
        assert agent._coding_gate.evidence.unchecked_change
        assert not agent._coding_gate.armed
        assert agent._guard_tool_call("finish", done) is None
        refusal = agent._guard_tool_call("finish", hedged)
        assert refusal is not None and '"untested"' in refusal
        assert agent._guard_tool_call("finish", hedged) is None
        # a new task starts with a fresh gate
        agent._begin_task(None)
        assert not agent._coding_gate.evidence.used
        assert agent._guard_tool_call("finish", hedged) is not None
