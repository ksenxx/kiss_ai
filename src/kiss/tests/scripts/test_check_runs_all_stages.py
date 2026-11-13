# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.check.run_checks``.

A 24-hour audit of ``history.db`` (2026-09-21) found ``uv run check
--full`` executed 65 times across 30 tasks: the script stopped at the
first failing stage, so an agent fixed ruff, re-ran, fixed mypy, re-ran,
fixed pyright, re-ran — one full model step and about a minute per
stage.  ``run_checks`` now runs every stage and returns all failures.

The stages here are real commands (``true``, ``false``, ``sh -c
'touch …'``, ``uv sync`` on a directory without a project), so the
observed behavior is the real subprocess path.  ``main()`` itself runs
only with ``--clean-only``, as a copy of the script in a scratch
project: its checks are the repository's whole ruff/mypy/pyright pass,
which takes minutes and depends on the state of the checkout.

Not reached here: ``run_command`` without the POSIX leash (``os.name !=
"posix"``), where a stage is a plain child killed alone.  Only a faked
platform would take that path here, and the tests of the leash need a
FIFO, ``fork`` and ``SIGKILL``, so they are skipped where it is taken.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.scripts import check
from kiss.scripts.check import (
    MAX_DIGEST_LINES,
    error_lines,
    run_checks,
    tree_fingerprint,
    verdict_text,
)
from kiss.tests.conftest import posix_only

pytestmark = pytest.mark.skipif(
    shutil.which("sh") is None, reason="needs a POSIX shell"
)


def _touch_stage(marker: Path, description: str) -> tuple[list[str], str]:
    return (["sh", "-c", f"touch '{marker}'"], description)


def test_all_stages_run_after_a_failure(tmp_path: Path) -> None:
    """A failing stage does not stop the later stages from running."""
    marker = tmp_path / "later-stage-ran"
    failed = run_checks([
        (["false"], "Lint code (ruff)"),
        _touch_stage(marker, "Type check (mypy)"),
        (["false"], "Type check (pyright)"),
    ])
    assert [name for name, _ in failed] == ["Lint code (ruff)", "Type check (pyright)"]
    assert marker.exists(), "the stage after the first failure must still run"


def test_all_passing_returns_empty(tmp_path: Path) -> None:
    """No failures gives an empty list."""
    marker = tmp_path / "ran"
    assert run_checks([(["true"], "a"), _touch_stage(marker, "b")]) == []
    assert marker.exists()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_failed_uv_sync_stops_the_run(tmp_path: Path) -> None:
    """``uv sync`` is the prerequisite: when it fails nothing else runs."""
    marker = tmp_path / "must-not-run"
    # A pyproject.toml uv cannot parse: with none at all uv walks up from
    # ``--project`` and, under a ``--basetemp`` inside this repository, finds
    # this project and syncs it for real (into the shared environment when
    # the stages of a worktree run there), re-pointing its editable install.
    broken_project = tmp_path / "broken-project"
    broken_project.mkdir()
    (broken_project / "pyproject.toml").write_text("[project\n")
    failed = run_checks([
        (["uv", "sync", "--project", str(broken_project)], "Install dependencies (uv sync)"),
        _touch_stage(marker, "Lint code (ruff)"),
    ])
    assert [name for name, _ in failed] == ["Install dependencies (uv sync)"]
    assert failed[0][1], "the uv error line must be in the digest"
    assert not marker.exists()


def test_a_stage_runs_at_the_project_root_wherever_check_was_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit SC1: the stages name ``src/`` relatively and ran in the caller's directory.

    Started below the root every stage failed on a tree that was fine, and ``--fix`` kept
    that verdict under the fingerprint of the root, for the next run there to replay.
    """
    monkeypatch.chdir(tmp_path)
    stage = 'echo "x.py:1: error: ran in $(pwd -P)"; exit 1'
    failed = run_checks([(["sh", "-c", stage], "where")], quiet=True)
    assert failed == [("where", [f"x.py:1: error: ran in {check.PROJECT_ROOT.resolve()}"])]


def test_failed_stage_digest_repeats_its_error_lines() -> None:
    """The digest of a failed stage keeps only the lines that locate an error.

    An agent that pipes ``check --full`` through ``tail`` must still see
    every error, so ``main`` reprints these lines after the stage list.
    """
    output = (
        "Checking 12 files\n"
        "E501 Line too long (108 > 100)\n"
        "   --> src/kiss/x.py:99:101\n"
        "    |\n"
        "src/kiss/y.py:12: error: Incompatible return value  [return-value]\n"
        "  /abs/z.py:5:7 - error: Object of type None is not subscriptable\n"
        "Found 3 errors.\n"
        "All done\n"
    )
    failed = run_checks([
        (["sh", "-c", f"printf '%s' '{output}' >&2; exit 1"], "Lint code (ruff)"),
    ])
    assert [name for name, _ in failed] == ["Lint code (ruff)"]
    assert failed[0][1] == [
        "E501 Line too long (108 > 100)",
        "   --> src/kiss/x.py:99:101",
        "src/kiss/y.py:12: error: Incompatible return value  [return-value]",
        "  /abs/z.py:5:7 - error: Object of type None is not subscriptable",
        "Found 3 errors.",
    ]


def test_digest_is_capped() -> None:
    """A stage with hundreds of errors is cut to MAX_DIGEST_LINES plus a count."""
    lines = error_lines(f"a.py:{i}: error: boom\n" for i in range(MAX_DIGEST_LINES + 7))
    assert len(lines) == MAX_DIGEST_LINES + 1
    assert lines[-1] == "... 7 more"
    assert error_lines(["all good\n", "nothing here\n"]) == []


def test_quiet_stage_prints_one_status_line(capsys: pytest.CaptureFixture[str]) -> None:
    """``--fix`` mode keeps a stage's output off the console but in the digest."""
    failed = run_checks(
        [(["sh", "-c", "echo noise; echo 'a.py:1: error: bad'; exit 1"], "lint")], quiet=True,
    )
    out = capsys.readouterr().out
    assert "noise" not in out and "Running:" not in out
    assert out.strip() == "❌ lint failed with exit code 1"
    assert failed == [("lint", ["a.py:1: error: bad"])]


@pytest.fixture
def held(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    """A FIFO and its read end: a read there is EOF only once no process holds the write end."""
    path = tmp_path / "held"
    os.mkfifo(path)
    fifo = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    yield path, fifo
    os.close(fifo)


def _released(fifo: int, seconds: float = 30.0) -> bool:
    """Whether every process holding the FIFO's write end is gone within *seconds*."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if not os.read(fifo, 1):
                return True
        except BlockingIOError:  # held, and nothing written
            pass
        time.sleep(0.05)
    return False


leashed = posix_only("a stage's leash: its process group, a FIFO to see who is left of it")


@leashed
def test_the_leash_of_a_stage_does_not_show_in_the_stage() -> None:
    """A missing command still raises (the leash's ``sh`` alone would make it exit code 127),
    and the stage has an empty stdin and no child it did not start: it can wait for them all."""
    with pytest.raises(FileNotFoundError, match="no-such-command-w1b18"):
        check.run_command(["no-such-command-w1b18"], "missing", quiet=True)
    stage = (
        "import os, sys\n"
        "assert sys.stdin.read() == ''\n"
        "try:\n"
        "    while True:\n"
        "        os.wait()\n"
        "except ChildProcessError:\n"
        "    pass\n"
    )
    assert check.run_command([sys.executable, "-c", stage], "reaps", quiet=True, timeout=30) is None


@leashed
@pytest.mark.parametrize(
    "stall", ['sleep 60 & echo "$1"; wait', 'echo "$1"; exec >&- 2>&-; sleep 60'],
    ids=["output-held", "output-closed"],
)
def test_a_stalled_stage_is_killed_at_its_deadline_with_what_it_started(
    held: tuple[Path, int], stall: str,
) -> None:
    """Audit B/X1-F4: a stage that never exited held the later stages and the verdict for good."""
    path, fifo = held
    line = "a.py:1: error: before the stall"
    digest = check.run_command(
        ["sh", "-c", f'exec 3>"$0"; {stall}', str(path), line], "stalled", quiet=True, timeout=3)
    assert digest == [line, "timed out: still running after 3 s, killed"]
    assert _released(fifo), "the stage or its child outlived the deadline"


@leashed
def test_a_stage_is_over_when_its_process_exits_whoever_holds_its_output(
    held: tuple[Path, int], tmp_path: Path,
) -> None:
    """Audit B/X1-F4: a child that kept the stage's stdout held ``check`` until it exited too."""
    path, fifo = held
    stage = 'exec 3>"$0"; sleep 60 & echo "b.py:2: error: the stage is done"; exit 3'
    started = time.monotonic()
    digest = check.run_command(["sh", "-c", stage, str(path)], "left a child", quiet=True)
    assert digest == ["b.py:2: error: the stage is done"]
    assert time.monotonic() - started < 30, "waited for the child, which sleeps 60 s"
    assert _released(fifo), "the child the stage left behind was not killed"
    # A child in a session of its own escapes that kill: its pipe is given up after a grace,
    # and closed at the child's next line, which nobody is left to take.
    alive, cut_off = tmp_path / "escapee-alive", tmp_path / "escapee-cut-off"
    alive.touch()
    escapee = (
        "import os, sys, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    try:\n"
        "        for _ in range(1200):\n"
        "            if not os.path.exists(sys.argv[1]):\n"
        "                break\n"
        "            print('still here', flush=True)\n"
        "            time.sleep(0.05)\n"
        "    except BrokenPipeError:\n"
        "        os.rename(sys.argv[1], sys.argv[2])\n"
        "    os._exit(0)\n"
        "print('c.py:3: error: the stage is done', flush=True)\n"
        "sys.exit(3)\n"
    )
    started = time.monotonic()
    try:
        digest = check.run_command(
            [sys.executable, "-c", escapee, str(alive), str(cut_off)], "left an escapee",
            quiet=True)
        assert digest == ["c.py:3: error: the stage is done"]
        assert time.monotonic() - started < 30, "waited for the escaped child, which lives 60 s"
        while alive.exists() and time.monotonic() - started < 60:
            time.sleep(0.05)
        assert cut_off.exists(), "the escaped child still writes into a pipe nobody reads"
    finally:
        alive.unlink(missing_ok=True)


@leashed
@pytest.mark.parametrize("how", ["SIGINT", "SIGKILL"], ids=["interrupted", "killed"])
def test_a_stage_dies_with_the_check_that_runs_it(held: tuple[Path, int], how: str) -> None:
    """Audit B/X1-F4: interrupted or killed, ``check`` left its stage running."""
    path, fifo = held
    code = (  # a test run in the background of a shell hands its children an ignored SIGINT
        "import signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        f"sys.path.insert(0, {str(Path(check.__file__).parents[2])!r})\n"
        "from kiss.scripts.check import run_command\n"
        """run_command(['sh', '-c', 'exec 3>"$0"; echo up; sleep 60', sys.argv[1]], 'stage')\n"""
    )
    with subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    ) as runner:
        try:
            assert runner.stdout is not None
            assert "up\n" in iter(runner.stdout.readline, ""), "the stage did not start"
            runner.send_signal(getattr(signal, how))
            runner.wait(timeout=60)
            assert _released(fifo), "the stage outlived the check that ran it"
        finally:
            runner.kill()


def test_verdict_text_lists_failed_stages_and_their_errors() -> None:
    assert verdict_text([]).strip().splitlines()[1] == "✅ All checks passed!"
    text = verdict_text([("lint", ["a.py:1: E501 too long"]), ("types", ["b.py:2: error: x"])])
    assert "   ❌ lint\n   ❌ types\n" in text
    assert "--- lint ---\na.py:1: E501 too long\n\n--- types ---\nb.py:2: error: x\n" in text


def test_tree_fingerprint_follows_source_edits_and_skips_caches(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "a.py").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    before = tree_fingerprint(tmp_path)
    assert tree_fingerprint(tmp_path) == before
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "a.cpython.pyc").write_bytes(b"\x00")
    (tmp_path / "src" / "node_modules").mkdir()
    (tmp_path / "src" / "node_modules" / "m.js").write_text("1")
    assert tree_fingerprint(tmp_path) == before, "caches and node_modules change nothing"
    (src / "a.py").write_text("x = 2\n")
    # Same size; an edit a second later (the kernel's mtime tick can be
    # coarser than this test) is what the fingerprint sees.
    later = (src / "a.py").stat().st_mtime_ns + 10**9
    os.utime(src / "a.py", ns=(later, later))
    edited = tree_fingerprint(tmp_path)
    assert edited != before
    # Audits A/X1-F3, B/X1-F7, same cause: mypy and pyright follow the imports of ``src/``
    # into ``projects/``, the other source tree, which the fingerprint left out.
    (tmp_path / "projects" / "pkg").mkdir(parents=True)
    (tmp_path / "projects" / "pkg" / "b.py").write_text("y = 1\n")
    assert tree_fingerprint(tmp_path) != edited


def test_tree_fingerprint_follows_the_tool_configuration(tmp_path: Path) -> None:
    """A new mypy/ruff/pyright setting or lock change invalidates the cached verdict."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    assert set(check.CHECK_CONFIG_FILES) >= {"pyproject.toml", "uv.lock", "pyrightconfig.json"}
    seen = {tree_fingerprint(tmp_path)}
    for name in check.CHECK_CONFIG_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(f"# {name}\n")
        fingerprint = tree_fingerprint(tmp_path)
        assert fingerprint not in seen, name
        seen.add(fingerprint)
    (tmp_path / "unrelated.cfg").write_text("ignored\n")
    assert tree_fingerprint(tmp_path) in seen


def test_tree_fingerprint_follows_every_file_sea_lint_reads(tmp_path: Path) -> None:
    """Audits A/X1-F3, B/X1-F7: an edit to a page ``sea lint`` reads outside ``src/`` kept the
    cache key, so ``check --fix`` replayed the verdict of the tree before the edit."""
    from kiss.agents.sorcar import sea_lint

    (tmp_path / "src").mkdir()
    for rel in sea_lint.PROSE_FILES:
        page = tmp_path / rel
        page.parent.mkdir(parents=True, exist_ok=True)
        absent = tree_fingerprint(tmp_path)
        page.write_text("Fine.\n")
        clean = tree_fingerprint(tmp_path)
        page.write_text("The settings are in ~/.kiss/config.json.\n")
        assert sea_lint.lint_prose(tmp_path), f"the edit of {rel} must fail sea lint"
        failing = tree_fingerprint(tmp_path)
        page.unlink()
        assert len({absent, clean, failing}) == 3, f"{rel} added, then edited"
        assert tree_fingerprint(tmp_path) == absent, f"{rel} removed"


def test_cached_verdict_ignores_a_cache_that_is_not_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check, "CACHE_FILE", tmp_path / "tmp" / "check-cache.json")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "check-cache.json").write_text("[1, 2]")
    assert check.cached_verdict("f1", True) is None
    check.save_verdict("f1", True, 0, "")
    cached = check.cached_verdict("f1", True)
    assert cached is not None and cached["exit_code"] == 0 and cached["log"] == ""


def test_cached_verdict_round_trips_per_tree_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check, "CACHE_FILE", tmp_path / "tmp" / "check-cache.json")
    assert check.cached_verdict("f1", True) is None
    check.save_verdict("f1", True, 1, "the log\n")
    cached = check.cached_verdict("f1", True)
    assert cached is not None and cached["exit_code"] == 1 and cached["log"] == "the log\n"
    assert check.cached_verdict("f2", True) is None, "another tree"
    assert check.cached_verdict("f1", False) is None, "another mode"
    (tmp_path / "tmp" / "check-cache.json").write_text("not json")
    assert check.cached_verdict("f1", True) is None


def _save_verdicts(log: str, times: int) -> None:
    """Save *times* verdicts of the tree ``f1`` that carry *log*."""
    for _ in range(times):
        check.save_verdict("f1", True, 0, log)


@posix_only("Windows refuses the rename over a cache a reader holds open")
def test_checks_ending_together_each_publish_a_whole_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit SC1: written in place, a short verdict kept the tail of a longer one saved between
    its open and its write (no JSON: the next run checked the unchanged tree again), and a
    reader found the cache empty in between.  Each save renames a file of its own over the
    cache: the hard link goes on naming the whole verdict of the save before."""
    cache = tmp_path / "tmp" / "check-cache.json"
    monkeypatch.setattr(check, "CACHE_FILE", cache)
    logs = ["", "a failed stage\n" * 2000]
    check.save_verdict("f1", True, 1, logs[1])
    os.link(cache, tmp_path / "earlier")
    check.save_verdict("f1", True, 0, logs[0])
    assert json.loads((tmp_path / "earlier").read_text())["exit_code"] == 1, "rewritten in place"
    writers = [threading.Thread(target=_save_verdicts, args=(log, 300)) for log in logs]
    for writer in writers:
        writer.start()
    seen = []
    while any(writer.is_alive() for writer in writers):
        seen.append(check.cached_verdict("f1", True))
    assert all(verdict is not None and verdict["log"] in logs for verdict in seen)
    assert [entry.name for entry in cache.parent.iterdir()] == [cache.name], "staging files left"
    # A cache that cannot be replaced (a directory has its name) costs the save, not the check.
    cache.unlink()
    cache.mkdir()
    check.save_verdict("f1", True, 0, "")
    assert [entry.name for entry in cache.parent.iterdir()] == [cache.name], "staging files left"
    assert check.cached_verdict("f1", True) is None


def test_the_sync_inside_a_worktree_leaves_the_shared_editable_install_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-01 (task 4aa14638): 19 ``check --full`` runs inside the task's
    worktree each re-pointed the shared venv's editable ``kiss`` at the worktree."""
    # pretend checkouts (never created): outside any worktree, which a pytest
    # ``--basetemp`` under this repository would put ``tmp_path`` inside of
    base = tmp_path if not check.in_worktree(tmp_path) else Path(tempfile.gettempdir()) / "x"
    repo = base / "repo"
    assert check.sync_command(repo) == ["uv", "sync"]
    worktree = repo / check.WORKTREES_DIR / "kiss_wt-1-abc"
    # --inexact as well: ``uv sync --no-install-project --dry-run`` on the
    # shared environment would otherwise "uninstall kiss-agent-framework"
    assert check.sync_command(worktree) == ["uv", "sync", "--inexact", "--no-install-project"]
    # a project merely named like the directory is not a worktree
    assert check.sync_command(repo / "kiss-worktrees-notes") == ["uv", "sync"]
    # the real check runs on this checkout: a worktree checkout syncs dependencies only
    expected = (
        ["uv", "sync", "--inexact", "--no-install-project"] if check.in_worktree()
        else ["uv", "sync"])
    assert check.sync_command() == expected
    # every later stage's ``uv run`` would sync (and so install) the project
    # again: inside a worktree it runs on the environment as it is, a console
    # script of the project as its module on the worktree's own source
    assert check.uv_run("ruff", "src/", root=repo) == ["uv", "run", "ruff", "src/"]
    assert check.uv_run("mypy", "src/", root=worktree) == ["uv", "run", "--no-sync", "mypy", "src/"]
    assert check.project_script("sea", "kiss.agents.sorcar.sea_cli", "lint", root=repo) == [
        "uv", "run", "sea", "lint"]
    assert check.project_script("sea", "kiss.agents.sorcar.sea_cli", "lint", root=worktree) == [
        "uv", "run", "--no-sync", "python", "-m", "kiss.agents.sorcar.sea_cli", "lint"]
    assert check.stage_env(repo) is None
    # the stages of a worktree run in the main checkout's environment (uv would
    # otherwise build the worktree one of its own) on the worktree's sources
    assert check.main_checkout(worktree) == repo.resolve() and check.main_checkout(repo) == repo
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    env = check.stage_env(worktree)
    assert env is not None and env["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(worktree / "src"), str(worktree / "projects")]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(repo.resolve() / ".venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/elsewhere/.venv")  # the caller's choice holds
    assert (check.stage_env(worktree) or {})["UV_PROJECT_ENVIRONMENT"] == "/elsewhere/.venv"
    assert all(uv_run[:2] == ["uv", "run"] for uv_run, _ in check.FIXERS)
    assert ("--no-sync" in check.FIXERS[0][0]) == check.in_worktree()


def test_clean_only_removes_the_build_directories_and_nothing_else(tmp_path: Path) -> None:
    """Audits C/C14-F2 and A/X1-F2: the clean step deleted the project's ``.claude`` settings
    and skills, and asked ``git`` about every ``__pycache__`` and ``.pyc`` to remove none.

    The real script runs as the program of a scratch project with no tool on ``PATH``.
    """
    script = tmp_path / "src" / "kiss" / "scripts" / "check.py"
    script.parent.mkdir(parents=True)
    shutil.copy(check.__file__, script)
    removed = ["dist", "build", "kiss.egg-info", "artifacts", "uv_cache"]
    kept = [".claude/settings.json", ".claude/skills/review/SKILL.md",
            "src/kiss/__pycache__/a.cpython-313.pyc", "stray.pyc"]
    for name in removed:
        (tmp_path / name / "sub").mkdir(parents=True)
    for name in kept:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("mine")
    done = subprocess.run(
        [sys.executable, str(script), "--clean-only"],
        env={**os.environ, "PATH": str(tmp_path / "no-tools")},
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert [name for name in removed if (tmp_path / name).exists()] == []
    assert [name for name in kept if not (tmp_path / name).is_file()] == []
    # Audit SC1: with --no-clean as well, nothing was cleaned and every check ran.
    (tmp_path / "dist").mkdir()
    done = subprocess.run(
        [sys.executable, str(script), "--no-clean", "--clean-only"],
        env={**os.environ, "PATH": str(tmp_path / "no-tools")},
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 2 and "not allowed with argument" in done.stderr, done.stderr
    assert (tmp_path / "dist").is_dir() and "Running" not in done.stdout


def test_a_read_only_run_changes_nothing_and_keeps_every_judging_stage(tmp_path: Path) -> None:
    """``--read-only`` is the lint that runs beside the tests.

    It neither cleans nor syncs nor generates docs, and it runs every stage
    that judges the code.  The real script runs as the program of a scratch
    project with no tool on ``PATH``, so the first stage it starts is named
    in the output and is the last thing it does.
    """
    script = tmp_path / "src" / "kiss" / "scripts" / "check.py"
    script.parent.mkdir(parents=True)
    shutil.copy(check.__file__, script)
    (tmp_path / "dist").mkdir()
    env = {**os.environ, "PATH": str(tmp_path / "no-tools")}
    done = subprocess.run(
        [sys.executable, str(script), "--read-only", "--full"],
        env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 1 and (tmp_path / "dist").is_dir(), done.stderr
    assert re.findall(r"^Running: (.+)$", done.stdout, re.MULTILINE) == ["Lint SEAs (sea lint)"]
    reading = [description for _command, description in check.check_stages(True, True)]
    assert reading[:5] == [
        "Lint SEAs (sea lint)", "Syntax check (compileall)", "Lint code (ruff)",
        "Type check (mypy)", "Type check (pyright)",
    ]
    # No stage lets ``uv run`` sync the environment first, in a tree that is no worktree
    # either: the tree is named, wherever this test runs, and the same run that may
    # write shows what ``uv_run`` alone builds for it.
    outside = Path(tmp_path.anchor) / "checkout"
    assert not check.in_worktree(outside)
    commands = [command for command, _description in check.check_stages(True, True, outside)]
    assert len(commands) == 5
    assert all(command[:3] == ["uv", "run", "--no-sync"] for command in commands), commands
    syncing = [command for command, _description in check.check_stages(True, False, outside)]
    assert ["uv", "run", "ruff", "check", "src/"] in syncing and ["uv", "sync"] in syncing
    # The same run without the flag starts with the stages that write.
    writing = [description for _command, description in check.check_stages(True, False)]
    assert writing[:2] == ["Install dependencies (uv sync)", "Generate API docs"]
    assert "Generate SEA vocabulary tables (sea docs)" in writing
    for flag in ("--fix", "--clean-only"):
        refused = subprocess.run(
            [sys.executable, str(script), "--read-only", flag],
            env=env, capture_output=True, text=True, timeout=300, check=False,
        )
        assert refused.returncode == 2, refused.stderr
        assert "--read-only changes nothing" in refused.stderr
        assert (tmp_path / "dist").is_dir() and "Running" not in refused.stdout


def test_concurrent_checks_of_one_tree_clean_it_without_failing(tmp_path: Path) -> None:
    """Audit SC1: checks started together in one tree (the sub-agents of a task share theirs)
    all saw the build directory and all removed it: every one but the fastest died of
    ``FileNotFoundError`` before a single stage ran."""
    script = tmp_path / "src" / "kiss" / "scripts" / "check.py"
    script.parent.mkdir(parents=True)
    shutil.copy(check.__file__, script)
    for folder in range(40):
        (tmp_path / "build" / str(folder)).mkdir(parents=True)
        for name in range(60):
            (tmp_path / "build" / str(folder) / str(name)).write_text("x")
    runs = [
        subprocess.Popen(
            [sys.executable, str(script), "--clean-only"],
            env={**os.environ, "PATH": str(tmp_path / "no-tools")},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(4)
    ]
    errors = [run.communicate(timeout=300)[1] for run in runs]
    assert [run.returncode for run in runs] == [0] * 4, errors
    assert not (tmp_path / "build").exists()
