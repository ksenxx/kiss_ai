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
observed behavior is the real subprocess path.  ``main()`` itself is
not executed: it runs the repository's whole ruff/mypy/pyright pass,
which takes minutes and depends on the state of the checkout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
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
    assert tree_fingerprint(tmp_path) != before


def test_tree_fingerprint_follows_the_tool_configuration(tmp_path: Path) -> None:
    """A new mypy/ruff/pyright setting or lock change invalidates the cached verdict."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    assert set(check.CHECK_CONFIG_FILES) >= {"pyproject.toml", "uv.lock", "pyrightconfig.json"}
    seen = {tree_fingerprint(tmp_path)}
    for name in check.CHECK_CONFIG_FILES:
        (tmp_path / name).write_text(f"# {name}\n")
        fingerprint = tree_fingerprint(tmp_path)
        assert fingerprint not in seen, name
        seen.add(fingerprint)
    (tmp_path / "unrelated.cfg").write_text("ignored\n")
    assert tree_fingerprint(tmp_path) in seen


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
