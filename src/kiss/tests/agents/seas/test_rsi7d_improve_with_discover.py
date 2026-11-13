# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: rsi7d's ``improve_sea_with_discover`` lets ``/discover`` edit a SEA's script in a worktree.

The dispatch is the real ``agent_dispatch`` job (``daemon_client.run``
over the local WSS endpoint) against a daemon stand-in that, in place
of the ``/discover`` child, edits the worktree the tool checked out
(committing and resetting as the discovery loop does) and answers a
scripted result.  Everything around it is the real code: the worktree
under ``tmp/rsi7d/discover/<name>`` seeded with the checkout's current
script and test file, the clone of the past run's repository at its
base commit, the baseline figures from the persisted run, the task the
child receives, the guard that confines the child's changes to the
SEA's script and test file, the compile/load/lint gate, the snapshot,
the revert (an interrupt inside the gate included), the worktree's
removal and ``replay_findings``.

Not covered: the "daemon's installed kiss package predates
``agent_dispatch.dispatch_result``" branch, reachable only when the
daemon runs an older installed package than the checkout whose SEA file
it loads (see ``test_rsi7d_replay_in_clone.py``), and git failing
inside the worktree after ``git worktree add`` succeeded.
"""

from __future__ import annotations

import json
import os
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.discover import discover_sea
from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.tests.agents.seas.conftest import (
    add_extra,
    commit,
    fake_checkout,
    git,
    init_repo,
    persist_task,
)
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon
from kiss.tests.conftest import is_root

DEMO_SEA = '''"""Demo SEA: a worker with a tool."""

from kiss.agents.seas.base.base_sea import WorkerSea

SYSTEM_PROMPT = "You run demo tasks. Report the exit code."


class Sea(WorkerSea):
    def description(self):
        """Help text."""
        return "Runs demo tasks."

    def settings(self, settings):
        """Settings."""
        return settings | {"timeout": 7200}

    def system_prompt(self, system_prompt):
        """Prompt."""
        return SYSTEM_PROMPT
'''

DEMO_TEST = '"""Tests of the demo SEA."""\n\n\ndef test_demo() -> None:\n    assert True\n'

REL_SEAS_DIR = "src/kiss/agents/seas"
REL_SEA = f"{REL_SEAS_DIR}/tunedemo/tunedemo_sea.py"
REL_TEST = "src/kiss/tests/agents/seas/test_tunedemo_sea.py"
WORKTREE = sea.DISCOVER_DIR / "tunedemo"

# A script that raises the task's Stop while the gate loads it: the loader re-raises a Stop
# unchanged (``sea_settings.is_stop``), as when the user stops the sweep during the gate.
STOPPED_SEA = (
    "from kiss.core.tool_interrupt import ToolCallInterrupted\n\n"
    'raise ToolCallInterrupted("stopped")\n'
)


def _break_settings(path: Path) -> None:
    """Make ``settings`` return a value the daemon's load refuses: the gate rejects it."""
    path.write_text(path.read_text().replace('"timeout": 7200', '"timeout": "soon"'))


def writes(files: dict[str, str], extra: bool = False) -> Callable[[Path], None]:
    """Script a child's edit of its worktree: with *extra*, append ``extra()`` to the script;
    then write *files* (worktree-relative path -> text), creating folders as the child's
    ``Write`` tool does."""
    def edit(worktree: Path) -> None:
        if extra:
            add_extra(worktree / REL_SEA)
        for rel, text in files.items():
            (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
            (worktree / rel).write_text(text, encoding="utf-8")
    return edit


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_history: None) -> Path:
    """A fake KISS checkout (a git repository) holding the editable ``tunedemo`` SEA and its test.

    The sweep's tools find the checkout through the cwd; the past run
    the child replays is persisted with this repository as its work dir,
    so ``prepare_replay_clone`` clones it at the commit before the run.
    """
    root = tmp_path.resolve()
    seas = fake_checkout(root, {"tunedemo": DEMO_SEA})
    (seas.parents[1] / "SYSTEM.md").write_text("You are a test Sorcar.\n", encoding="utf-8")
    (root / REL_TEST).parent.mkdir(parents=True)
    (root / REL_TEST).write_text(DEMO_TEST, encoding="utf-8")
    (root / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    init_repo(root)
    commit(root, "baseline", when=int(time.time()) - 600)
    monkeypatch.chdir(root)
    return root


def _past_run(root: Path, cost: float = 4.0, model: str = "model-a") -> str:
    """Persist a finished ``tunedemo`` run of *model* in *root* for *cost* USD; return its id."""
    return persist_task(
        f"Run the demo in {root}", result="success: true\nsummary: done", work_dir=str(root),
        sea="tunedemo_sea", cost=cost, steps=7, model=model, success=True,
    )


def loop(child_id: str, *edits: Callable[[Path], object]) -> Callable[[Path], None]:
    """Script a discovery loop for a child *child_id*: commit the baseline B, then apply each
    of *edits* to the worktree and commit it as a point, resetting to B in between so that
    only the last point is left in the tree, every commit held by a ``refs/discover`` ref."""
    def run(worktree: Path) -> None:
        git(worktree, "update-ref", f"refs/discover/{child_id}/B", commit(worktree, "B"))
        for index, edit in enumerate(edits, 1):
            git(worktree, "reset", "-q", "--hard", f"refs/discover/{child_id}/B")
            edit(worktree)
            point = commit(worktree, f"P{index}")
            git(worktree, "update-ref", f"refs/discover/{child_id}/P{index}", point)
    return run


class _DiscoverDaemon(RecordingDaemon):
    """A daemon stand-in whose "child" edits the worktree it is dispatched into.

    ``improve_sea_with_discover`` dispatches the ``/discover`` SEA through
    the real ``daemon_client.run``; the stand-in records the command and,
    in place of the child, runs :attr:`act` on the worktree the tool
    checked out (:data:`WORKTREE` under *root*) before the scripted
    result is sent.  :attr:`seen` keeps the script the child found there.
    """

    def __init__(self, root: Path, text: str = "best point found", success: bool = True) -> None:
        super().__init__(text=text, cost=3.5, steps=40)
        # A real child's id is a task_history UUID; its replays name it as their parent.
        self.child_id = self.result["taskId"] = uuid.uuid4().hex
        self.result["success"] = success
        self.worktree = root / WORKTREE
        self.act: Callable[[Path], object] | None = None
        self.seen = ""

    async def _handle(self, ws: Any) -> None:
        self.seen = (self.worktree / REL_SEA).read_text(encoding="utf-8")
        if self.act is not None:
            self.act(self.worktree)
        await super()._handle(ws)


def _improve(run_id: str, heldout_id: str = "", **kwargs: Any) -> dict[str, Any]:
    """Call ``improve_sea_with_discover`` on ``tunedemo`` with test caps; return its JSON."""
    report: dict[str, Any] = json.loads(sea.improve_sea_with_discover(
        "tunedemo", "Idea 1: state the exit code first (evidence: run X [3]).", 30.0, run_id,
        8.0, heldout_id, timeout=60.0, **kwargs,
    ))
    return report


def _assert_untouched(root: Path, test: str | None = DEMO_TEST) -> None:
    """The checkout holds the demo SEA (and *test*, or no test file) and no worktree is left."""
    assert (root / REL_SEA).read_text() == DEMO_SEA
    assert ((root / REL_TEST).read_text() if (root / REL_TEST).exists() else None) == test
    assert not (root / WORKTREE).exists()
    assert len(git(root, "worktree", "list").splitlines()) == 1  # the checkout itself
    assert git(root, "status", "--porcelain") == ""


def test_refusals_before_any_dispatch(root: Path) -> None:
    """A SEA outside the editable folders and an unknown past run are refused untouched."""
    assert sea.improve_sea_with_discover("slack", "x", 1.0, "t", 1.0).startswith("Error:")
    assert sea.improve_sea_with_discover("tunedemo", "x", 1.0, "nope", 1.0) == (
        "Error: unknown task id 'nope'"
    )
    assert sea.improve_sea_with_discover("tunedemo", "x", 1.0, _past_run(root), 1.0, "nope") == (
        "Error: unknown task id 'nope'"
    )
    _assert_untouched(root)
    assert sea.replay_findings("nobody") == "[]"


def test_child_task_worktree_gate_replays_and_revert(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child gets the evidence, the clone recipes and the baselines, and a worktree holding
    the checkout's current (uncommitted) script; the script and test it leaves there go
    through the gate into the checkout; its replays are measured; the worktree is removed,
    its commits stay under the child's refs; the snapshot reverts everything."""
    run_id = _past_run(root)
    heldout_id = _past_run(root, cost=2.0)
    # This checkout's own ``/discover`` file is the one dispatched when it is editable here.
    discover_dir = root / "src" / "kiss" / "agents" / "seas" / "discover"
    discover_dir.mkdir()
    (discover_dir / "discover_sea.py").write_text(
        Path(discover_sea.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    commit(root, "add the discover SEA")
    patched = DEMO_SEA.replace("Report the exit code.", "Report the exit code first.")
    (root / REL_SEA).write_text(patched, encoding="utf-8")  # an earlier, uncommitted edit
    daemon = _DiscoverDaemon(root)
    try:
        monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
        daemon.act = loop(
            daemon.child_id,
            lambda wt: _break_settings(wt / REL_SEA),  # P1, reset away
            writes({REL_TEST: DEMO_TEST + "\n# extra\n"}, extra=True),
        )
        replay_id = persist_task(  # a replay the child dispatched: a sub-task of the child's id
            "replay", result="success: true\nsummary: ok", success=True, sea="tunedemo_sea",
            parent_task_id=daemon.child_id, cost=1.25, steps=4,
        )
        report = _improve(run_id, heldout_id, rounds=3)
    finally:
        daemon.close()
    assert report["task_id"] == daemon.child_id and "error" not in report, report
    assert daemon.seen == patched  # the worktree started from the checkout's working tree
    assert report["result"] == {
        "success": True, "summary": "best point found", "cost": 3.5, "steps": 40,
    }
    assert report["applied"] == [REL_SEA, REL_TEST]
    assert (root / REL_SEA).read_text() == patched + "\n\ndef extra() -> int:\n    return 1\n"
    assert (root / REL_TEST).read_text() == DEMO_TEST + "\n# extra\n"
    assert "tunedemo_sea.py" in report["diff_stat"]
    assert "test_tunedemo_sea.py" in report["diff_stat"]
    assert report["other_changes"] == []
    assert report["refs"] == f"refs/discover/{daemon.child_id}/"
    refs = git(root, "for-each-ref", "--format=%(refname)", report["refs"]).splitlines()
    assert refs == [report["refs"] + name for name in ("B", "P1", "P2")]
    assert not (root / WORKTREE).exists()
    assert len(git(root, "worktree", "list").splitlines()) == 1
    (replay,) = report["replays"]
    assert replay["task_id"] == replay_id and replay["status"] == "success"
    assert replay["cost"] == 1.25 and replay["steps"] == 4 and "signals" not in replay
    assert json.loads(sea.replay_findings(daemon.child_id)) == report["replays"]

    (command,) = daemon.run_commands
    assert command["seaPath"] == str(discover_dir / "discover_sea.py")
    assert command["workDir"] == str(root / WORKTREE)
    prompt = command["prompt"]
    assert prompt.startswith("Improve the Sorcar Extension Agent (SEA) `tunedemo` by AI discovery.")
    assert "Idea 1: state the exit code first (evidence: run X [3])." in prompt
    assert f"Your work dir `{root / WORKTREE}`\nis a git worktree of the checkout `{root}`" in (
        prompt
    )
    assert "call\n`create_worktree()` once before your first edit" in prompt
    script = root / WORKTREE / REL_SEA
    assert (
        f"Edit the script `{script}` and, when a change needs\none, its test file "
        f"`{root / WORKTREE / REL_TEST}`, and nothing else"
    ) in prompt
    assert f"Run `uv run ...` commands from `{root}` (`cd {root} &&\nuv run ...`)" in prompt
    assert f"`uv run sea lint {script}` is FAILED before any replay. Round limit: 3." in prompt
    dev_clone, heldout_clone = report["clones"]
    assert dev_clone == str(root / sea.REPLAY_DIR / f"tunedemo-{run_id[:8]}")
    assert heldout_clone == str(root / sea.REPLAY_DIR / f"tunedemo-{heldout_id[:8]}")
    assert Path(dev_clone).is_dir() and Path(heldout_clone).is_dir()
    assert f"past run {run_id} on `model-a`: success, $4.00, 7 steps, 0 mistake signals." in prompt
    # The reset command names the base commit the clone was checked out at.
    base = git(Path(dev_clone), "rev-parse", "HEAD")
    assert base == git(Path(heldout_clone), "rev-parse", "HEAD") and len(base) == 40
    dev_reset = f"git -C {dev_clone} reset --hard --quiet {base} && git -C {dev_clone} clean -fdxq"
    assert f"before each replay run\n  `{dev_reset}`." in prompt
    assert f'options=\'{{"work_dir": "{dev_clone}",\n  "inherit": false}}\'' in prompt
    assert f'agent="{script}"' in prompt
    assert 'model="model-a", max_budget="8",\n  timeout="3600"' in prompt
    assert "from kiss.agents.seas.rsi7d.rsi7d_sea import replay_findings" in prompt
    assert "within 25% of the baseline is noise" in prompt
    heldout_task_file = Path(heldout_clone).with_suffix(".task.txt")
    heldout_reset = (
        f"git -C {heldout_clone} reset --hard --quiet {base} && git -C {heldout_clone} clean -fdxq"
    )
    assert (  # the held-out replay has its own cap: twice the held-out run's cost by default
        f"Held-out check: the same procedure once on the baseline past run {heldout_id} on "
        f"`model-a`: success, $2.00, 7 steps, 0 mistake signals: clone `{heldout_clone}` "
        f"(reset first with `{heldout_reset}`), work dir `{heldout_clone}`, task text in "
        f'`{heldout_task_file}`, max_budget="4"; run it on the point that meets the goal on '
        "the dev replay."
    ) in prompt
    # The task text the child replays names the clone, not the repository.
    task_file = Path(dev_clone).with_suffix(".task.txt")
    assert f"task=<the text of\n  {task_file}, verbatim>" in prompt
    assert task_file.read_text(encoding="utf-8") == f"Run the demo in {dev_clone}"
    assert heldout_task_file.read_text(encoding="utf-8") == f"Run the demo in {heldout_clone}"

    snapshot = Path(report["snapshot"])
    assert (snapshot / "tunedemo_sea.py").read_text() == patched
    folder = root / "src" / "kiss" / "agents" / "seas" / "tunedemo"
    assert sea.revert_sea_code("tunedemo") == f"Restored {folder} from {snapshot}"
    assert (root / REL_SEA).read_text() == patched
    assert (root / REL_TEST).read_text() == DEMO_TEST
    assert git(root, "status", "--porcelain") == f"M {REL_SEA}"  # the earlier edit alone


def test_default_held_out_and_an_uncommitted_point(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a held-out run the child is told to replay the winning point a second time; a
    run that recorded no model is replayed on the default model; the bundled ``/discover``
    file is dispatched when this checkout has none; an edit the child left uncommitted in
    the worktree is kept like a committed one."""
    run_id = _past_run(root, model="")
    daemon = _DiscoverDaemon(root)
    try:
        monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
        daemon.act = writes({}, extra=True)
        report = _improve(run_id)
    finally:
        daemon.close()
    assert "error" not in report and report["applied"] == [REL_SEA], report
    assert "def extra" in (root / REL_SEA).read_text()
    assert report["clones"] == [str(root / sea.REPLAY_DIR / f"tunedemo-{run_id[:8]}")]
    assert report["replays"] == [] and not (root / WORKTREE).exists()
    (command,) = daemon.run_commands
    assert command["seaPath"] == str(Path(discover_sea.__file__).resolve())
    prompt = command["prompt"]
    assert "Round limit: 2." in prompt
    assert (
        "Held-out check: a second dev replay of the point that meets the goal; its metrics "
        "must hold again"
    ) in prompt
    assert "on `the default model`: success, $4.00" in prompt and "task text in" not in prompt
    assert 'model="' not in prompt and 'verbatim>, max_budget="8",' in prompt


def test_failed_children_leave_the_folder_as_it_was(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that did not succeed, one that changed a file outside the script and test
    (committed, or left uncommitted), one that added a helper module next to the script,
    one that renamed the script, one that broke the SEA, one that kept the baseline and one
    that deleted the script each end with the checkout as it was, the worktree removed and
    the reason reported."""
    run_id = _past_run(root)
    system_md = root / "src" / "kiss" / "SYSTEM.md"
    refused = "the child changed files other than the SEA's script and test: "
    daemon = _DiscoverDaemon(root, success=False)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        daemon.act = writes({}, extra=True)
        report = _improve(run_id)
        assert report["error"] == "the child did not report success" and "applied" not in report
        _assert_untouched(root)
        daemon.result["success"] = True

        outside = writes({"src/kiss/SYSTEM.md": "changed\n"}, extra=True)
        daemon.act = loop(daemon.child_id, outside)
        report = _improve(run_id)
        assert report["error"] == refused + "['src/kiss/SYSTEM.md']" and "applied" not in report
        assert system_md.read_text() == "You are a test Sorcar.\n"
        _assert_untouched(root)

        daemon.act = outside  # the same, left uncommitted
        report = _improve(run_id)
        assert report["error"] == refused + "['src/kiss/SYSTEM.md']"
        _assert_untouched(root)

        helper = f"{REL_SEAS_DIR}/tunedemo/helper.py"  # the replays never ran a worktree helper
        daemon.act = loop(daemon.child_id, writes({helper: "X = 1\n"}, extra=True))
        report = _improve(run_id)
        assert report["error"] == refused + f"['{helper}']"
        assert not (root / helper).exists()
        _assert_untouched(root)

        renamed = f"{REL_SEAS_DIR}/tunedemo/other_sea.py"
        daemon.act = loop(
            daemon.child_id, lambda wt: (wt / REL_SEA).rename(wt / renamed)
        )
        report = _improve(run_id)
        assert report["error"] == refused + f"['{renamed}']"
        assert not (root / renamed).exists()
        _assert_untouched(root)

        daemon.act = loop(daemon.child_id, lambda wt: _break_settings(wt / REL_SEA))
        report = _improve(run_id)
        assert report["error"].startswith("the patched SEA no longer loads (")
        assert report["error"].endswith("; file restored") and "applied" not in report
        _assert_untouched(root)

        daemon.act = loop(daemon.child_id)  # the baseline stays the best point
        report = _improve(run_id)
        assert report["error"] == "the child kept the baseline: nothing to apply"
        _assert_untouched(root)

        daemon.act = loop(daemon.child_id, lambda wt: (wt / REL_SEA).unlink())
        report = _improve(run_id)
        assert report["error"] == "the child deleted the script" and "applied" not in report
        _assert_untouched(root)

        daemon.act = loop(daemon.child_id, lambda wt: (wt / REL_SEA).write_bytes(b"\xff\xfe"))
        report = _improve(run_id)
        assert report["error"] == "the script the child left is not UTF-8 text"
        _assert_untouched(root)
    finally:
        daemon.close()
    assert len(daemon.run_commands) == 9
    assert all(c["workDir"] == str(root / WORKTREE) for c in daemon.run_commands)
    assert report["refs"] == f"refs/discover/{daemon.child_id}/"
    assert git(root, "for-each-ref", "--format=%(refname)", report["refs"]).splitlines()


def test_an_unreachable_daemon(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable daemon is a reported failure that leaves the checkout as it was."""
    run_id = _past_run(root)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(root / "no-daemon.json"))
    report = _improve(run_id)
    assert report["task_id"] == "" and "could not run" in report["result"]
    assert report["error"] == "the child did not report success" and report["refs"] == ""
    assert report["replays"] == []
    _assert_untouched(root)


def test_a_sea_folder_outside_git_gets_no_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_history: None
) -> None:
    """A SEA whose folder is in no git repository is refused before any dispatch: the child
    has no worktree to edit it in (the past run's own repository is still cloned)."""
    root = tmp_path.resolve()
    fake_checkout(root, {"tunedemo": DEMO_SEA})
    repo = root / "repo"
    init_repo(repo)
    commit(repo, "seed", {"a.txt": "a\n"}, when=int(time.time()) - 600)
    monkeypatch.chdir(root)
    run_id = persist_task(
        f"Run the demo in {repo}", result="success: true\nsummary: done", work_dir=str(repo),
        sea="tunedemo_sea", cost=4.0, steps=7, model="model-a", success=True,
    )
    assert sea.improve_sea_with_discover("tunedemo", "x", 30.0, run_id, 8.0).startswith(
        "Error: git worktree add failed: fatal: not a git repository"
    )
    assert (root / sea.REPLAY_DIR / f"tunedemo-{run_id[:8]}").is_dir()
    assert (root / REL_SEA).read_text() == DEMO_SEA and not (root / WORKTREE).exists()


def test_created_test_file_held_out_cap_and_an_interrupted_gate(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SEA with no test file yet: the test file the child creates is applied and removed again
    by the revert; the held-out replay gets the cap the caller names; a Stop raised while the
    gate loads the child's script restores the script, creates no test file and removes the
    worktree before it propagates."""
    (root / REL_TEST).unlink()
    commit(root, "no test yet")
    run_id = _past_run(root)
    heldout_id = _past_run(root, cost=2.0)
    daemon = _DiscoverDaemon(root)
    try:
        monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
        daemon.act = loop(daemon.child_id, writes({REL_TEST: DEMO_TEST}, extra=True))
        report = _improve(run_id, heldout_id, heldout_budget=12.5)
        assert "error" not in report and report["applied"] == [REL_SEA, REL_TEST], report
        assert (root / REL_TEST).read_text() == DEMO_TEST
        assert "def extra" in (root / REL_SEA).read_text()
        # ``diff_stat`` covers tracked files; the created test file is in ``applied`` above.
        assert report["diff_stat"].endswith("1 file changed, 4 insertions(+)")
        assert report["other_changes"] == []
        prompt = daemon.run_commands[0]["prompt"]
        assert f"its test file `{root / WORKTREE / REL_TEST}`, and nothing else" in prompt
        assert 'max_budget="8",\n  timeout="3600"' in prompt  # the dev cap
        assert 'max_budget="12.5"; run it on the point that meets the goal' in prompt
        assert sea.revert_sea_code("tunedemo").startswith("Restored ")
        _assert_untouched(root, test=None)

        daemon.act = loop(daemon.child_id, writes({REL_SEA: STOPPED_SEA, REL_TEST: DEMO_TEST}))
        with pytest.raises(ToolCallInterrupted, match="stopped"):
            _improve(run_id)
    finally:
        daemon.close()
    assert len(daemon.run_commands) == 2
    _assert_untouched(root, test=None)


@pytest.mark.skipif(is_root(), reason="root writes read-only files")
def test_a_failed_test_file_copy_puts_the_script_back(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the test file cannot be written after the script passed the gate, the error
    propagates with the checkout's script and test file as they were (never the new script
    with the old test) and the worktree removed."""
    run_id = _past_run(root)
    daemon = _DiscoverDaemon(root)
    try:
        monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
        daemon.act = loop(daemon.child_id, writes({REL_TEST: DEMO_TEST + "\n# x\n"}, extra=True))
        os.chmod(root / REL_TEST, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        with pytest.raises(PermissionError):
            _improve(run_id)
    finally:
        os.chmod(root / REL_TEST, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        daemon.close()
    _assert_untouched(root)
