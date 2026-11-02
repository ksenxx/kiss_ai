# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: rsi7d's ``improve_sea_with_discover`` runs ``/discover`` on a SEA and gates its patch.

The dispatch is the real ``agent_dispatch`` job (``daemon_client.run``
over the local WSS endpoint) against a daemon stand-in that, in place
of the ``/discover`` child, writes a ``best.patch`` where the child's
scratch dir would hold it and answers a scripted result.  Everything
around it is the real code: the clone of the past run's repository at
its base commit, the baseline figures from the persisted run, the task
the child receives, ``git apply`` of the patch (checkout-relative or
folder-relative layout), the guard that confines it to the SEA's
script and test file, the compile/load/lint gate, the snapshot, the
revert (an interrupt inside the gate included) and ``replay_findings``.

Not covered: the "daemon's installed kiss package predates
``agent_dispatch.dispatch_result``" branch, reachable only when the
daemon runs an older installed package than the checkout whose SEA file
it loads (see ``test_rsi7d_replay_in_clone.py``), and a ``git apply
--numstat`` failure after ``git apply --check`` passed (both parse the
same patch, so one cannot fail alone).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.discover import discover_sea
from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.tests.agents.seas.conftest import commit, fake_checkout, git, init_repo, persist_task
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

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

# A script that raises the task's Stop while the gate loads it: the loader re-raises a Stop
# unchanged (``sea_settings.is_stop``), as when the user stops the sweep during the gate.
STOPPED_SEA = (
    "from kiss.core.tool_interrupt import ToolCallInterrupted\n\n"
    'raise ToolCallInterrupted("stopped")\n'
)


def _add_extra(path: Path) -> None:
    """Append a harmless ``extra()`` helper: a change the gate accepts."""
    path.write_text(path.read_text() + "\n\ndef extra() -> int:\n    return 1\n")


def _break_settings(path: Path) -> None:
    """Make ``settings`` return a value the daemon's load refuses: the gate rejects it."""
    path.write_text(path.read_text().replace('"timeout": 7200', '"timeout": "soon"'))


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
        f"Run the demo in {root}", result="success: true\nsummary: done", success=True,
        work_dir=str(root), sea="tunedemo_sea", cost=cost, steps=7, model=model,
    )


def _lab_patch(
    lab: Path, files: dict[str, str], edits: dict[str, Callable[[Path], object]]
) -> Path:
    """Build ``best.patch`` as a ``/discover`` lab does: commit *files* as the baseline B,
    apply *edits*, commit the point and diff the two commits (``git diff --binary B <point>``,
    so files the point added, renamed or deleted are in the patch)."""
    init_repo(lab)
    baseline = commit(lab, "B", files)
    for rel, edit in edits.items():
        (lab / rel).parent.mkdir(parents=True, exist_ok=True)
        edit(lab / rel)
    point = commit(lab, "point")
    patch = lab.parent / f"{lab.name}.best.patch"  # a lab's patch is always ``best.patch``
    patch.write_text(git(lab, "diff", "--binary", baseline, point) + "\n", encoding="utf-8")
    return patch


class _DiscoverDaemon(RecordingDaemon):
    """A daemon stand-in whose "child" drops a ``best.patch`` into its scratch dir on ``run``.

    ``improve_sea_with_discover`` dispatches the ``/discover`` SEA through
    the real ``daemon_client.run``; the stand-in records the command and,
    in place of the child, copies :attr:`patch` (``None``: nothing) to
    ``<work dir>/tmp/<child task id>/best.patch`` (*scratch*) before the
    scripted result is sent, so the tool finds the patch where the
    child's discovery loop writes it.
    """

    def __init__(self, root: Path, text: str = "best point found") -> None:
        super().__init__(text=text, cost=3.5, steps=40)
        # A real child's id is a task_history UUID; its replays name it as their parent.
        self.child_id = self.result["taskId"] = uuid.uuid4().hex
        self.scratch = root / "tmp" / self.child_id
        self.patch: Path | None = None

    async def _handle(self, ws: Any) -> None:
        if self.patch is not None:
            self.scratch.mkdir(parents=True, exist_ok=True)
            (self.scratch / "best.patch").write_text(self.patch.read_text(), encoding="utf-8")
        await super()._handle(ws)


def _improve(run_id: str, heldout_id: str = "", **kwargs: Any) -> dict[str, Any]:
    """Call ``improve_sea_with_discover`` on ``tunedemo`` with test caps; return its JSON."""
    report: dict[str, Any] = json.loads(sea.improve_sea_with_discover(
        "tunedemo", "Idea 1: state the exit code first (evidence: run X [3]).", 30.0, run_id,
        8.0, heldout_id, timeout=60.0, **kwargs,
    ))
    return report


def test_refusals_before_any_dispatch(root: Path) -> None:
    """A SEA outside the editable folders and an unknown past run are refused untouched."""
    assert sea.improve_sea_with_discover("slack", "x", 1.0, "t", 1.0).startswith("Error:")
    assert sea.improve_sea_with_discover("tunedemo", "x", 1.0, "nope", 1.0) == (
        "Error: unknown task id 'nope'"
    )
    assert sea.improve_sea_with_discover("tunedemo", "x", 1.0, _past_run(root), 1.0, "nope") == (
        "Error: unknown task id 'nope'"
    )
    assert (root / REL_SEA).read_text() == DEMO_SEA
    assert sea.replay_findings("nobody") == "[]"


def test_child_task_patch_gate_replays_and_revert(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child gets the evidence, the clone recipes and the baselines; its patch (script and
    test) goes through the gate; its replays are measured; the snapshot reverts everything."""
    run_id = _past_run(root)
    heldout_id = _past_run(root, cost=2.0)
    # This checkout's own ``/discover`` file is the one dispatched when it is editable here.
    discover_dir = root / "src" / "kiss" / "agents" / "seas" / "discover"
    discover_dir.mkdir()
    (discover_dir / "discover_sea.py").write_text(
        Path(discover_sea.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    commit(root, "add the discover SEA")
    daemon = _DiscoverDaemon(root)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    daemon.patch = _lab_patch(
        root / "tmp" / "lab-root",
        {REL_SEA: DEMO_SEA, REL_TEST: DEMO_TEST},
        {REL_SEA: _add_extra, REL_TEST: lambda p: p.write_text(p.read_text() + "\n# extra\n")},
    )
    replay_id = persist_task(  # a replay the child dispatched: a sub-task of the child's id
        "replay", result="success: true\nsummary: ok", success=True, sea="tunedemo_sea",
        parent_task_id=daemon.child_id, cost=1.25, steps=4,
    )
    try:
        report = _improve(run_id, heldout_id, rounds=3)
    finally:
        daemon.close()
    assert report["task_id"] == daemon.child_id and "error" not in report, report
    assert report["result"] == {
        "success": True, "summary": "best point found", "cost": 3.5, "steps": 40,
    }
    assert report["patch"] == str(daemon.scratch / "best.patch")
    assert report["applied"] == [REL_SEA, REL_TEST]
    assert "def extra" in (root / REL_SEA).read_text()
    assert (root / REL_TEST).read_text() == DEMO_TEST + "\n# extra\n"
    assert "tunedemo_sea.py" in report["diff_stat"]
    assert "test_tunedemo_sea.py" in report["diff_stat"]
    assert report["other_changes"] == []
    (replay,) = report["replays"]
    assert replay["task_id"] == replay_id and replay["status"] == "success"
    assert replay["cost"] == 1.25 and replay["steps"] == 4 and "signals" not in replay
    assert json.loads(sea.replay_findings(daemon.child_id)) == report["replays"]

    (command,) = daemon.run_commands
    assert command["seaPath"] == str(discover_dir / "discover_sea.py")
    assert command["workDir"] == str(root)
    prompt = command["prompt"]
    assert prompt.startswith("Improve the Sorcar Extension Agent (SEA) `tunedemo` by AI discovery.")
    assert "Idea 1: state the exit code first (evidence: run X [3])." in prompt
    # The lab holds the whole SEAs folder (sibling SEAs named by path exist) and the test file;
    # edits are confined to the script and the test file.
    assert f"copy `{root / REL_SEAS_DIR}` (the folder of every SEA folder" in prompt
    assert f"paths (`{REL_SEAS_DIR}/` and `{REL_TEST}`)" in prompt
    assert "Edit only the script and its test file:" in prompt
    assert f"Its script is `{root / REL_SEA}` in\nthe checkout `{root}`;" in prompt
    assert f"its test file is `{root / REL_TEST}`" in prompt and "Round limit: 3." in prompt
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
    assert f'agent="<your lab copy of {root / REL_SEA}>"' in prompt
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
    assert (snapshot / "tunedemo_sea.py").read_text() == DEMO_SEA
    folder = root / "src" / "kiss" / "agents" / "seas" / "tunedemo"
    assert sea.revert_sea_code("tunedemo") == f"Restored {folder} from {snapshot}"
    assert (root / REL_SEA).read_text() == DEMO_SEA
    assert (root / REL_TEST).read_text() == DEMO_TEST
    assert git(root, "status", "--porcelain") == ""


def test_folder_relative_patch_and_default_held_out(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lab that held the folder's files at its root still applies (``--directory``); without
    a held-out run the child is told to replay the winning point a second time; a run that
    recorded no model is replayed on the default model."""
    run_id = _past_run(root, model="")
    daemon = _DiscoverDaemon(root)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    daemon.patch = _lab_patch(
        root / "tmp" / "lab-flat", {"tunedemo_sea.py": DEMO_SEA}, {"tunedemo_sea.py": _add_extra}
    )
    try:
        report = _improve(run_id)
    finally:
        daemon.close()
    assert "error" not in report and report["applied"] == [REL_SEA], report
    assert "def extra" in (root / REL_SEA).read_text()
    assert report["clones"] == [str(root / sea.REPLAY_DIR / f"tunedemo-{run_id[:8]}")]
    assert report["replays"] == []
    prompt = daemon.run_commands[0]["prompt"]
    assert command_sea_path(daemon) == str(Path(discover_sea.__file__).resolve())
    assert "Round limit: 2." in prompt
    assert (
        "Held-out check: a second dev replay of the point that meets the goal; its metrics "
        "must hold again"
    ) in prompt
    assert "on `the default model`: success, $4.00" in prompt and "task text in" not in prompt
    assert 'model="' not in prompt and 'verbatim>, max_budget="8",' in prompt


def command_sea_path(daemon: RecordingDaemon) -> str:
    """Return the ``seaPath`` of the one ``run`` command *daemon* recorded."""
    (command,) = daemon.run_commands
    return str(command["seaPath"])


def test_failed_patches_leave_the_folder_as_it_was(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No patch, a patch outside the folder, one whose traditional ``---``/``+++`` section
    (no ``diff --git`` header) reaches outside, one adding a helper module next to the script,
    one renaming the script, one that breaks the SEA, a stale one and one that deletes the
    script each end with the snapshot restored and the reason reported."""
    run_id = _past_run(root)
    system_md = root / "src" / "kiss" / "SYSTEM.md"
    daemon = _DiscoverDaemon(root)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    restored = "; folder restored from the snapshot"
    refused = "best.patch touches files other than the SEA's script and test: "
    try:
        report = _improve(run_id)
        assert report["error"] == "the child reported no best.patch" + restored
        assert "patch" not in report and "applied" not in report

        daemon.patch = _lab_patch(
            root / "tmp" / "lab-outside",
            {REL_SEA: DEMO_SEA, "src/kiss/SYSTEM.md": system_md.read_text()},
            {REL_SEA: _add_extra, "src/kiss/SYSTEM.md": lambda p: p.write_text("changed\n")},
        )
        report = _improve(run_id)
        assert report["error"] == refused + "['src/kiss/SYSTEM.md']" + restored
        assert system_md.read_text() == "You are a test Sorcar.\n" and "applied" not in report

        daemon.patch = _lab_patch(  # git applies a header-less section like any other
            root / "tmp" / "lab-traditional", {REL_SEA: DEMO_SEA}, {REL_SEA: _add_extra}
        )
        daemon.patch.write_text(
            daemon.patch.read_text()
            + "--- a/src/kiss/SYSTEM.md\n+++ b/src/kiss/SYSTEM.md\n@@ -1 +1 @@\n"
            "-You are a test Sorcar.\n+changed\n"
        )
        report = _improve(run_id)
        assert report["error"] == refused + "['src/kiss/SYSTEM.md']" + restored
        assert system_md.read_text() == "You are a test Sorcar.\n" and "applied" not in report

        helper = f"{REL_SEAS_DIR}/tunedemo/helper.py"  # the replays never ran a lab helper
        daemon.patch = _lab_patch(
            root / "tmp" / "lab-helper", {REL_SEA: DEMO_SEA},
            {REL_SEA: _add_extra, helper: lambda p: p.write_text("X = 1\n")},
        )
        report = _improve(run_id)
        assert report["error"] == refused + f"['{helper}']" + restored
        assert not (root / helper).exists() and "applied" not in report

        renamed = f"{REL_SEAS_DIR}/tunedemo/other_sea.py"  # both sides of a rename count
        daemon.patch = _lab_patch(
            root / "tmp" / "lab-rename", {REL_SEA: DEMO_SEA},
            {REL_SEA: lambda p: p.rename(p.with_name("other_sea.py"))},
        )
        assert "rename to" in daemon.patch.read_text()
        report = _improve(run_id)
        assert report["error"] == refused + f"['{renamed}']" + restored
        assert not (root / renamed).exists() and "applied" not in report

        daemon.patch = _lab_patch(
            root / "tmp" / "lab-broken", {REL_SEA: DEMO_SEA}, {REL_SEA: _break_settings}
        )
        report = _improve(run_id)
        assert report["error"].startswith("the patched SEA no longer loads (")
        assert report["error"].endswith(restored) and report["applied"] == [REL_SEA]

        daemon.patch = _lab_patch(  # made against a prompt line this checkout does not have
            root / "tmp" / "lab-stale",
            {REL_SEA: DEMO_SEA.replace("demo tasks", "other tasks")},
            {REL_SEA: lambda p: p.write_text(p.read_text().replace("other tasks", "other jobs"))},
        )
        report = _improve(run_id)
        assert report["error"].startswith("best.patch does not apply to this checkout: ")
        assert "applied" not in report

        daemon.patch = _lab_patch(
            root / "tmp" / "lab-delete", {REL_SEA: DEMO_SEA}, {REL_SEA: lambda p: p.unlink()}
        )
        report = _improve(run_id)
        assert report["error"] == "the patch deleted the script" + restored
        assert report["applied"] == [REL_SEA]
    finally:
        daemon.close()
    assert len(daemon.run_commands) == 8
    assert (root / REL_SEA).read_text() == DEMO_SEA
    assert (root / REL_TEST).read_text() == DEMO_TEST
    assert git(root, "status", "--porcelain") == ""


def test_patch_named_in_the_report_and_an_unreachable_daemon(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch the child reports by path (outside its scratch dir) is found; an unreachable
    daemon is a reported failure that leaves the folder as it was."""
    run_id = _past_run(root)
    patch = _lab_patch(root / "tmp" / "lab-named", {REL_SEA: DEMO_SEA}, {REL_SEA: _add_extra})
    named = patch.relative_to(root)  # a checkout-relative path, as a child reports it
    daemon = _DiscoverDaemon(root, text=f"Best point B1 (dev: success, $1.10). Patch: {named}")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        report = _improve(run_id)
    finally:
        daemon.close()
    assert report["patch"] == str(patch) and "error" not in report, report
    assert "def extra" in (root / REL_SEA).read_text()
    assert sea.revert_sea_code("tunedemo").startswith("Restored ")
    assert (root / REL_SEA).read_text() == DEMO_SEA

    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(root / "no-daemon.json"))
    report = _improve(run_id)
    assert report["task_id"] == "" and "could not run" in report["result"]
    assert report["error"] == (
        "the child reported no best.patch; folder restored from the snapshot"
    )
    assert report["replays"] == [] and (root / REL_SEA).read_text() == DEMO_SEA


def test_created_test_file_held_out_cap_and_an_interrupted_gate(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SEA with no test file yet: the test file the patch creates is applied and removed again
    by the revert; the held-out replay gets the cap the caller names; a Stop raised while the
    gate loads the patched script restores the script and removes the created test file before
    it propagates."""
    (root / REL_TEST).unlink()
    commit(root, "no test yet")
    run_id = _past_run(root)
    heldout_id = _past_run(root, cost=2.0)
    daemon = _DiscoverDaemon(root)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    daemon.patch = _lab_patch(
        root / "tmp" / "lab-new-test", {REL_SEA: DEMO_SEA},
        {REL_SEA: _add_extra, REL_TEST: lambda p: p.write_text(DEMO_TEST)},
    )
    try:
        report = _improve(run_id, heldout_id, heldout_budget=12.5)
        assert "error" not in report and report["applied"] == [REL_SEA, REL_TEST], report
        assert (root / REL_TEST).read_text() == DEMO_TEST
        assert "def extra" in (root / REL_SEA).read_text()
        # ``diff_stat`` covers tracked files; the created test file is in ``applied`` above.
        assert report["diff_stat"].endswith("1 file changed, 4 insertions(+)")
        assert report["other_changes"] == []
        prompt = daemon.run_commands[0]["prompt"]
        assert f"its test file is `{root / REL_TEST}` (when it exists)" in prompt
        assert 'max_budget="8",\n  timeout="3600"' in prompt  # the dev cap
        assert 'max_budget="12.5"; run it on the point that meets the goal' in prompt
        assert sea.revert_sea_code("tunedemo").startswith("Restored ")
        assert not (root / REL_TEST).exists() and (root / REL_SEA).read_text() == DEMO_SEA
        assert git(root, "status", "--porcelain") == ""

        daemon.patch = _lab_patch(
            root / "tmp" / "lab-stopped", {REL_SEA: DEMO_SEA}, {
                REL_SEA: lambda p: p.write_text(STOPPED_SEA),
                REL_TEST: lambda p: p.write_text(DEMO_TEST),
            },
        )
        with pytest.raises(ToolCallInterrupted, match="stopped"):
            _improve(run_id)
    finally:
        daemon.close()
    assert len(daemon.run_commands) == 2
    assert not (root / REL_TEST).exists() and (root / REL_SEA).read_text() == DEMO_SEA
    assert git(root, "status", "--porcelain") == ""
