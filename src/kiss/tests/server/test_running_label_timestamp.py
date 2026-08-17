# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_running_label_timestamp``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import re
from pathlib import Path

import kiss.server


def test_task_runner_persists_start_end_ts_to_extra():
    """``_save_task_extra`` must persist both timestamps to the
    ``extra`` JSON column so a later history load can flip "Running …"
    to "Done (…)" without a live ``task_done`` event.  The payload is
    built by :func:`build_task_extra_payload`, so verify the returned
    dict carries ``startTs`` and ``endTs`` end-to-end."""
    from kiss.server.task_runner import build_task_extra_payload

    payload = build_task_extra_payload(
        model="m",
        work_dir="/repo",
        version="v",
        tokens=0,
        cost=0.0,
        steps=0,
        is_parallel=False,
        is_worktree=False,
        auto_commit_mode=False,
        start_ms=123,
        end_ms=456,
    )
    assert payload["startTs"] == 123, "payload must carry startTs"
    assert payload["endTs"] == 456, "payload must carry endTs"


# Placement-independent: resolve the server sources from the package
# rather than from this test file's location.
TASK_RUNNER = Path(kiss.server.__file__).parent / "task_runner.py"
SERVER_PY = Path(kiss.server.__file__).parent / "server.py"


def _read(p: Path) -> str:
    assert p.is_file(), f"missing: {p}"
    return p.read_text()


def test_task_runner_broadcasts_start_ts_on_status_running():
    """``_run_task`` must include ``startTs`` (agent's true start time,
    ms since epoch) on the ``status: running=True`` broadcast so the
    frontend can anchor its timer to the agent clock."""
    src = _read(TASK_RUNNER)
    m = re.search(r"def _run_task\(self, cmd: dict\[str, Any\]\) -> None:",
                  src)
    assert m, "_run_task not found"
    rest = src[m.end():]
    next_def = re.search(r"\n    def\s", rest)
    body = rest[: next_def.start()] if next_def else rest
    assert '"running": True' in body, (
        "expected the running=True broadcast in _run_task body"
    )
    assert '"startTs"' in body, (
        "_run_task must add startTs (ms since epoch) to the "
        "status: running=True broadcast"
    )


def test_task_runner_broadcasts_end_ts_on_task_end():
    """The ``task_end_event`` broadcast must carry ``startTs`` and
    ``endTs`` so the frontend can compute the duration without
    relying on its own wall clock."""
    src = _read(TASK_RUNNER)
    assert "**task_end_event" in src, (
        "expected broadcast({**task_end_event, ...}) site"
    )
    for m in re.finditer(
        r"\.broadcast\(\{\*\*task_end_event[^}]*\}\)", src,
    ):
        chunk = m.group(0)
        assert '"startTs"' in chunk, (
            "task_end_event broadcast must include startTs"
        )
        assert '"endTs"' in chunk, (
            "task_end_event broadcast must include endTs"
        )


def test_get_history_emits_start_ts_per_session():
    """Every history row must carry ``startTs`` (from the row's
    ``timestamp`` column converted to ms) and ``endTs`` (from the
    persisted ``extra.endTs`` or 0 if still running)."""
    src = _read(SERVER_PY)
    m = re.search(r"def _get_history\(\s*self,", src)
    assert m, "_get_history not found"
    rest = src[m.end():]
    next_def = re.search(r"\n    def\s", rest)
    body = rest[: next_def.start()] if next_def else rest
    assert 'session["startTs"]' in body, (
        "_get_history must set session['startTs'] (ms since epoch)"
    )
    assert 'session["endTs"]' in body, (
        "_get_history must set session['endTs'] (ms since epoch or 0)"
    )
