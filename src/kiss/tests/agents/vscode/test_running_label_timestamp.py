# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the Running / Done label at the top of a chat
webview must be computed from the agent's real start / end
timestamps — not from the client's wall clock at the moment the tab
or webview is loaded.

Two bugs were patched to enforce this invariant:

1. ``startTimer()`` in ``media/main.js`` used to anchor ``t0`` to
   ``Date.now()`` when no value was set, which meant a chat that
   started an hour ago and is then loaded from history showed
   "Running 0s" instead of "Running 1h 0s".  The fix routes the
   agent's real start timestamp through the backend ``status``
   event (``ev.startTs`` in ms since epoch) and through the
   history-list payload (``s.startTs`` per row) so the frontend
   knows the true elapsed time.

2. The ``task_done`` handler in ``media/main.js`` computed the
   completion duration from ``Date.now() - doneT0``; for tabs that
   joined late this is wrong.  The fix uses ``ev.endTs - ev.startTs``
   when both are present.  The ``endTs`` value is persisted into the
   task_history ``extra`` JSON column by ``_save_task_extra`` and
   replayed on history list / replay so the frontend can flip the
   "Running …" label to "Done (Xm Ys)" as soon as
   ``Date.now() >= endTs``, even when no live ``task_done`` event
   arrives (e.g. the task finished while no client was connected).
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.agents.vscode import web_server

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)
TASK_RUNNER = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "task_runner.py"
)
SERVER_PY = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "server.py"
)


def _read(p: Path) -> str:
    assert p.is_file(), f"missing: {p}"
    return p.read_text()


def _extract_case_body(src: str, case_name: str) -> str:
    """Return the body of ``case '<case_name>':`` in main.js."""
    m = re.search(rf"case\s+'{re.escape(case_name)}'\s*:\s*\{{", src)
    assert m, f"case '{case_name}' not found"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


def _extract_fn_body(src: str, fn_name: str) -> str:
    """Return the body of ``function <fn_name>(...) { ... }`` in main.js."""
    m = re.search(rf"function\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{", src)
    assert m, f"function '{fn_name}' not found"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


# ---------- Backend: task_runner broadcasts startTs / endTs ----------


def test_task_runner_broadcasts_start_ts_on_status_running():
    """``_run_task`` must include ``startTs`` (agent's true start time,
    ms since epoch) on the ``status: running=True`` broadcast so the
    frontend can anchor its timer to the agent clock."""
    src = _read(TASK_RUNNER)
    # Find the body of ``_run_task``.
    m = re.search(r"def _run_task\(self, cmd: dict\[str, Any\]\) -> None:",
                  src)
    assert m, "_run_task not found"
    # Slice forward to the next top-level def (4-space indent).
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
    # Find the broadcast site that spreads task_end_event.
    assert "**task_end_event" in src, (
        "expected broadcast({**task_end_event, ...}) site"
    )
    # All spreads must include both startTs and endTs.
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


def test_task_runner_persists_start_end_ts_to_extra():
    """``_save_task_extra`` must persist both timestamps to the
    ``extra`` JSON column so a later history load can flip "Running …"
    to "Done (…)" without a live ``task_done`` event."""
    src = _read(TASK_RUNNER)
    # The single _save_task_extra dict literal in task_runner.py.
    m = re.search(r"_save_task_extra\(\s*\{(.+?)\},", src, re.DOTALL)
    assert m, "_save_task_extra({...}) call not found"
    body = m.group(1)
    assert '"startTs"' in body, (
        "extra dict must include startTs"
    )
    assert '"endTs"' in body, (
        "extra dict must include endTs"
    )


# ---------- Backend: server._get_history surfaces startTs/endTs ----------


def test_get_history_emits_start_ts_per_session():
    """Every history row must carry ``startTs`` (from the row's
    ``timestamp`` column converted to ms) and ``endTs`` (from the
    persisted ``extra.endTs`` or 0 if still running)."""
    src = _read(SERVER_PY)
    # Slice the body of _get_history.  ``\s*`` tolerates the
    # multi-line signature (``conn_id`` pushed the params onto
    # their own lines).
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


# ---------- Frontend: status handler anchors t0 to ev.startTs ----------


def test_status_handler_anchors_t0_to_ev_start_ts():
    """The ``case 'status':`` handler must seed ``t0`` (and
    ``evTab.t0``) from ``ev.startTs`` before calling
    ``setRunningState(true)`` so the running label reflects the
    agent's actual elapsed time."""
    body = _extract_case_body(_read(MAIN_JS), "status")
    assert "ev.startTs" in body, (
        "case 'status' handler must consume ev.startTs"
    )
    # ``t0`` is assigned from the agent's startTs (not Date.now()).
    assert re.search(r"t0\s*=\s*ev\.startTs", body), (
        "case 'status' must set t0 = ev.startTs"
    )


# ---------- Frontend: timer tick flips Running → Done at endTs ----------


def test_timer_tick_flips_to_done_when_now_exceeds_end_ts():
    """``_renderTimerTick`` must check the agent's recorded end
    timestamp and switch the label to ``Done (Xm Ys)`` (computed
    from ``endTs - t0``) as soon as ``Date.now() >= endTs``."""
    body = _extract_fn_body(_read(MAIN_JS), "_renderTimerTick")
    assert "endTs" in body, (
        "_renderTimerTick must consult the agent's endTs"
    )
    assert "Done" in body, (
        "_renderTimerTick must emit a 'Done (…)' label when "
        "Date.now() >= endTs"
    )
    assert "Date.now()" in body and ">=" in body, (
        "_renderTimerTick must compare Date.now() >= endTs"
    )


# ---------- Frontend: task_done uses ev.endTs - ev.startTs ----------


def test_task_done_uses_event_start_and_end_ts():
    """The ``case 'task_done':`` handler must compute the duration
    from the agent-supplied ``ev.endTs - ev.startTs`` (not from the
    client wall clock at event arrival time)."""
    body = _extract_case_body(_read(MAIN_JS), "task_done")
    assert "ev.endTs" in body and "ev.startTs" in body, (
        "task_done must compute elapsed from ev.endTs - ev.startTs"
    )
    # Reject the old anchor on Date.now() - doneT0 as the sole path:
    # the new code may still fall back to it, but only when the agent
    # timestamps are missing.  Assert that a guarded ev.endTs path exists.
    assert re.search(
        r"ev\.endTs\s*-\s*ev\.startTs|ev\.startTs\s*&&\s*ev\.endTs",
        body,
    ), "task_done must subtract ev.startTs from ev.endTs"


# ---------- Sanity: built HTML still loads ----------


def test_built_html_loads_main_js():
    """Smoke check: the served HTML still references main.js so the
    changes above are actually delivered to the webview."""
    html = web_server._build_html()  # type: ignore[attr-defined]
    assert "main.js" in html
