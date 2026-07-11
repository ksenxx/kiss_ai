# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: prompt / system-prompt panels must appear at submit time.

Issue reproduced here
---------------------
When a task is submitted to the chat webview, the authoritative
``prompt`` / ``system_prompt`` events are emitted deep inside
``KISSAgent.run`` — AFTER the git pre-snapshot, worktree creation,
chat-context loading and model/tool setup — so the webview shows an
empty tab for the first seconds after submit.

The fix (``_TaskRunnerMixin._broadcast_early_prompts``) broadcasts
optimistic ``early``-flagged panels the moment the task thread starts,
scoped to the launching tab and excluded from recording/persistence so
replayed sessions still contain exactly one authoritative pair.

The agent stack is real (``_run_task`` → ``WorktreeSorcarAgent.run`` →
``ChatSorcarAgent.run``); only the innermost LLM-driven ``run`` (the
grandparent of ``SorcarAgent``) is replaced — it sleeps to simulate the
slow model/tool setup that precedes the authoritative prompt events,
then emits those authoritative events exactly like ``KISSAgent.run``
does (``system_prompt`` first, then ``prompt``).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models

# Simulated duration of the slow pre-prompt setup (model construction,
# tool-schema build, model.initialize) that the REAL inner agent
# performs before printing the authoritative prompt events.
_SLOW_SETUP_S = 1.5
# The early panels must appear well before the slow setup completes.
_EARLY_DEADLINE_S = 1.0


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    """Create a ``VSCodeServer`` whose broadcasts mirror ``WebPrinter``.

    Events with an explicit ``tabId`` are captured verbatim; events
    with a (thread-local) task id are recorded, persisted, and fanned
    out once per subscribed tab with the viewer's ``tabId`` stamped.
    """
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        if "tabId" in event:
            with lock:
                events.append(event)
            return
        ev = printer._inject_task_id(event)
        if not ev.get("taskId"):
            with lock:
                events.append(ev)
            return
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        for tab_id in printer._fanout_targets(ev.get("taskId")):
            with lock:
                events.append({**ev, "tabId": tab_id})

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


def _patch_grandparent_run() -> Any:
    """Replace the LLM-driven grandparent ``run`` with a slow stub.

    ``_run_task_inner`` (which owns the early-prompt broadcast under
    test), ``WorktreeSorcarAgent.run`` and ``ChatSorcarAgent.run`` stay
    REAL.  The stub reproduces the REAL inner-agent timeline: a slow
    setup phase (``_SLOW_SETUP_S``) followed by the authoritative
    ``system_prompt`` / ``prompt`` events, exactly the emission order
    of ``KISSAgent.run``.
    """
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: Any, **kwargs: Any) -> str:
        printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
        time.sleep(_SLOW_SETUP_S)
        if printer is not None:
            printer.broadcast({
                "type": "system_prompt",
                "text": kwargs.get("system_prompt", "") or "authoritative-sys",
            })
            printer.broadcast({
                "type": "prompt",
                "text": kwargs.get("prompt_template", "") or "authoritative-prompt",
            })
        return str(yaml.dump({"success": True, "summary": "done"}))

    parent.run = _run_proxy
    return original


def _unpatch_grandparent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


class TestPromptPanelsEarly(unittest.TestCase):
    """Prompt / system-prompt panels must be broadcast at submit time."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.original_run = _patch_grandparent_run()

    def tearDown(self) -> None:
        _unpatch_grandparent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _events_snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)

    def _submit(self, tab_id: str, prompt: str,
                use_worktree: bool = False) -> None:
        self.server._handle_command({
            "type": "run", "prompt": prompt, "model": self.model,
            "workDir": self.tmpdir, "tabId": tab_id, "autoCommit": True,
            "useWorktree": use_worktree,
        })

    def _join_task(self, tab_id: str) -> None:
        t = self.server._get_tab(tab_id).task_thread
        assert t is not None
        t.join(timeout=60)
        assert not t.is_alive()

    def _wait_for_early_pair(
        self, tab_id: str, deadline_s: float,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Poll until both early panels for *tab_id* arrive or timeout."""
        deadline = time.monotonic() + deadline_s
        early_prompt: dict[str, Any] | None = None
        early_sys: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            for ev in self._events_snapshot():
                if not ev.get("early") or ev.get("tabId") != tab_id:
                    continue
                if ev.get("type") == "prompt":
                    early_prompt = ev
                elif ev.get("type") == "system_prompt":
                    early_sys = ev
            if early_prompt is not None and early_sys is not None:
                break
            time.sleep(0.02)
        return early_prompt, early_sys

    def test_prompt_panels_appear_before_slow_setup_finishes(self) -> None:
        """Reproduction: with a slow pre-prompt setup, both panels must
        still be broadcast to the launching tab within a fraction of
        the setup time (pre-fix: NO prompt event exists until the inner
        agent finishes its setup, seconds after submit)."""
        tab = "tab-early"
        user_prompt = "early-panel user prompt"
        self._submit(tab, user_prompt)
        try:
            early_prompt, early_sys = self._wait_for_early_pair(
                tab, _EARLY_DEADLINE_S,
            )
            snapshot = self._events_snapshot()
            assert early_prompt is not None, (
                "no early 'prompt' event within "
                f"{_EARLY_DEADLINE_S}s of submit: {snapshot}"
            )
            assert early_sys is not None, (
                "no early 'system_prompt' event within "
                f"{_EARLY_DEADLINE_S}s of submit: {snapshot}"
            )
            # The early panel shows the user's prompt verbatim.
            assert early_prompt.get("text") == user_prompt
            # The early system prompt is non-empty (SYSTEM_PROMPT text).
            assert str(early_sys.get("text") or "").strip()
            # The slow setup is still running: the AUTHORITATIVE events
            # must not have arrived yet — the early pair is the only
            # reason the panels are visible this soon.
            authoritative = [
                e for e in snapshot
                if e.get("type") in ("prompt", "system_prompt")
                and not e.get("early")
            ]
            assert not authoritative, (
                f"authoritative events arrived too soon: {authoritative}"
            )
        finally:
            self._join_task(tab)
        # After the run, the authoritative pair (emitted by the inner
        # agent) reached the tab as well — the frontend replaces the
        # early panels with these in place.
        post = self._events_snapshot()
        auth_types = {
            e.get("type") for e in post
            if e.get("type") in ("prompt", "system_prompt")
            and not e.get("early") and e.get("tabId") == tab
        }
        assert auth_types == {"prompt", "system_prompt"}, f"got: {post}"

    def test_prompt_panels_precede_worktree_creation(self) -> None:
        """Ordering reproduction: in worktree mode the early panels must
        be broadcast BEFORE the (slow) worktree setup completes
        (pre-fix: the first prompt event trails ``worktree_created``)."""
        tab = "tab-wt"
        self._submit(tab, "worktree ordering prompt", use_worktree=True)
        self._join_task(tab)
        events = self._events_snapshot()
        types = [
            e.get("type") for e in events
            if e.get("type") in ("prompt", "system_prompt",
                                 "worktree_created")
        ]
        assert "worktree_created" in types, f"no worktree run: {events}"
        wt_idx = types.index("worktree_created")
        assert "prompt" in types[:wt_idx] and "system_prompt" in types[:wt_idx], (
            "prompt panels were not broadcast before worktree creation: "
            f"{types}"
        )

    def test_early_prompt_events_are_not_persisted(self) -> None:
        """Replay correctness: the persisted event stream must contain
        ONLY the authoritative prompt pair — never the early copies
        (which have no task row and would duplicate panels on replay)."""
        tab = "tab-persist"
        self._submit(tab, "persistence prompt")
        self._join_task(tab)
        th._flush_chat_events()
        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            rows = conn.execute("SELECT event_json FROM events").fetchall()
        finally:
            conn.close()
        persisted = [json.loads(r[0]) for r in rows]
        early = [e for e in persisted if e.get("early")]
        assert not early, f"early events must never be persisted: {early}"
        prompt_types = [
            e.get("type") for e in persisted
            if e.get("type") in ("prompt", "system_prompt")
        ]
        assert sorted(prompt_types) == ["prompt", "system_prompt"], (
            "persisted stream must contain exactly one authoritative "
            f"pair, got: {prompt_types}"
        )


if __name__ == "__main__":
    unittest.main()
