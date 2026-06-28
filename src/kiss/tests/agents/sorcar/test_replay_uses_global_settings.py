# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test: loading a chat into a tab must NOT stamp the live toggle UI
with the loaded task's per-task snapshot of global settings.

Spec
----
1. A first task T1 runs in chat X with the global settings at the time
   it was launched (``is_worktree=True``, ``is_parallel=True``,
   ``auto_commit_mode=True``, ``model="claude-opus-4-6"``).  The
   :meth:`_TaskRunnerMixin._run_task_inner` cleanup persists this
   snapshot into the ``task_history.extra`` JSON column.

2. The user changes their GLOBAL settings — typically by clicking the
   worktree/parallel/auto-commit toggles in the toolbar — to a new
   configuration (``is_worktree=False``, etc.).  The toggle state
   in the webview is the source of truth for the next task's
   ``useWorktree`` / ``useParallel`` / ``autoCommit`` flags
   (see ``media/main.js`` ``submit`` handler around the
   ``useWorktree: !!(worktreeToggleBtn && worktreeToggleBtn.checked)``
   block).

3. The user clicks the history row for T1.  The frontend issues
   ``newChat`` then ``resumeSession`` for chat X into a fresh tab Y.
   The backend's :meth:`_replay_session` broadcasts a single
   ``task_events`` event whose ``extra`` payload is the persisted
   ``task_history.extra`` JSON of T1.

4. PRIOR BEHAVIOR (the bug): ``extra`` carried ``is_worktree``,
   ``is_parallel``, ``auto_commit_mode`` and ``model`` keys.  The
   frontend's ``task_events`` handler used those values to overwrite
   the live toggle UI and ``selectedModel``.  The next ``submit`` in
   tab Y then sent those STALE values as ``useWorktree`` /
   ``useParallel`` / ``autoCommit`` / ``model`` — silently making
   the follow-up task run with T1's old settings instead of the
   user's CURRENT global settings.

5. FIXED BEHAVIOR: the backend strips the four global-setting keys
   from the broadcast ``extra`` (see ``_extra_for_replay`` in
   ``kiss/agents/vscode/server.py``).  The webview's live toggles
   stay at the user's current global settings, and the follow-up
   ``submit`` in tab Y carries those settings to the next run.

The test patches the underlying ``SorcarAgent.run`` so no real LLM
call is made — only the orchestration is exercised end-to-end.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


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
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        # Mirror the real ``JsonPrinter.broadcast`` side effects so
        # persistence and per-task recording see the event.
        ev = printer._inject_task_id(event)
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        with lock:
            events.append(ev)

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class _RunCapture:
    """Capture the kwargs passed to ``SorcarAgent.run`` and return a
    canned YAML success result so the task completes synchronously."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.summaries = ["first done", "second done"]

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        printer = kwargs.get("printer")
        summary = self.summaries.pop(0) if self.summaries else "done"
        if printer is not None and hasattr(printer, "broadcast"):
            cast(Any, printer).broadcast({
                "type": "result",
                "text": summary,
                "success": True,
            })
        return f"success: true\nsummary: {summary}\n"


def _patch_parent_run(capture: _RunCapture) -> Any:
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: object, **kwargs: object) -> str:
        return capture.fake_run(self_agent, **kwargs)

    parent.run = _run_proxy
    return original


def _unpatch_parent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


def _run_and_wait(
    server: VSCodeServer,
    *,
    tab_id: str,
    prompt: str,
    work_dir: str,
    model: str,
    use_worktree: bool,
    use_parallel: bool,
    auto_commit: bool,
) -> None:
    server._handle_command({
        "type": "run",
        "prompt": prompt,
        "model": model,
        "workDir": work_dir,
        "tabId": tab_id,
        "useWorktree": use_worktree,
        "useParallel": use_parallel,
        "autoCommit": auto_commit,
    })
    t = server._get_tab(tab_id).task_thread
    assert t is not None
    t.join(timeout=10)
    assert not t.is_alive()


class TestReplayUsesGlobalSettings(unittest.TestCase):
    """The backend's ``_replay_session`` broadcast must NOT instruct
    the frontend to stamp the live toggle / model state with the
    loaded task's per-task snapshot."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.capture = _RunCapture()
        self.original_run = _patch_parent_run(self.capture)

    def tearDown(self) -> None:
        _unpatch_parent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _persisted_extra_for(self, task_id: str) -> dict[str, object]:
        """Return the persisted ``task_history.extra`` JSON for *task_id*."""
        rows = th._load_history(limit=10)
        for row in rows:
            if cast(str, row["id"]) == task_id:
                raw = row.get("extra", "") or ""
                if isinstance(raw, str) and raw:
                    return cast(dict[str, object], json.loads(raw))
                break
        return {}

    def _broadcasted_task_events(self) -> dict[str, object] | None:
        """Return the LAST ``task_events`` broadcast captured by the
        test printer.  Used to inspect what ``_replay_session`` sends
        the frontend on a chat-load."""
        with self.lock:
            for ev in reversed(self.events):
                if ev.get("type") == "task_events":
                    return ev
        return None

    def test_replay_strips_global_setting_keys_from_extra(self) -> None:
        # Step 1: run a first task with the old global settings.
        first_tab = "tab-first"
        _run_and_wait(
            self.server,
            tab_id=first_tab,
            prompt="remember the magic word: banana",
            work_dir=self.tmpdir,
            model="claude-opus-4-6",
            use_worktree=True,
            use_parallel=True,
            auto_commit=True,
        )
        first_row = th._load_history(limit=10)[0]
        chat_id = str(first_row["chat_id"])
        first_task_id = cast(str, first_row["id"])
        assert chat_id, "first task should have a chat id"

        # Sanity-check the persisted snapshot — the bug is only
        # interesting BECAUSE these keys live in ``extra``.
        persisted = self._persisted_extra_for(first_task_id)
        assert persisted.get("is_worktree") is True, persisted
        assert persisted.get("is_parallel") is True, persisted
        assert persisted.get("auto_commit_mode") is True, persisted
        assert persisted.get("model") == "claude-opus-4-6", persisted

        # Clear captured events so the replay broadcast is unambiguous.
        with self.lock:
            self.events.clear()

        # Step 2: user clicks the history row for T1 into a fresh tab.
        history_tab = "tab-history"
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab},
        )
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": first_task_id,
            "tabId": history_tab,
        })

        # Step 3: inspect the broadcast ``task_events``.  Its ``extra``
        # must NOT carry the four global-setting keys, so the
        # frontend's ``task_events`` handler cannot stamp the live
        # toggles / model with the loaded task's stale snapshot.
        replay = self._broadcasted_task_events()
        assert replay is not None, "expected a task_events broadcast on resume"
        extra_str = cast(str, replay.get("extra", ""))
        assert extra_str, "task_events.extra should not be empty"
        extra_json = json.loads(extra_str)
        assert isinstance(extra_json, dict), extra_json
        for stripped in (
            "is_worktree", "is_parallel", "auto_commit_mode", "model",
        ):
            assert stripped not in extra_json, (
                f"replay broadcast leaked stripped key {stripped!r}: "
                f"{extra_json!r}"
            )
        # Keys we DO want to keep for the chat header / timer / tab
        # work-dir routing must survive the strip.
        assert extra_json.get("work_dir") == self.tmpdir, extra_json
        assert int(cast(int, extra_json.get("startTs", 0))) > 0, extra_json
        assert int(cast(int, extra_json.get("endTs", 0))) > 0, extra_json

    def test_followup_task_uses_new_global_settings(self) -> None:
        # Step 1: run a first task under the OLD global settings.
        first_tab = "tab-first"
        _run_and_wait(
            self.server,
            tab_id=first_tab,
            prompt="task 1: remember the magic word: banana",
            work_dir=self.tmpdir,
            model="claude-opus-4-6",
            use_worktree=True,
            use_parallel=True,
            auto_commit=True,
        )
        first_row = th._load_history(limit=10)[0]
        chat_id = str(first_row["chat_id"])
        first_task_id = cast(str, first_row["id"])

        # Step 2: user clicks the history row for T1 → fresh tab.
        history_tab = "tab-history"
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab},
        )
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": first_task_id,
            "tabId": history_tab,
        })

        # The replay must NOT have seeded ``tab.use_worktree`` with
        # the loaded task's snapshot — otherwise a follow-up run that
        # forgot to pass ``useWorktree`` would silently inherit the
        # stale True from the loaded T1.  The new global setting is
        # ``False`` (the user toggled it off after T1 finished).
        loaded_tab = self.server._get_tab(history_tab)
        assert not loaded_tab.use_worktree, (
            "loading a chat must not stamp tab.use_worktree from the "
            "loaded task's historical snapshot"
        )

        # Step 3: the user submits a follow-up under the NEW global
        # settings (toggles all flipped off, model swapped).
        _run_and_wait(
            self.server,
            tab_id=history_tab,
            prompt="task 2: what was the magic word?",
            work_dir=self.tmpdir,
            model="claude-sonnet-4-5",
            use_worktree=False,
            use_parallel=False,
            auto_commit=False,
        )

        # The second persisted task's extras must reflect the NEW
        # global settings.  ``_load_history`` returns rows in
        # most-recent-first order so the follow-up is row 0.
        rows = th._load_history(limit=10)
        assert len(rows) >= 2, rows
        second_task_id = cast(str, rows[0]["id"])
        assert second_task_id != first_task_id
        second_extra = self._persisted_extra_for(second_task_id)
        assert second_extra.get("is_worktree") is False, second_extra
        assert second_extra.get("is_parallel") is False, second_extra
        assert second_extra.get("auto_commit_mode") is False, second_extra
        assert second_extra.get("model") == "claude-sonnet-4-5", second_extra


    def test_history_load_resets_use_parallel_and_auto_commit(self) -> None:
        """Symmetric to ``test_followup_task_uses_new_global_settings``
        for the parallel / auto-commit toggles.  A history load must
        leave ``tab.use_parallel`` and ``tab.auto_commit_mode`` at the
        baseline (``False``), so a follow-up run that forgets to pass
        ``useParallel`` / ``autoCommit`` cannot silently inherit the
        loaded task's stale ``True``."""
        first_tab = "tab-first-pa"
        _run_and_wait(
            self.server,
            tab_id=first_tab,
            prompt="task with parallel + auto-commit enabled",
            work_dir=self.tmpdir,
            model="claude-opus-4-6",
            use_worktree=True,
            use_parallel=True,
            auto_commit=True,
        )
        first_row = th._load_history(limit=10)[0]
        chat_id = str(first_row["chat_id"])
        first_task_id = cast(str, first_row["id"])

        history_tab = "tab-history-pa"
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab},
        )
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": first_task_id,
            "tabId": history_tab,
        })

        loaded_tab = self.server._get_tab(history_tab)
        assert not loaded_tab.use_worktree, loaded_tab.use_worktree
        assert not loaded_tab.use_parallel, loaded_tab.use_parallel
        assert not loaded_tab.auto_commit_mode, loaded_tab.auto_commit_mode

    def test_history_load_resets_selected_model(self) -> None:
        """A history load must reset ``tab.selected_model`` to the
        server's default (the persisted last-picked global model),
        not leave it at whatever the previous chat in the same tab
        was using.  Without the reset, ``_get_history``'s
        running-session sidebar would read the stale per-tab model
        for any LIVE task in the newly-loaded chat."""
        first_tab = "tab-first-model"
        _run_and_wait(
            self.server,
            tab_id=first_tab,
            prompt="task to create a chat",
            work_dir=self.tmpdir,
            model=self.server._default_model,
            use_worktree=False,
            use_parallel=False,
            auto_commit=False,
        )
        first_row = th._load_history(limit=10)[0]
        chat_id = str(first_row["chat_id"])
        first_task_id = cast(str, first_row["id"])

        # Now load that chat into a brand-new tab.  We seed
        # ``selected_model`` on the tab to a non-default value
        # AFTER ``newChat`` (which itself resets it), to simulate
        # a tab that had run a task with a non-default model in
        # the past.  ``_replay_session`` (with the fix) must reset
        # it back to the server default.
        history_tab = "tab-history-model"
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab},
        )
        stale_model = "stale-test-model"
        self.server._get_tab(history_tab).selected_model = stale_model
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": first_task_id,
            "tabId": history_tab,
        })

        loaded_tab = self.server._get_tab(history_tab)
        assert (
            loaded_tab.selected_model == self.server._default_model
        ), loaded_tab.selected_model
        # Defensive: must not be the stale value we seeded.
        assert loaded_tab.selected_model != stale_model

    def test_history_load_preserves_state_during_merge_review(self) -> None:
        """Regression for the round-2 gpt-5.5 review finding: the
        ``tab_alive`` guard must cover the ``tab.is_merging`` flag
        too.  Without it, a ``_replay_session`` re-entry on a tab
        whose post-task worktree-merge prompt is still open (e.g.
        VS Code window reload mid-merge) would clobber
        ``tab.use_worktree=False``, causing ``_finish_merge`` /
        ``_present_pending_worktree`` to short-circuit and leak
        the worktree directory."""
        tab_id = "tab-merging"
        _run_and_wait(
            self.server,
            tab_id=tab_id,
            prompt="seed task for merge guard",
            work_dir=self.tmpdir,
            model=self.server._default_model,
            use_worktree=True,
            use_parallel=True,
            auto_commit=True,
        )
        row = th._load_history(limit=10)[0]
        chat_id = str(row["chat_id"])
        task_id = cast(str, row["id"])

        tab = self.server._get_tab(tab_id)
        # Simulate mid-merge: task finished, merge review still open.
        tab.is_task_active = False
        tab.is_merging = True
        tab.use_worktree = True
        tab.use_parallel = True
        tab.auto_commit_mode = True
        tab.selected_model = "stale-merge-model"

        try:
            self.server._replay_session(
                chat_id=chat_id, tab_id=tab_id, task_id=task_id,
            )
            assert tab.use_worktree is True, tab.use_worktree
            assert tab.use_parallel is True, tab.use_parallel
            assert tab.auto_commit_mode is True, tab.auto_commit_mode
            assert (
                tab.selected_model == "stale-merge-model"
            ), tab.selected_model
        finally:
            tab.is_merging = False

    def test_history_load_preserves_state_when_thread_alive(self) -> None:
        """Cover the ``task_thread.is_alive()`` arm of the
        ``_tab_busy`` predicate inside ``_replay_session``."""
        tab_id = "tab-thread"
        _run_and_wait(
            self.server,
            tab_id=tab_id,
            prompt="seed task for thread guard",
            work_dir=self.tmpdir,
            model=self.server._default_model,
            use_worktree=True,
            use_parallel=False,
            auto_commit=False,
        )
        row = th._load_history(limit=10)[0]
        chat_id = str(row["chat_id"])
        task_id = cast(str, row["id"])

        tab = self.server._get_tab(tab_id)
        stop_evt = threading.Event()
        alive_thread = threading.Thread(target=stop_evt.wait, daemon=True)
        alive_thread.start()
        tab.is_task_active = False
        tab.is_merging = False
        tab.task_thread = alive_thread
        tab.use_worktree = True
        tab.use_parallel = True
        tab.auto_commit_mode = True
        tab.selected_model = "stale-thread-model"

        try:
            self.server._replay_session(
                chat_id=chat_id, tab_id=tab_id, task_id=task_id,
            )
            assert tab.use_worktree is True
            assert tab.use_parallel is True
            assert tab.auto_commit_mode is True
            assert tab.selected_model == "stale-thread-model"
        finally:
            stop_evt.set()
            alive_thread.join(timeout=2)

    def test_history_load_preserves_state_during_live_run(self) -> None:
        """The reset added to ``_replay_session`` is gated by a
        ``tab_alive`` guard: when the loaded chat is being re-rendered
        into the SAME tab that owns the live run, the in-flight
        per-tab fields are the source of truth for
        ``_get_history``'s sidebar metadata and must not be
        clobbered.  Reproduce by simulating an active task (set
        ``is_task_active = True``) on the tab BEFORE calling
        ``_replay_session``, then assert the fields are unchanged."""
        tab_id = "tab-live"
        # Spin up a chat by running a task to completion so a
        # task_history row exists for the replay.
        _run_and_wait(
            self.server,
            tab_id=tab_id,
            prompt="seed task to create chat",
            work_dir=self.tmpdir,
            model="claude-opus-4-6",
            use_worktree=True,
            use_parallel=True,
            auto_commit=True,
        )
        row = th._load_history(limit=10)[0]
        chat_id = str(row["chat_id"])
        task_id = cast(str, row["id"])

        tab = self.server._get_tab(tab_id)
        # Simulate an in-flight run by flipping the "alive" flag.
        # ``_replay_session``'s ``tab_alive`` guard reads both
        # ``is_task_active`` and ``task_thread.is_alive``; setting
        # the flag is the minimum repro.
        tab.is_task_active = True
        tab.use_worktree = True
        tab.use_parallel = True
        tab.auto_commit_mode = True
        tab.selected_model = "claude-opus-4-6"

        try:
            self.server._replay_session(
                chat_id=chat_id, tab_id=tab_id, task_id=task_id,
            )
            # The guard prevents the reset, so the live values
            # survive.
            assert tab.use_worktree is True, tab.use_worktree
            assert tab.use_parallel is True, tab.use_parallel
            assert tab.auto_commit_mode is True, tab.auto_commit_mode
            assert (
                tab.selected_model == "claude-opus-4-6"
            ), tab.selected_model
        finally:
            tab.is_task_active = False


class TestExtraForReplayUnit(unittest.TestCase):
    """Unit-level edge cases for ``_extra_for_replay``.  Mirrors the
    docstring contract; the round-1 gpt-5.5 review flagged the
    non-dict pass-through as a defensive hole that has now been
    fixed to return ``""``."""

    def test_non_dict_json_returns_empty(self) -> None:
        from kiss.agents.vscode.server import _extra_for_replay
        for malformed in ('[1,2,3]', '"x"', '42', 'null', 'true'):
            assert _extra_for_replay(malformed) == "", malformed

    def test_dict_with_stripped_keys_removes_them(self) -> None:
        from kiss.agents.vscode.server import _extra_for_replay
        payload = json.dumps({
            "model": "claude-opus-4-6",
            "is_worktree": True,
            "is_parallel": True,
            "auto_commit_mode": True,
            "work_dir": "/tmp/x",
            "startTs": 1, "endTs": 2,
        })
        out = _extra_for_replay(payload)
        parsed = json.loads(out)
        assert "model" not in parsed
        assert "is_worktree" not in parsed
        assert "is_parallel" not in parsed
        assert "auto_commit_mode" not in parsed
        assert parsed["work_dir"] == "/tmp/x"
        assert parsed["startTs"] == 1
        assert parsed["endTs"] == 2

    def test_dict_without_stripped_keys_passes_through(self) -> None:
        from kiss.agents.vscode.server import _extra_for_replay
        payload = json.dumps({"work_dir": "/tmp/x", "startTs": 1})
        # The implementation returns the ORIGINAL string when no
        # stripped key was present (skips re-serialization).
        out = _extra_for_replay(payload)
        assert json.loads(out) == json.loads(payload)

    def test_empty_or_non_string_returns_empty(self) -> None:
        from kiss.agents.vscode.server import _extra_for_replay
        assert _extra_for_replay("") == ""
        assert _extra_for_replay(None) == ""
        assert _extra_for_replay(123) == ""
        assert _extra_for_replay({"k": "v"}) == ""

    def test_invalid_json_returns_original(self) -> None:
        from kiss.agents.vscode.server import _extra_for_replay
        assert _extra_for_replay("not-json") == "not-json"


class TestSubagentReplayStripsGlobalSettings(unittest.TestCase):
    """Loading a parent task that fans out to sub-agents reopens every
    persisted sub-agent tab with its own ``task_events`` broadcast.
    Those broadcasts must also strip the global-setting keys from
    ``extra`` so a sub-agent's per-task snapshot of model / toggles
    cannot leak into the live model picker when the user later
    switches to the sub-tab (see the ``bgExtra`` branch of the
    frontend's ``task_events`` handler in ``media/main.js``)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        # No git repo needed: we seed task_history directly.

    def tearDown(self) -> None:
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_persisted_subagent_extras_are_stripped(self) -> None:
        chat_id = "chat-subs"
        # Seed a parent row with a persisted extra that contains the
        # stripped keys.
        parent_id, _ = th._add_task("parent", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "parent-stream"},
            task_id=parent_id,
        )
        th._save_task_extra(
            {
                "model": "claude-opus-4-6",
                "work_dir": self.tmpdir,
                "version": "test",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": True,
                "is_worktree": True,
                "auto_commit_mode": True,
                "startTs": 1000,
                "endTs": 2000,
            },
            task_id=parent_id,
        )
        # Seed two sub-agent rows whose extras also carry stripped
        # keys + the subagent metadata.
        sub_ids: list[str] = []
        for i in range(2):
            sub_id, _ = th._add_task(f"sub {i}", chat_id=chat_id)
            th._append_chat_event(
                {"type": "text_delta", "text": f"sub-{i}-stream"},
                task_id=sub_id,
            )
            th._save_task_extra(
                {
                    "model": "claude-opus-4-6",
                    "work_dir": self.tmpdir,
                    "version": "test",
                    "tokens": 0,
                    "cost": 0.0,
                    "is_parallel": False,
                    "is_worktree": True,
                    "auto_commit_mode": False,
                    "startTs": 1100 + i,
                    "endTs": 1900 + i,
                    "subagent": {"parent_task_id": parent_id},
                },
                task_id=sub_id,
            )
            sub_ids.append(sub_id)

        server, events, lock = _make_server()
        parent_tab = "tab-parent"
        server._replay_session(
            chat_id=chat_id, tab_id=parent_tab, task_id=parent_id,
        )

        # Filter task_events for sub-agent tabs.
        sub_tab_ids = {f"{parent_tab}__sub_{sid}" for sid in sub_ids}
        with lock:
            sub_task_events = [
                e for e in events
                if e.get("type") == "task_events"
                and e.get("tabId") in sub_tab_ids
            ]
        assert len(sub_task_events) == 2, sub_task_events

        for te in sub_task_events:
            extra_str = cast(str, te.get("extra", "") or "")
            assert extra_str, f"sub-tab task_events.extra empty: {te}"
            extra_json = json.loads(extra_str)
            assert isinstance(extra_json, dict), extra_json
            for stripped in (
                "is_worktree", "is_parallel", "auto_commit_mode", "model",
            ):
                assert stripped not in extra_json, (
                    f"sub-agent replay broadcast leaked stripped key "
                    f"{stripped!r}: {extra_json!r}"
                )
            # Preserved keys: startTs / endTs / work_dir survive so
            # the sub-tab's "Running …" / "Done" header keeps working,
            # and the subagent metadata is still present.
            assert extra_json.get("startTs", 0) > 0, extra_json
            assert extra_json.get("endTs", 0) > 0, extra_json
            assert extra_json.get("work_dir") == self.tmpdir, extra_json
            assert isinstance(extra_json.get("subagent"), dict), extra_json


if __name__ == "__main__":
    unittest.main()
