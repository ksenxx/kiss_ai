# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: malformed non-string command fields (group E).

The websocket/UDS receive loops wrap the WHOLE ``async for`` in one
``try``, so any exception escaping
:meth:`VSCodeServer._handle_command` kills the ENTIRE client
connection (one VS Code window / browser tab).  Iteration 3 fixed
this class for an unhashable ``type`` and four unguarded ``int()``
fields, but every handler that uses ``tabId``/``workDir`` as a dict
key — or forwards other fields to SQLite — still raised on non-string
values:

* ``run``/``userAnswer``/``appendUserMessage``/``selectModel``/
  ``complete``/``closeTab``/``newChat``/``stop``/``getAdjacentTask``/
  ``mergeAction``/``getFiles`` with a list ``tabId``/``workDir``:
  ``TypeError: unhashable type`` out of a registry/dict lookup.
* ``selectModel`` with a list ``model``: corrupted
  ``tab.selected_model`` / ``self._default_model`` to a list FIRST,
  then raised ``sqlite3.ProgrammingError`` from
  ``_record_model_usage``.
* ``setWorkDir`` with a list ``workDir``: silently corrupted the
  daemon-global ``self.work_dir`` (and ``printer.work_dir``) to a
  list — crashing every later task that falls back to it.
* ``recordFileUsage`` with a non-string ``path``:
  ``sqlite3.ProgrammingError``.
* ``saveConfig`` with a non-dict ``config``/``apiKeys``:
  ``AttributeError``.
* ``getHistory`` with a non-string ``query`` (``AttributeError``) or
  a non-int ``offset`` (``sqlite3.IntegrityError``).
* ``complete`` with a dict ``query``: the request was queued and
  killed the SINGLETON ``_complete_worker_loop`` thread
  (``AttributeError`` in ``_prefix_match_task``); the worker is never
  restarted, so ghost-text autocomplete died for the daemon's whole
  lifetime, for every window.
* ``getFiles`` with a dict ``prefix``: crashed the background
  ``_do_refresh`` thread (uncaught ``TypeError`` in
  ``rank_file_suggestions``), so the file picker never received its
  reply.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from collections.abc import Generator
from typing import Any

import pytest

from kiss.agents.sorcar.persistence import _close_db
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Point persistence/config at a temp dir (pattern from
    test_model_usage_on_select_only.py) — ``selectModel``/``saveConfig``
    payloads in these tests must not persist ``last_model``/config
    values into the process-shared KISS_HOME and pollute later tests."""
    import kiss.agents.sorcar.persistence as pm
    import kiss.server.vscode_config as vc

    _close_db()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(pm, "_KISS_DIR", type(pm._KISS_DIR)(tmpdir))
    monkeypatch.setattr(
        pm, "_DB_PATH", type(pm._DB_PATH)(os.path.join(tmpdir, "sorcar.db")),
    )
    cfg_path = os.path.join(tmpdir, "config.json")
    monkeypatch.setattr(vc, "CONFIG_DIR", type(vc.CONFIG_DIR)(tmpdir))
    monkeypatch.setattr(vc, "CONFIG_PATH", type(vc.CONFIG_PATH)(cfg_path))
    yield
    _close_db()


class _ThreadCrashRecorder:
    """Record uncaught exceptions from background threads."""

    def __init__(self) -> None:
        self.crashes: list[str] = []
        self._orig = threading.excepthook

    def __enter__(self) -> _ThreadCrashRecorder:
        def hook(args: Any) -> None:
            self.crashes.append(
                f"{args.thread.name}: {args.exc_type.__name__}: {args.exc_value}"
            )

        threading.excepthook = hook
        return self

    def __exit__(self, *exc: Any) -> None:
        threading.excepthook = self._orig


class TestMalformedFields(unittest.TestCase):
    """Malformed payloads must never raise out of ``_handle_command``."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_nonstring_tab_id_does_not_raise(self) -> None:
        payloads = [
            {"type": "run", "tabId": [1], "task": "x"},
            {"type": "userAnswer", "tabId": [1], "answer": "y"},
            {"type": "appendUserMessage", "tabId": [1], "prompt": "p"},
            {"type": "selectModel", "tabId": [1], "model": "m"},
            {"type": "complete", "tabId": [1], "query": "qq"},
            {"type": "closeTab", "tabId": [1]},
            {"type": "newChat", "tabId": [1]},
            {"type": "stop", "tabId": [1]},
            {"type": "getAdjacentTask", "tabId": [1]},
            {"type": "mergeAction", "action": "all-done", "tabId": [1]},
            {"type": "resumeSession", "chatId": "c", "tabId": [1]},
        ]
        for p in payloads:
            self.server._handle_command(dict(p))
        # No phantom registry entries may have been minted for the
        # malformed ids either.
        for key in _RunningAgentState.running_agent_states:
            assert isinstance(key, str), f"non-string registry key {key!r}"

    def test_nonstring_model_does_not_corrupt_default(self) -> None:
        before = self.server._default_model
        self.server._handle_command(
            {"type": "selectModel", "tabId": "t1", "model": [1]},
        )
        assert self.server._default_model == before, (
            f"_default_model corrupted to {self.server._default_model!r}"
        )
        tab = _RunningAgentState.running_agent_states.get("t1")
        if tab is not None:
            assert isinstance(tab.selected_model, str), (
                f"selected_model corrupted to {tab.selected_model!r}"
            )

    def test_nonstring_work_dir_does_not_corrupt_state(self) -> None:
        before = self.server.work_dir
        self.server._handle_command({"type": "setWorkDir", "workDir": [1]})
        assert self.server.work_dir == before, (
            f"work_dir corrupted to {self.server.work_dir!r}"
        )
        self.server._handle_command(
            {"type": "getFiles", "prefix": "x", "workDir": [1]},
        )
        self.server._handle_command(
            {"type": "generateCommitMessage", "tabId": "t1", "workDir": [1]},
        )

    def test_record_file_usage_nonstring_path_does_not_raise(self) -> None:
        self.server._handle_command(
            {"type": "recordFileUsage", "path": ["x"]},
        )
        self.server._handle_command(
            {"type": "recordFileUsage", "path": {"x": 1}},
        )

    def test_save_config_nondict_does_not_raise(self) -> None:
        self.server._handle_command({"type": "saveConfig", "config": ["x"]})
        self.server._handle_command({"type": "saveConfig", "config": "x"})
        self.server._handle_command(
            {"type": "saveConfig", "config": {}, "apiKeys": ["x"]},
        )

    def test_get_history_malformed_fields_do_not_raise(self) -> None:
        self.server._handle_command({"type": "getHistory", "query": [1]})
        self.server._handle_command(
            {"type": "getHistory", "query": "x", "offset": "abc"},
        )
        histories = [e for e in self.events if e.get("type") == "history"]
        assert histories, "getHistory must still reply with a history event"

    def test_malformed_complete_query_does_not_kill_worker(self) -> None:
        # A dict query (>= 2 keys so it passes the len() < 2 guard)
        # previously killed the singleton autocomplete worker thread;
        # ghost text then never worked again for the whole daemon.
        with _ThreadCrashRecorder() as rec:
            # Distinct connIds: the per-connection staleness check must
            # not skip the malformed item (a same-connection follow-up
            # would mark it stale before the worker dequeues it).
            self.server._handle_command(
                {
                    "type": "complete",
                    "query": {"a": 1, "b": 2},
                    "tabId": "",
                    "connId": "win-A",
                },
            )
            time.sleep(0.3)
            deadline = time.time() + 5.0
            ghosts: list[dict[str, Any]] = []
            self.server._handle_command(
                {
                    "type": "complete",
                    "query": "abcd",
                    "tabId": "",
                    "connId": "win-B",
                },
            )
            while time.time() < deadline:
                ghosts = [
                    e
                    for e in self.events
                    if e.get("type") == "ghost" and e.get("query") == "abcd"
                ]
                if ghosts:
                    break
                time.sleep(0.05)
        assert ghosts, (
            "autocomplete worker died after a malformed query; "
            f"no ghost reply for the valid follow-up (crashes={rec.crashes})"
        )

    def test_malformed_get_files_prefix_no_thread_crash(self) -> None:
        with _ThreadCrashRecorder() as rec:
            self.server._handle_command(
                {"type": "getFiles", "prefix": {"a": 1}},
            )
            time.sleep(1.0)
        assert not rec.crashes, (
            f"getFiles with non-string prefix crashed a thread: {rec.crashes}"
        )


if __name__ == "__main__":
    unittest.main()
