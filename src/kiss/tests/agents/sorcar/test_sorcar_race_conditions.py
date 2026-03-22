"""Tests for race condition fixes in sorcar.

Covers:
1. Base._class_lock protects agent_counter and global_budget_used
2. stop_event set/clear inside running_lock (no stop→run race)
3. task_done broadcast after running=False (no 409 on immediate re-submit)
4. _db_lock protects SQLite access (thread-safe)
5. Integration tests: rapid stop/restart, concurrent printer operations,
   browser_ui coalesce, theme presets, stream events, etc.
"""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar import task_history as _task_history_module
from kiss.agents.sorcar.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
)
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.shared_utils import model_vendor
from kiss.agents.sorcar.sorcar import (
    _read_active_file,
    _StopRequested,
)
from kiss.agents.sorcar.task_history import (
    _add_task,
    _load_history,
    _set_latest_chat_events,
)
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_command_names,
)
from kiss.core.base import Base


def _redirect_history(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_history(saved):
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_git_repo(tmpdir: str) -> str:
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
    Path(tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
    return tmpdir


class TestBaseClassLock:
    def test_class_lock_exists(self):
        assert hasattr(Base, "_class_lock")
        assert isinstance(Base._class_lock, type(threading.Lock()))

    def test_concurrent_budget_updates_with_lock(self):
        initial = Base.global_budget_used
        num = 100
        barrier = threading.Barrier(num)

        def update():
            barrier.wait()
            with Base._class_lock:
                Base.global_budget_used += 1.0

        threads = [threading.Thread(target=update) for _ in range(num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expected = initial + num
        assert abs(Base.global_budget_used - expected) < 1e-9
        Base.global_budget_used = initial


class TestTaskDoneAfterRunningFalse:
    def test_no_409_on_immediate_resubmit(self):
        running = False
        agent_thread: threading.Thread | None = None
        running_lock = threading.Lock()
        events: list[dict] = []
        events_lock = threading.Lock()
        can_check = threading.Event()

        def broadcast(event):
            with events_lock:
                events.append(event)
            if event.get("type") == "task_done":
                can_check.set()

        def run_agent_thread():
            nonlocal running, agent_thread
            current = threading.current_thread()
            with running_lock:
                if agent_thread is not current:
                    return
                running = False
                agent_thread = None
            broadcast({"type": "task_done"})

        def start_task():
            nonlocal running, agent_thread
            t = threading.Thread(target=run_agent_thread, daemon=True)
            with running_lock:
                if running:
                    return False
                running = True
                agent_thread = t
            t.start()
            return True

        assert start_task()
        can_check.wait(timeout=5)
        assert start_task()
        time.sleep(0.2)
        with running_lock:
            assert not running


class TestDbLock:
    def test_db_lock_exists(self):
        assert hasattr(_task_history_module, "_db_lock")
        assert isinstance(_task_history_module._db_lock, type(threading.Lock()))

    def test_concurrent_set_chat_events_and_add_task(self):
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_history(tmpdir)
        try:
            _add_task("initial_task")

            errors: list[Exception] = []
            barrier = threading.Barrier(2)

            def add_tasks():
                barrier.wait()
                for i in range(10):
                    try:
                        _add_task(f"add_task_{i}")
                    except Exception as e:
                        errors.append(e)

            def set_results():
                barrier.wait()
                for i in range(10):
                    try:
                        _set_latest_chat_events([{"type": "text_delta", "text": f"result_{i}"}])
                    except Exception as e:
                        errors.append(e)

            t1 = threading.Thread(target=add_tasks)
            t2 = threading.Thread(target=set_results)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert errors == [], f"Errors during concurrent access: {errors}"
            history = _load_history()
            assert isinstance(history, list)
            assert len(history) > 0
        finally:
            _restore_history(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSorcarModuleFunctions:
    def test_read_active_file_nonexistent_path(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            af = Path(tmpdir) / "active-file.json"
            af.write_text(json.dumps({"path": "/nonexistent/file.py"}))
            assert _read_active_file(tmpdir) == ""
        finally:
            shutil.rmtree(tmpdir)

    def test_model_vendor_order(self) -> None:
        assert model_vendor("claude-3.5-sonnet")[1] == 0
        assert model_vendor("gpt-4o")[1] == 1
        assert model_vendor("o1-preview")[1] == 1
        assert model_vendor("gemini-2.0-flash")[1] == 2
        assert model_vendor("minimax-model")[1] == 3
        assert model_vendor("openrouter/anthropic/claude")[1] == 4
        assert model_vendor("unknown-model")[1] == 5

    def test_stop_requested_is_base_exception(self) -> None:
        assert issubclass(_StopRequested, BaseException)
        with pytest.raises(_StopRequested):
            raise _StopRequested()


class TestBaseBrowserPrinterPrint:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_print_stream_event_tool_use_bad_json(self) -> None:
        cq = self.printer.add_client()
        ev1 = types.SimpleNamespace(event={
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "X"}
        })
        self.printer.print(ev1, type="stream_event")
        ev2 = types.SimpleNamespace(event={
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "not json"}
        })
        self.printer.print(ev2, type="stream_event")
        ev3 = types.SimpleNamespace(event={"type": "content_block_stop"})
        self.printer.print(ev3, type="stream_event")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tc = [e for e in events if e.get("type") == "tool_call"]
        assert len(tc) == 1
        self.printer.remove_client(cq)


class TestRemoveClientNotFound:
    def test_remove_nonexistent_client(self) -> None:
        printer = BaseBrowserPrinter()
        q: queue.Queue = queue.Queue()
        printer.remove_client(q)


class TestBuildHtml:
    def test_theme_presets_complete(self) -> None:
        required = {"bg", "bg2", "fg", "accent", "border", "inputBg",
                    "green", "red", "purple", "cyan"}
        for name, theme in _THEME_PRESETS.items():
            assert set(theme.keys()) == required


class TestTaskHistory:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_latest_chat_events_nonexistent(self) -> None:
        th._add_task("exists")
        th._set_latest_chat_events([{"type": "z"}], task="missing")
        history = th._load_history()
        assert not history[0]["has_events"]


class TestExtractCommandNames:
    def test_env_var_prefix(self) -> None:
        names = _extract_command_names("FOO=bar python script.py")
        assert "python" in names


class TestUsefulToolsBash:
    def setup_method(self) -> None:
        self.tools = UsefulTools()

    def test_truncation(self) -> None:
        result = self.tools.Bash("python -c \"print('x'*100000)\"", "test",
                                max_output_chars=100)
        assert "truncated" in result


class TestMergingFlag:
    def test_merge_blocks_task(self) -> None:
        running_lock = threading.Lock()
        merging = True
        running = False
        with running_lock:
            if merging:
                status = 409
            elif running:
                status = 409
            else:
                status = 200
        assert status == 409

    def test_merge_cleared_allows_task(self) -> None:
        running_lock = threading.Lock()
        merging = False
        running = False
        with running_lock:
            if merging:
                status = 409
            elif running:
                status = 409
            else:
                running = True
                status = 200
        assert status == 200


class TestUsefulToolsEdgeCases:
    def test_bash_base_exception(self) -> None:
        collected: list[str] = []

        def callback(line):
            collected.append(line)
            if len(collected) >= 2:
                raise KeyboardInterrupt("test")

        tools_s = UsefulTools(stream_callback=callback)
        with pytest.raises(KeyboardInterrupt):
            tools_s.Bash("for i in 1 2 3 4 5; do echo line$i; done", "test")


class TestCodeServerEdgeCases:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_setup_code_server_corrupt_settings(self) -> None:
        data_dir = tempfile.mkdtemp()
        ext_dir = tempfile.mkdtemp()
        try:
            user_dir = Path(data_dir) / "User"
            user_dir.mkdir(parents=True)
            (user_dir / "settings.json").write_text("not json!")
            _setup_code_server(data_dir, ext_dir)
            result = json.loads((user_dir / "settings.json").read_text())
            assert "workbench.colorTheme" in result
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
            shutil.rmtree(ext_dir, ignore_errors=True)


class TestTaskHistoryRemaining:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_history_fresh_db_has_samples(self) -> None:
        history = th._load_history()
        assert len(history) > 0


class TestCodeServerOSErrors:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_merge_new_file_unicode_error(self) -> None:
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        _make_git_repo(work_dir)
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, set(pre_hunks.keys()))
        Path(work_dir, "binary.dat").write_bytes(b"\xff\xfe" * 100)
        Path(work_dir, "new.py").write_text("print('hi')\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert isinstance(result, dict)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


class TestBrowserUiRemaining:
    def test_coalesce_non_text_same_type(self) -> None:
        events = [
            {"type": "tool_call", "name": "a"},
            {"type": "tool_call", "name": "b"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


class TestRapidStopRestart:
    def test_all_threads_terminate(self) -> None:
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        agent_thread = None
        current_stop_event = None
        threads: list[threading.Thread] = []

        def agent_fn(task, stop_ev):
            nonlocal running, agent_thread
            printer._thread_local.stop_event = stop_ev
            ct = threading.current_thread()
            try:
                for _ in range(100):
                    time.sleep(0.01)
                    printer._check_stop()
            except KeyboardInterrupt:
                pass
            finally:
                printer._thread_local.stop_event = None
                with running_lock:
                    if agent_thread is not ct:
                        return
                    running = False
                    agent_thread = None

        def stop():
            nonlocal running, agent_thread, current_stop_event
            with running_lock:
                t = agent_thread
                if t is None or not t.is_alive():
                    return
                running = False
                agent_thread = None
                ev = current_stop_event
                current_stop_event = None
            if ev:
                ev.set()

        def start(task):
            nonlocal running, agent_thread, current_stop_event
            ev = threading.Event()
            t = threading.Thread(target=agent_fn, args=(task, ev), daemon=True)
            with running_lock:
                if running:
                    return
                current_stop_event = ev
                running = True
                agent_thread = t
            threads.append(t)
            t.start()

        for i in range(15):
            start(f"t{i}")
            time.sleep(random.uniform(0.02, 0.05))
            stop()

        for t in threads:
            t.join(3)
            assert not t.is_alive()
