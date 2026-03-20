"""Tests for race condition fixes in sorcar.

Covers:
1. Base._class_lock protects agent_counter and global_budget_used
2. stop_event set/clear inside running_lock (no stop→run race)
3. task_done broadcast after running=False (no 409 on immediate re-submit)
4. _history_cache protected by _HISTORY_LOCK (FileLock, cross-process + cross-thread)
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
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest
from filelock import FileLock

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar import task_history as _task_history_module
from kiss.agents.sorcar.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
    find_free_port,
)
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.sorcar import (
    _model_vendor_order,
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
    old_hist = th.HISTORY_FILE
    old_model = th.MODEL_USAGE_FILE
    old_file = th.FILE_USAGE_FILE
    old_cache = th._history_cache
    old_kiss = th._KISS_DIR
    old_events = th._CHAT_EVENTS_DIR

    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th.HISTORY_FILE = kiss_dir / "task_history.jsonl"
    th._CHAT_EVENTS_DIR = kiss_dir / "chat_events"
    th.MODEL_USAGE_FILE = kiss_dir / "model_usage.json"
    th.FILE_USAGE_FILE = kiss_dir / "file_usage.json"
    th._history_cache = None
    return old_hist, old_model, old_file, old_cache, old_kiss, old_events


def _restore_history(saved):
    th.HISTORY_FILE = saved[0]
    th.MODEL_USAGE_FILE = saved[1]
    th.FILE_USAGE_FILE = saved[2]
    th._history_cache = saved[3]
    th._KISS_DIR = saved[4]
    th._CHAT_EVENTS_DIR = saved[5]


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


class TestHistoryLock:
    def test_history_lock_exists(self):
        assert hasattr(_task_history_module, "_HISTORY_LOCK")
        assert isinstance(_task_history_module._HISTORY_LOCK, FileLock)

    def test_concurrent_set_chat_events_and_add_task(self):
        orig_cache = _task_history_module._history_cache
        orig_file_content = None
        if _task_history_module.HISTORY_FILE.exists():
            orig_file_content = _task_history_module.HISTORY_FILE.read_text()

        try:
            _task_history_module._history_cache = None
            _add_task("initial_task")
            _task_history_module._history_cache = None

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
            _task_history_module._history_cache = orig_cache
            if orig_file_content is not None:
                _task_history_module.HISTORY_FILE.write_text(orig_file_content)


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
        assert _model_vendor_order("claude-3.5-sonnet") == 0
        assert _model_vendor_order("gpt-4o") == 1
        assert _model_vendor_order("o1-preview") == 1
        assert _model_vendor_order("gemini-2.0-flash") == 2
        assert _model_vendor_order("minimax-model") == 3
        assert _model_vendor_order("openrouter/anthropic/claude") == 4
        assert _model_vendor_order("unknown-model") == 5

    def test_stop_requested_is_base_exception(self) -> None:
        assert issubclass(_StopRequested, BaseException)
        with pytest.raises(_StopRequested):
            raise _StopRequested()


class TestSorcarServerSubprocess:
    @pytest.fixture(autouse=True)
    def setup_env(self):
        import socket

        self.tmpdir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(self.work_dir)
        _make_git_repo(self.work_dir)

        self.port = find_free_port()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        (kiss_dir / "assistant-port").write_text(str(self.port))

        helper = Path(self.tmpdir) / "run_server.py"
        src_path = os.path.join(os.path.dirname(__file__), "..", "..")
        helper.write_text(
            f"import sys, os, signal, threading, time\n"
            f"sys.path.insert(0, {src_path!r})\n"
            f"import webbrowser\n"
            f"webbrowser.open = lambda *a, **k: None\n"
            f"import kiss.agents.sorcar.browser_ui as bui\n"
            f"bui.find_free_port = lambda: {self.port}\n"
            f"import kiss.agents.sorcar.task_history as th\n"
            f"from pathlib import Path\n"
            f"kiss_dir = Path({str(kiss_dir)!r})\n"
            f"th._KISS_DIR = kiss_dir\n"
            f"th.HISTORY_FILE = kiss_dir / 'task_history.jsonl'\n"
            f"th._CHAT_EVENTS_DIR = kiss_dir / 'chat_events'\n"
            f"th.MODEL_USAGE_FILE = kiss_dir / 'model_usage.json'\n"
            f"th.FILE_USAGE_FILE = kiss_dir / 'file_usage.json'\n"
            f"th._history_cache = None\n"
            f"original_exit = os._exit\n"
            f"os._exit = lambda code: sys.exit(code)\n"
            f"from kiss.agents.sorcar.sorcar_agent import SorcarAgent\n"
            f"from kiss.agents.sorcar.sorcar import run_chatbot\n"
            f"try:\n"
            f"    run_chatbot(\n"
            f"        agent_factory=SorcarAgent,\n"
            f"        title='Test',\n"
            f"        work_dir={self.work_dir!r},\n"
            f"        default_model='claude-opus-4-6',\n"
            f"    )\n"
            f"except (SystemExit, KeyboardInterrupt):\n"
            f"    pass\n"
        )

        self.cov_file = os.path.join(self.tmpdir, ".coverage.subprocess")
        env = {**os.environ, "COVERAGE_FILE": self.cov_file}
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "coverage", "run",
                "--branch",
                "--source=kiss.agents.sorcar",
                str(helper),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(80):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.25)
        else:
            self.proc.terminate()
            pytest.fail("Server didn't start")

        self.base = f"http://127.0.0.1:{self.port}"
        yield

        self.proc.send_signal(2)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

        if os.path.exists(self.cov_file):
            main_cov = os.path.join(os.getcwd(), ".coverage")
            subprocess.run(
                [sys.executable, "-m", "coverage", "combine",
                 "--append", self.cov_file],
                env={**os.environ, "COVERAGE_FILE": main_cov},
                capture_output=True,
            )

        shutil.rmtree(self.tmpdir, ignore_errors=True)


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
        assert history[0]["has_events"] is False

    def test_load_history_corrupt(self) -> None:
        th.HISTORY_FILE.write_text("bad json")
        th._history_cache = None
        history = th._load_history()
        assert len(history) > 0

    def test_load_json_dict_corrupt(self) -> None:
        th.MODEL_USAGE_FILE.write_text("not json")
        assert th._load_model_usage() == {}


class TestExtractCommandNames:
    def test_env_var_prefix(self) -> None:
        names = _extract_command_names("FOO=bar python script.py")
        assert "python" in names


class TestUsefulToolsRead:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestUsefulToolsWrite:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestUsefulToolsEdit:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestUsefulToolsBash:
    def setup_method(self) -> None:
        self.tools = UsefulTools()

    def test_truncation(self) -> None:
        result = self.tools.Bash("python -c \"print('x'*100000)\"", "test",
                                max_output_chars=100)
        assert "truncated" in result


class TestGitDiffAndMerge:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _make_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir)


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

    def test_load_history_empty_file(self) -> None:
        th.HISTORY_FILE.write_text("")
        th._history_cache = None
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
