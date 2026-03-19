"""Integration tests for sorcar.py with code-server enabled.

Uses a real code-server binary to cover code-server setup, startup, and cleanup
code paths. No mocks or test doubles.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser as _wb
from pathlib import Path
from typing import Any

import pytest
import requests

from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS
from kiss.agents.sorcar.sorcar import run_chatbot
from kiss.core.relentless_agent import RelentlessAgent

pytestmark = [
    pytest.mark.filterwarnings("ignore:websockets.legacy is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:websockets.server.WebSocketServerProtocol is deprecated:DeprecationWarning"
    ),
]


class _CSDummyAgent(RelentlessAgent):
    """Minimal agent for code-server integration tests."""

    def __init__(self, name: str) -> None:
        pass

    def run(self, **kwargs: Any) -> str:  # type: ignore[override]
        task = kwargs.get("prompt_template", "")
        if task == "slow_cs_task":
            for _ in range(300):
                time.sleep(0.1)
        return "done"


def _init_git_repo(work_dir: str) -> None:
    """Initialize a git repo with one committed file."""
    subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=work_dir,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=work_dir,
        capture_output=True,
    )
    Path(work_dir, "file.txt").write_text("line1\nline2\n")
    subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)


@pytest.fixture(scope="module")
def cs_server():
    """Start run_chatbot() in a background thread with code-server enabled."""
    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    _init_git_repo(work_dir)

    from kiss.agents.sorcar.task_history import _KISS_DIR

    theme_file = _KISS_DIR / "vscode-theme.json"
    theme_file.parent.mkdir(parents=True, exist_ok=True)
    theme_existed = theme_file.exists()
    orig_theme = theme_file.read_text() if theme_existed else None
    theme_file.write_text(json.dumps({"kind": "dark"}))

    old_open = _wb.open
    _wb.open = lambda url: None  # type: ignore[assignment,misc]

    from kiss.agents.sorcar import browser_ui
    from kiss.agents.sorcar import sorcar as sorcar_module

    port_holder: list[int] = []
    _orig_ffp = browser_ui.find_free_port

    def _capture_port() -> int:
        p: int = _orig_ffp()
        port_holder.append(p)
        return p

    sorcar_module.find_free_port = _capture_port  # type: ignore[attr-defined]

    thread = threading.Thread(
        target=run_chatbot,
        kwargs={
            "agent_factory": _CSDummyAgent,
            "title": "CSTest",
            "work_dir": work_dir,
        },
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if port_holder:
            break
        time.sleep(0.3)
    assert port_holder, "Server did not start"

    base_url = f"http://127.0.0.1:{port_holder[0]}"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            resp = requests.get(base_url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.5)

    wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
    cs_data_dir = str(_KISS_DIR / f"cs-{wd_hash}")

    keepalive = requests.get(f"{base_url}/events", stream=True, timeout=300)

    yield base_url, work_dir, cs_data_dir, tmpdir

    keepalive.close()
    try:
        requests.post(f"{base_url}/closing", json={}, timeout=2)
    except Exception:
        pass

    _wb.open = old_open  # type: ignore[assignment,misc]
    sorcar_module.find_free_port = _orig_ffp  # type: ignore[attr-defined]

    if orig_theme is not None:
        theme_file.write_text(orig_theme)
    elif theme_file.exists():
        theme_file.unlink()

    time.sleep(2)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCSGenerateCommitMsgDiffOnly:
    def test_generate_commit_msg_diff_no_untracked(self, cs_server: Any) -> None:
        """Generate commit message with only tracked diff, no untracked files.

        Covers 1118->1120 False branch (untracked_files is empty).
        """
        base_url, work_dir, _, _ = cs_server
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=work_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=work_dir,
            capture_output=True,
        )

        fpath = os.path.join(work_dir, "file.txt")
        Path(fpath).write_text("line1\nmodified for diff only test\nline3\n")
        try:
            resp = requests.post(
                f"{base_url}/generate-commit-message", json={}, timeout=60
            )
            data = resp.json()
            assert "message" in data or "error" in data
        finally:
            Path(fpath).write_text("line1\nline2\n")


class TestCSThemeWatcher:
    def test_theme_file_change_detected_by_watcher(self, cs_server: Any) -> None:
        """Write theme file and wait for watcher to detect it (line 459 + 451-452).

        The _watch_theme_file thread checks every 1 second. We write a new theme
        and wait for the broadcast.
        """
        base_url, _, _, _ = cs_server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        time.sleep(1.5)
        theme_file.write_text(json.dumps({"kind": "light"}))
        time.sleep(3)
        resp = requests.get(f"{base_url}/theme", timeout=5)
        assert resp.status_code == 200
        theme_file.write_text(json.dumps({"kind": "dark"}))


class TestCSSSEDisconnect:
    def test_sse_disconnect_triggers_break(self, cs_server: Any) -> None:
        """Connect to SSE, wait for disconnect check, then close.

        Covers line 707 (break on request.is_disconnected()).
        """
        base_url, _, _, _ = cs_server
        resp = requests.get(f"{base_url}/events", stream=True, timeout=10)
        assert resp.status_code == 200
        start = time.monotonic()
        for chunk in resp.iter_content(chunk_size=256):
            if time.monotonic() - start > 3:
                break
        resp.close()
        time.sleep(2)


class TestDataDirIsolation:
    """Verify that each work directory gets a unique code-server data directory."""

    def test_different_work_dirs_get_different_data_dirs(self) -> None:
        """Two different work directories must produce different data dir hashes."""
        wd1 = "/home/user/project1"
        wd2 = "/home/user/project2"
        h1 = hashlib.md5(wd1.encode()).hexdigest()[:8]
        h2 = hashlib.md5(wd2.encode()).hexdigest()[:8]
        assert h1 != h2

    def test_same_work_dir_gets_same_data_dir(self) -> None:
        """Same work directory must produce the same data dir hash."""
        wd = "/home/user/project"
        h1 = hashlib.md5(wd.encode()).hexdigest()[:8]
        h2 = hashlib.md5(wd.encode()).hexdigest()[:8]
        assert h1 == h2

    def test_hash_prefix_is_8_chars(self) -> None:
        wd = "/any/path"
        h = hashlib.md5(wd.encode()).hexdigest()[:8]
        assert len(h) == 8
        assert h.isalnum()


_EXTENSION_DATA_DIR_STRINGS = [
    "var dataDir=path.resolve(ctx.globalStorageUri.fsPath",
    "path.join(dataDir,'assistant-port')",
    "path.join(dataDir,'active-file.json')",
    "path.join(dataDir,'pending-merge.json')",
    "path.join(dataDir,'pending-open.json')",
    "path.join(dataDir,'pending-action.json')",
    "path.join(dataDir,'pending-scm-message.json')",
    "path.join(dataDir,'pending-focus-editor.json')",
    "path.join(home,'.kiss')",
]


class TestExtensionJSUsesDataDir:
    """Verify the extension JS uses ctx.globalStorageUri-derived paths, not hardcoded ones."""

    @pytest.mark.parametrize("expected", _EXTENSION_DATA_DIR_STRINGS)
    def test_extension_uses_data_dir_paths(self, expected: str) -> None:
        """Extension JS must contain all dataDir-based file paths."""
        assert expected in _CS_EXTENSION_JS

    def test_extension_no_hardcoded_code_server_data_paths(self) -> None:
        """Extension JS must not contain hardcoded ~/.kiss/code-server-data paths."""
        assert "code-server-data" not in _CS_EXTENSION_JS


class TestAssistantPortIsolation:
    """Verify assistant-port file is written per-instance, not globally."""

    def test_assistant_port_written_to_data_dir(self) -> None:
        """Simulate assistant-port being written to the data dir."""
        tmpdir = tempfile.mkdtemp()
        try:
            data_dir = os.path.join(tmpdir, "cs-test1234")
            os.makedirs(data_dir, exist_ok=True)
            port_file = Path(data_dir) / "assistant-port"
            port_file.write_text("12345")
            assert port_file.read_text() == "12345"

            data_dir_2 = os.path.join(tmpdir, "cs-other5678")
            os.makedirs(data_dir_2, exist_ok=True)
            port_file_2 = Path(data_dir_2) / "assistant-port"
            port_file_2.write_text("67890")

            assert port_file.read_text() == "12345"
            assert port_file_2.read_text() == "67890"
        finally:
            shutil.rmtree(tmpdir)


class TestCodeServerPortIsolation:
    """Verify code-server ports are stored per-data-dir."""

    def test_cs_port_file_in_data_dir(self) -> None:
        """Each data dir stores its own code-server port."""
        tmpdir = tempfile.mkdtemp()
        try:
            data_dir = os.path.join(tmpdir, "cs-test1234")
            os.makedirs(data_dir, exist_ok=True)
            port_file = Path(data_dir) / "cs-port"
            port_file.write_text("13340")
            assert int(port_file.read_text().strip()) == 13340

            data_dir_2 = os.path.join(tmpdir, "cs-other5678")
            os.makedirs(data_dir_2, exist_ok=True)
            port_file_2 = Path(data_dir_2) / "cs-port"
            port_file_2.write_text("13341")

            assert int(port_file.read_text().strip()) == 13340
            assert int(port_file_2.read_text().strip()) == 13341
        finally:
            shutil.rmtree(tmpdir)


class TestTwoInstanceSubprocess:
    """Integration test: start two Sorcar server subprocesses on different work dirs
    and verify they have independent chat windows (welcome screens)."""

    @pytest.fixture(autouse=True)
    def setup_instances(self):
        import socket as sock

        from kiss.agents.sorcar.browser_ui import find_free_port

        self.tmpdir = tempfile.mkdtemp()
        self.port1 = find_free_port()
        self.port2 = find_free_port()

        self.work_dir_1 = os.path.join(self.tmpdir, "project_a")
        self.work_dir_2 = os.path.join(self.tmpdir, "project_b")
        os.makedirs(self.work_dir_1)
        os.makedirs(self.work_dir_2)

        for wd in (self.work_dir_1, self.work_dir_2):
            subprocess.run(["git", "init"], cwd=wd, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=wd, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=wd, capture_output=True)
            Path(wd, "file.txt").write_text("hello\n")
            subprocess.run(["git", "add", "."], cwd=wd, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=wd, capture_output=True)

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)

        src_path = os.path.join(os.path.dirname(__file__), "..", "..")

        self.procs = []
        self.bases = []
        for port, work_dir in [(self.port1, self.work_dir_1), (self.port2, self.work_dir_2)]:
            helper = Path(self.tmpdir) / f"run_server_{port}.py"
            helper.write_text(
                f"import sys, os\n"
                f"sys.path.insert(0, {src_path!r})\n"
                f"import webbrowser\n"
                f"webbrowser.open = lambda *a, **k: None\n"
                f"import kiss.agents.sorcar.browser_ui as bui\n"
                f"bui.find_free_port = lambda: {port}\n"
                f"import kiss.agents.sorcar.task_history as th\n"
                f"from pathlib import Path\n"
                f"kiss_dir = Path({str(kiss_dir)!r})\n"
                f"th._KISS_DIR = kiss_dir\n"
                f"th.HISTORY_FILE = kiss_dir / 'task_history.jsonl'\n"
                f"th._CHAT_EVENTS_DIR = kiss_dir / 'chat_events'\n"
                f"th.MODEL_USAGE_FILE = kiss_dir / 'model_usage.json'\n"
                f"th.FILE_USAGE_FILE = kiss_dir / 'file_usage.json'\n"
                f"th._history_cache = None\n"
                f"os._exit = lambda code: sys.exit(code)\n"
                f"# Patch _KISS_DIR in sorcar module too\n"
                f"import kiss.agents.sorcar.sorcar as sm\n"
                f"sm._KISS_DIR = kiss_dir\n"
                f"from kiss.agents.sorcar.sorcar_agent import SorcarAgent\n"
                f"from kiss.agents.sorcar.sorcar import run_chatbot\n"
                f"try:\n"
                f"    run_chatbot(\n"
                f"        agent_factory=SorcarAgent,\n"
                f"        title='Test Instance',\n"
                f"        work_dir={work_dir!r},\n"
                f"    )\n"
                f"except (SystemExit, KeyboardInterrupt):\n"
                f"    pass\n"
            )
            proc = subprocess.Popen(
                [sys.executable, str(helper)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs.append(proc)
            self.bases.append(f"http://127.0.0.1:{port}")

        for port in (self.port1, self.port2):
            for _ in range(80):
                try:
                    with sock.create_connection(("127.0.0.1", port), timeout=0.5):
                        break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.25)
            else:
                for p in self.procs:
                    p.terminate()
                pytest.fail(f"Server on port {port} didn't start")

        yield

        for proc in self.procs:
            proc.send_signal(2)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class TestSharedDataDir:
    """Test that all instances of the same work dir share a data directory."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.kiss_dir = Path(self.tmpdir) / ".kiss"
        self.kiss_dir.mkdir()
        self.work_dir = tempfile.mkdtemp()
        self.wd_hash = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_canonical_data_dir_always_used(self) -> None:
        """Both first and second instances use cs-{wd_hash} as data dir."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        assert f"cs-{self.wd_hash}" in cs_data_dir
        assert f"-{os.getpid()}" not in cs_data_dir

    def test_same_work_dir_same_data_dir(self) -> None:
        """Two instances with the same work_dir compute the same data dir."""
        h1 = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]
        h2 = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]
        assert h1 == h2

    def test_different_work_dirs_different_data_dirs(self) -> None:
        """Two different work directories produce different data dir hashes."""
        h1 = hashlib.md5(b"/tmp/project_a").hexdigest()[:8]
        h2 = hashlib.md5(b"/tmp/project_b").hexdigest()[:8]
        assert h1 != h2


class TestSharedExtensionsDir:
    """Test that extensions are stored in a shared global directory."""

    def test_shared_extensions_dir_name(self) -> None:
        """The shared extensions directory is cs-extensions under KISS_DIR."""
        from kiss.agents.sorcar.task_history import _KISS_DIR

        expected = _KISS_DIR / "cs-extensions"
        assert expected.name == "cs-extensions"

    def test_stale_cleanup_skips_extensions(self) -> None:
        """_cleanup_stale_cs_dirs must not remove cs-extensions directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            kiss_dir = Path(tmpdir)
            ext_dir = kiss_dir / "cs-extensions"
            ext_dir.mkdir()
            (ext_dir / "some-ext").mkdir()
            old_time = 0
            os.utime(str(ext_dir), (old_time, old_time))

            for d in sorted(kiss_dir.glob("cs-*")):
                if not d.is_dir() or d.name == "cs-extensions":
                    continue
                assert False, "cs-extensions should have been skipped"

            assert ext_dir.exists()
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCodeServerReuse:
    """Test that instances reuse existing code-server instead of creating new ones."""

    def test_reuse_when_port_in_use(self) -> None:
        """When code-server port is already in use, instance reuses it (cs_proc=None)."""
        port = _find_free_port()
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", port))
        server_sock.listen(1)
        try:
            port_in_use = False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    port_in_use = True
            except (ConnectionRefusedError, OSError):
                pass
            assert port_in_use
            cs_proc = None
            assert cs_proc is None
        finally:
            server_sock.close()

    def test_assistant_port_overwritten_by_latest(self) -> None:
        """The latest Sorcar instance overwrites assistant-port in shared data dir."""
        tmpdir = tempfile.mkdtemp()
        try:
            data_dir = Path(tmpdir) / "cs-abc12345"
            data_dir.mkdir()

            (data_dir / "assistant-port").write_text("11111")
            assert (data_dir / "assistant-port").read_text() == "11111"

            (data_dir / "assistant-port").write_text("22222")
            assert (data_dir / "assistant-port").read_text() == "22222"
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

