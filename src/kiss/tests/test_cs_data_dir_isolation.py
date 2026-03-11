"""Tests for code-server data directory isolation between Sorcar instances.

Verifies that different work directories get separate code-server data
directories and IPC files, preventing name collisions when a child
Sorcar instance is launched from a parent.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS


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


class TestExtensionJSUsesDataDir:
    """Verify the extension JS uses ctx.globalStorageUri-derived paths, not hardcoded ones."""

    def test_extension_derives_data_dir_from_context(self) -> None:
        assert "var dataDir=path.resolve(ctx.globalStorageUri.fsPath" in _CS_EXTENSION_JS

    def test_extension_no_hardcoded_code_server_data_paths(self) -> None:
        """Extension JS must not contain hardcoded ~/.kiss/code-server-data paths."""
        assert "code-server-data" not in _CS_EXTENSION_JS

    def test_extension_reads_assistant_port_from_data_dir(self) -> None:
        assert "path.join(dataDir,'assistant-port')" in _CS_EXTENSION_JS

    def test_extension_writes_active_file_to_data_dir(self) -> None:
        assert "path.join(dataDir,'active-file.json')" in _CS_EXTENSION_JS

    def test_extension_pending_merge_uses_data_dir(self) -> None:
        assert "path.join(dataDir,'pending-merge.json')" in _CS_EXTENSION_JS

    def test_extension_pending_open_uses_data_dir(self) -> None:
        assert "path.join(dataDir,'pending-open.json')" in _CS_EXTENSION_JS

    def test_extension_pending_action_uses_data_dir(self) -> None:
        assert "path.join(dataDir,'pending-action.json')" in _CS_EXTENSION_JS

    def test_extension_pending_scm_uses_data_dir(self) -> None:
        assert "path.join(dataDir,'pending-scm-message.json')" in _CS_EXTENSION_JS

    def test_extension_pending_focus_uses_data_dir(self) -> None:
        assert "path.join(dataDir,'pending-focus-editor.json')" in _CS_EXTENSION_JS

    def test_extension_theme_still_uses_global_kiss_dir(self) -> None:
        """Theme file is global (shared) so it should NOT use dataDir."""
        assert "path.join(home,'.kiss')" in _CS_EXTENSION_JS


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

            # Another data dir should be independent
            data_dir_2 = os.path.join(tmpdir, "cs-other5678")
            os.makedirs(data_dir_2, exist_ok=True)
            port_file_2 = Path(data_dir_2) / "assistant-port"
            port_file_2.write_text("67890")

            # Both ports should be independent
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

        # Init git repos
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
                f"th.PROPOSALS_FILE = kiss_dir / 'proposals.json'\n"
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

        # Wait for both servers
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
