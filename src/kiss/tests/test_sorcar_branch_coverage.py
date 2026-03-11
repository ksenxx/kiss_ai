"""Integration tests targeting uncovered branches in kiss/agents/sorcar/.

No mocks, patches, or test doubles. Uses real files, real git repos, and
real objects.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
)
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _prepare_merge_view,
    _save_untracked_base,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.config import AgentConfig, SorcarConfig
from kiss.agents.sorcar.prompt_detector import PromptDetector
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
)
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_command_names,
)
from kiss.agents.sorcar.web_use_tool import (
    INTERACTIVE_ROLES,
    WebUseTool,
)


class TestPromptDetector:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # No crash, just no frontmatter bonus


# ---------------------------------------------------------------------------
# useful_tools.py
# ---------------------------------------------------------------------------


class TestExtractCommandNames:

    def test_invalid_shlex(self) -> None:
        """Unmatched quote should not crash."""
        names = _extract_command_names("echo 'unclosed")
        # Should handle gracefully
        assert isinstance(names, list)

class TestUsefulTools:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.tools = UsefulTools()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# task_history.py
# ---------------------------------------------------------------------------


class TestTaskHistory:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        # Save originals
        self._orig_history_file = th.HISTORY_FILE
        self._orig_model_usage_file = th.MODEL_USAGE_FILE
        self._orig_file_usage_file = th.FILE_USAGE_FILE
        self._orig_kiss_dir = th._KISS_DIR
        self._orig_events_dir = th._CHAT_EVENTS_DIR
        # Redirect to temp
        th._KISS_DIR = Path(self.tmpdir)
        th.HISTORY_FILE = Path(self.tmpdir) / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = Path(self.tmpdir) / "chat_events"
        th.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        th.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        # Clear cache
        th._history_cache = None

    def teardown_method(self) -> None:
        th.HISTORY_FILE = self._orig_history_file
        th._CHAT_EVENTS_DIR = self._orig_events_dir
        th.MODEL_USAGE_FILE = self._orig_model_usage_file
        th.FILE_USAGE_FILE = self._orig_file_usage_file
        th._KISS_DIR = self._orig_kiss_dir
        th._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# web_use_tool.py: _number_interactive_elements
# ---------------------------------------------------------------------------


class TestCoalesceEvents:

    def test_no_merge_missing_text(self) -> None:
        events = [
            {"type": "thinking_delta"},
            {"type": "thinking_delta", "text": "a"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

class TestScanFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestGitUtilities:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestSaveUntrackedBase:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(self.work_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestCleanupMergeData:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestRestoreMergeFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(self.work_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPrepareMergeView:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)
        self.data_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        # May or may not have changes depending on decode


class TestSetupCodeServer:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_setup_removes_chat_sessions(self) -> None:
        ws_dir = Path(self.tmpdir) / "User" / "workspaceStorage" / "ws1" / "chatSessions"
        ws_dir.mkdir(parents=True)
        (ws_dir / "session.json").write_text("{}")
        _setup_code_server(self.tmpdir)
        assert not ws_dir.exists()


class TestDisableCopilotScmButton:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Should not rewrite


# ---------------------------------------------------------------------------
# chatbot_ui.py
# ---------------------------------------------------------------------------


class TestThemePresets:
    def test_all_presets_exist(self) -> None:
        assert "dark" in _THEME_PRESETS
        assert "light" in _THEME_PRESETS
        assert "hcDark" in _THEME_PRESETS
        assert "hcLight" in _THEME_PRESETS

    def test_presets_have_keys(self) -> None:
        for name, preset in _THEME_PRESETS.items():
            assert "bg" in preset
            assert "fg" in preset
            assert "accent" in preset


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestSorcarConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.model_name == "claude-opus-4-6"
        assert cfg.max_steps == 100
        assert cfg.max_budget == 200.0
        assert cfg.headless is False

    def test_sorcar_config(self) -> None:
        cfg = SorcarConfig()
        assert isinstance(cfg.sorcar_agent, AgentConfig)


# ---------------------------------------------------------------------------
# sorcar.py utility functions
# ---------------------------------------------------------------------------


class TestReadActiveFile:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestResolveTask:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestSorcarAgentRunAttachments:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _try_run(self, **kwargs: object) -> None:
        agent = SorcarAgent("test")
        try:
            agent.run(
                prompt_template="test",
                work_dir=self.tmpdir,
                max_steps=0,
                max_budget=0.0,
                headless=True,
                verbose=False,
                **kwargs,  # type: ignore[arg-type]
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# web_use_tool.py: WebUseTool construction and close
# ---------------------------------------------------------------------------


class TestSorcarAgentMain:
    def test_main_no_work_dir(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kiss.agents.sorcar.sorcar_agent",
                "--max_steps", "0",
                "--max_budget", "0.0",
                "--headless", "true",
                "--verbose", "false",
                "--task", "say hello",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_main_with_file(self) -> None:
        tmpdir = tempfile.mkdtemp()
        task_file = os.path.join(tmpdir, "task.txt")
        Path(task_file).write_text("echo hello")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "-f", task_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

class TestUsefulToolsEdgeCases:
    """Cover remaining useful_tools.py branches."""

    def test_extract_command_names_escaped_chars(self) -> None:
        """Escaped characters."""
        names = _extract_command_names("echo hello\\ world")
        assert "echo" in names

    def test_extract_command_names_double_quote_escape(self) -> None:
        """Escaped quote in double-quoted string."""
        names = _extract_command_names('echo "hello \\"world\\""')
        assert "echo" in names


class TestBrowserUiEdgeCases:
    """Cover remaining browser_ui.py branches."""

    def test_handle_message_subtype_not_tool_output(self) -> None:
        """Cover message with subtype != tool_output."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            subtype = "other"
            data = {"content": "text"}

        printer._handle_message(Msg())
        assert cq.empty()
        printer.remove_client(cq)


class TestCodeServerEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cleanup_merge_data_manifest_readonly(self) -> None:
        """Cover OSError in manifest.unlink()."""
        manifest = Path(self.tmpdir) / "pending-merge.json"
        manifest.write_text("{}")
        os.chmod(self.tmpdir, 0o555)
        try:
            _cleanup_merge_data(self.tmpdir)
        finally:
            os.chmod(self.tmpdir, 0o755)


class TestTaskHistoryEdgeCases:
    """Cover remaining task_history.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self._orig_history_file = th.HISTORY_FILE
        self._orig_model_usage_file = th.MODEL_USAGE_FILE
        self._orig_file_usage_file = th.FILE_USAGE_FILE
        self._orig_kiss_dir = th._KISS_DIR
        self._orig_events_dir = th._CHAT_EVENTS_DIR
        th._KISS_DIR = Path(self.tmpdir)
        th.HISTORY_FILE = Path(self.tmpdir) / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = Path(self.tmpdir) / "chat_events"
        th.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        th.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        th._history_cache = None

    def teardown_method(self) -> None:
        th.HISTORY_FILE = self._orig_history_file
        th._CHAT_EVENTS_DIR = self._orig_events_dir
        th.MODEL_USAGE_FILE = self._orig_model_usage_file
        th.FILE_USAGE_FILE = self._orig_file_usage_file
        th._KISS_DIR = self._orig_kiss_dir
        th._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPromptDetectorEdgeCases:
    """Cover remaining prompt_detector.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Should not have frontmatter bonus


class TestWebUseToolEdgeCases:
    """Cover remaining web_use_tool.py branches without starting a browser."""

    def test_scroll_delta_values(self) -> None:
        """Cover _SCROLL_DELTA dict lookup."""
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)

    def test_interactive_roles_complete(self) -> None:
        """Ensure all expected roles are in INTERACTIVE_ROLES."""
        expected = {"link", "button", "textbox", "searchbox", "combobox",
                    "checkbox", "radio", "switch", "slider", "spinbutton",
                    "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
                    "option", "treeitem"}
        assert INTERACTIVE_ROLES == expected


class TestBrowserUiFinalEdgeCases:
    """Cover last few browser_ui.py branches."""

    def test_handle_message_content_mixed_blocks(self) -> None:
        """Content with both valid and invalid blocks."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class ValidBlock:
            is_error = False
            content = "ok"

        class InvalidBlock:
            pass

        class Msg:
            content = [InvalidBlock(), ValidBlock()]

        printer._handle_message(Msg())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        printer.remove_client(cq)

    def test_print_text_only_whitespace(self) -> None:
        """Cover empty text.strip() branch."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("   \n\t  ", type="text")
        assert cq.empty()
        printer.remove_client(cq)


# ---------------------------------------------------------------------------
# web_use_tool.py: Browser integration tests using headless Playwright
# ---------------------------------------------------------------------------
class TestWebUseToolBrowser:
    """Integration tests for WebUseTool with a real headless browser."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(
            browser_type="chromium",
            headless=True,
            user_data_dir=None,
        )

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_click_hover(self) -> None:
        url = self._write_html(
            "hover.html",
            "<html><body><button>Hover</button></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1, action="hover")
        assert "Page:" in result
        # May or may not open a new tab, but should not crash
        # Should handle empty body gracefully


# ---------------------------------------------------------------------------
# sorcar.py: Server integration test via subprocess
# ---------------------------------------------------------------------------


def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


class TestSorcarServerIntegration:
    """Integration tests for run_chatbot by starting it as a subprocess.

    Uses _sorcar_test_server_with_cov.py to collect coverage data from
    the server subprocess. Coverage is combined after the server stops.
    """

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        tmpdir = tmp_path_factory.mktemp("sorcar_server")
        work_dir = str(tmpdir / "work")
        os.makedirs(work_dir)

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True,
        )
        Path(work_dir, "file.txt").write_text("line1\nline2\n")
        Path(work_dir, "readme.md").write_text("# Test\n\nsome content\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        port_file = str(tmpdir / "port")
        cov_data_file = str(tmpdir / ".coverage.server")

        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "_sorcar_test_server_with_cov.py"),
                port_file,
                work_dir,
                cov_data_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        keepalive = None
        try:
            port = _wait_for_port_file(port_file)
            base_url = f"http://127.0.0.1:{port}"

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    resp = requests.get(base_url, timeout=2)
                    if resp.status_code == 200:
                        break
                except requests.ConnectionError:
                    time.sleep(0.3)
            else:
                raise TimeoutError("Server not responsive")

            # Keep an SSE client connected to prevent auto-shutdown
            keepalive = requests.get(
                f"{base_url}/events", stream=True, timeout=300,
            )

            yield base_url, work_dir, proc, str(tmpdir)
        finally:
            if keepalive is not None:
                keepalive.close()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            # Try to combine coverage from subprocess
            try:
                import coverage as cov_mod

                if os.path.exists(cov_data_file):
                    main_cov_file = os.path.join(os.getcwd(), ".coverage")
                    cov = cov_mod.Coverage(data_file=main_cov_file)
                    cov.combine(data_paths=[cov_data_file], keep=True)
                    cov.save()
            except Exception:
                pass

    def test_index(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(base_url, timeout=5)
        assert r.status_code == 200
        assert "KISS" in r.text or "html" in r.text.lower()

    def test_models(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "selected" in data

    def test_theme(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/theme", timeout=5)
        assert r.status_code == 200

    def test_tasks(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/tasks", timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_suggestions_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=", timeout=5)
        assert r.status_code == 200

    def test_suggestions_with_query(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=test", timeout=5)
        assert r.status_code == 200

    def test_suggestions_files_mode(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?mode=files&q=file", timeout=5)
        assert r.status_code == 200

    def test_complete_short(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert r.status_code == 200

    def test_complete_longer(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=test something", timeout=5)
        assert r.status_code == 200

    def test_active_file_info(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200

    def test_get_file_content(self, server) -> None:
        base_url, work_dir, _, _ = server
        fpath = os.path.join(work_dir, "file.txt")
        r = requests.get(f"{base_url}/get-file-content?path={fpath}", timeout=5)
        assert r.status_code == 200

    def test_get_file_content_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/get-file-content?path=/no/such/file", timeout=5)
        assert r.status_code == 404

    def test_run_empty_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run", json={"task": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/run",
            json={"task": "test task", "model": "claude-opus-4-6"},
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_stop_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/stop", timeout=5)
        assert r.status_code in (200, 404)

    def test_run_selection_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run-selection", json={"text": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_selection(self, server) -> None:
        base_url, _, _, _ = server
        time.sleep(1)  # Ensure previous task is done
        r = requests.post(f"{base_url}/run-selection", json={"text": "echo hello"}, timeout=5)
        assert r.status_code == 200
        time.sleep(1)

    def test_task_events_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=bad", timeout=5)
        assert r.status_code == 400

    def test_task_events_out_of_range(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=9999", timeout=5)
        assert r.status_code == 404

    def test_task_events_valid(self, server) -> None:
        base_url, _, _, _ = server
        tasks = requests.get(f"{base_url}/tasks", timeout=5).json()
        if tasks:
            r = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
            assert r.status_code == 200

    def test_open_file(self, server) -> None:
        base_url, work_dir, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": os.path.join(work_dir, "file.txt")},
            timeout=5,
        )
        assert r.status_code == 200

    def test_open_file_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": "/no/such/file.txt"},
            timeout=5,
        )
        assert r.status_code == 404

    def test_open_file_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/open-file", json={"path": ""}, timeout=5)
        assert r.status_code == 400

    def test_merge_action_next(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "next"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "invalid"}, timeout=5
        )
        assert r.status_code == 400

    def test_merge_action_all_done(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "all-done"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_accept(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "accept"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_reject(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "reject"}, timeout=5
        )
        assert r.status_code == 200

    def test_focus_chatbox(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-chatbox", timeout=5)
        assert r.status_code == 200

    def test_focus_editor(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-editor", timeout=5)
        assert r.status_code == 200

    def test_record_file_usage(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/test.py"},
            timeout=5,
        )
        assert r.status_code == 200

    def test_commit_no_changes(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/commit", timeout=30)
        assert r.status_code in (200, 400)

    def test_push_no_remote(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/push", timeout=10)
        assert r.status_code in (200, 400)

    def test_sse_events(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/events", stream=True, timeout=10)
        assert r.status_code == 200
        content = b""
        for chunk in r.iter_content(chunk_size=256):
            content += chunk
            if len(content) > 20:
                break
        r.close()

    def test_run_with_attachments(self, server) -> None:
        import base64

        base_url, _, _, _ = server
        time.sleep(1)
        fake_image = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        fake_pdf = base64.b64encode(b"%PDF-1.4 fake").decode()
        r = requests.post(
            f"{base_url}/run",
            json={
                "task": "test with attachments",
                "attachments": [
                    {"data": fake_image, "mime_type": "image/png"},
                    {"data": fake_pdf, "mime_type": "application/pdf"},
                ],
            },
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_active_file_md(self, server) -> None:
        """Cover active file with .md extension triggering prompt detection."""
        import hashlib

        base_url, work_dir, _, _ = server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
        cs_dir = _KISS_DIR / f"cs-{wd_hash}"
        cs_dir.mkdir(parents=True, exist_ok=True)
        (cs_dir / "active-file.json").write_text(
            json.dumps({"path": os.path.join(work_dir, "readme.md")})
        )
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200
        assert "is_prompt" in r.json()

    def test_closing(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/closing", timeout=5)
        assert r.status_code == 200


class TestWebUseToolBrowserExtra:
    """Additional browser tests for remaining web_use_tool.py branches."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(
            browser_type="chromium",
            headless=True,
            user_data_dir=None,
        )

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_screenshot_error(self) -> None:
        """Cover screenshot exception path (line 346-348)."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("ss.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.screenshot()
        assert "Error" in result
        tool2.close()

    def test_get_page_content_error(self) -> None:
        """Cover get_page_content exception path (line 368-370)."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("pc.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.get_page_content()
        assert "Error" in result
        tool2.close()

    def test_press_key_error(self) -> None:
        """Cover press_key exception path."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("k.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.press_key("Enter")
        assert "Error" in result
        tool2.close()

    def test_type_text_error(self) -> None:
        """Cover type_text exception path."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("t.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.type_text(1, "text")
        assert "Error" in result
        tool2.close()


class TestCodeServerFinalEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_merge_view_modified_untracked_already_in_file_hunks(self) -> None:
        """Pre-existing untracked file already in file_hunks -> skip."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        Path(work_dir, "f.txt").write_text("a\nb\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        # Create untracked file AND tracked file change
        Path(work_dir, "ut.txt").write_text("original\n")
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, pre_untracked | {"f.txt"})
        _save_untracked_base(work_dir, os.path.join(self.tmpdir, "data"), pre_untracked)

        # Agent modifies tracked file and untracked file
        Path(work_dir, "f.txt").write_text("a\nX\n")
        Path(work_dir, "ut.txt").write_text("modified\n")

        data_dir = os.path.join(self.tmpdir, "data")
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
        # Both files should be in merge view
        assert result.get("status") == "opened"
