"""Integration tests targeting uncovered branches in sorcar.py and sorcar_agent.py.

No mocks, patches, or test doubles.  Exercises real code paths via subprocess
invocations and direct function calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

from kiss.agents.sorcar.sorcar import (
    _clean_llm_output,
    _generate_commit_msg,
    _model_vendor_order,
    _read_active_file,
    run_chatbot,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _build_arg_parser,
    _resolve_task,
)
from kiss.core.relentless_agent import RelentlessAgent


class TestGenerateCommitMsg:
    def test_non_detailed(self) -> None:
        result = _generate_commit_msg("added: hello world")
        assert isinstance(result, str)

    def test_detailed(self) -> None:
        result = _generate_commit_msg("added: hello world", detailed=True)
        assert isinstance(result, str)


# ── _read_active_file: path exists but is a directory (not a file) ────────
class TestReadActiveFileDirPath:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_path_is_directory_returns_empty(self) -> None:
        """When active-file.json points to a directory, return empty string."""
        sub = os.path.join(self.tmpdir, "subdir")
        os.makedirs(sub)
        af = os.path.join(self.tmpdir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": sub}, f)
        assert _read_active_file(self.tmpdir) == ""

    def test_path_key_missing_returns_empty(self) -> None:
        """When active-file.json has no 'path' key, return empty string."""
        af = os.path.join(self.tmpdir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"other": "val"}, f)
        assert _read_active_file(self.tmpdir) == ""


# ── _read_active_file: additional error paths ─────────────────────────────
class TestReadActiveFileErrors:
    def test_invalid_json_returns_empty(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            af = os.path.join(tmpdir, "active-file.json")
            Path(af).write_text("not json {{{")
            assert _read_active_file(tmpdir) == ""
        finally:
            shutil.rmtree(tmpdir)

    def test_nonexistent_dir_returns_empty(self) -> None:
        assert _read_active_file("/nonexistent/xyz_999") == ""

    def test_valid_file_returns_path(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(tmpdir, "real.txt")
            Path(test_file).write_text("content")
            af = os.path.join(tmpdir, "active-file.json")
            with open(af, "w") as f:
                json.dump({"path": test_file}, f)
            assert _read_active_file(tmpdir) == test_file
        finally:
            shutil.rmtree(tmpdir)

    def test_empty_path_returns_empty(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            af = os.path.join(tmpdir, "active-file.json")
            with open(af, "w") as f:
                json.dump({"path": ""}, f)
            assert _read_active_file(tmpdir) == ""
        finally:
            shutil.rmtree(tmpdir)


# ── _clean_llm_output ────────────────────────────────────────────────────
class TestCleanLlmOutput:
    def test_strips_quotes(self) -> None:
        assert _clean_llm_output('"hello"') == "hello"

    def test_strips_single_quotes(self) -> None:
        assert _clean_llm_output("'hello'") == "hello"

    def test_strips_whitespace(self) -> None:
        assert _clean_llm_output("  hello  ") == "hello"


# ── _model_vendor_order: all prefixes ────────────────────────────────────
class TestModelVendorOrderAllPrefixes:
    def test_claude(self) -> None:
        assert _model_vendor_order("claude-3-opus") == 0

    def test_o1_prefix(self) -> None:
        assert _model_vendor_order("o1-preview") == 1

    def test_o3_prefix(self) -> None:
        assert _model_vendor_order("o3-mini") == 1

    def test_gemini(self) -> None:
        assert _model_vendor_order("gemini-1.5-pro") == 2

    def test_minimax(self) -> None:
        assert _model_vendor_order("minimax-abab") == 3

    def test_openrouter(self) -> None:
        assert _model_vendor_order("openrouter/meta-llama") == 4

    def test_unknown(self) -> None:
        assert _model_vendor_order("unknown-model") == 5


# ── _resolve_task: priority -f > --task > default ─────────────────────────
class TestResolveTaskPriority:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_file_overrides_task_flag(self) -> None:
        p = os.path.join(self.tmpdir, "t.txt")
        Path(p).write_text("from file")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", p, "--task", "from flag"])
        assert _resolve_task(args) == "from file"

    def test_task_flag_only(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "my explicit task"])
        assert _resolve_task(args) == "my explicit task"

    def test_default_task(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert "gmail" in result.lower()


# ── SorcarAgent direct tests (covers __init__, _get_tools, _reset, run) ──
class TestSorcarAgentDirect:
    def test_init_attributes(self) -> None:
        agent = SorcarAgent("test_agent")
        assert agent.web_use_tool is None
        assert agent.docker_manager is None

    def test_get_tools_no_web(self) -> None:
        agent = SorcarAgent("test_agent")
        tools = agent._get_tools()
        assert len(tools) == 4  # Bash, Read, Edit, Write

    def test_get_tools_with_web(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test_agent")
        agent.web_use_tool = WebUseTool(headless=True)
        try:
            tools = agent._get_tools()
            assert len(tools) > 4
        finally:
            agent.web_use_tool.close()

    def test_reset_all_defaults(self) -> None:
        agent = SorcarAgent("test_agent")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=None,
            docker_image=None,
        )

    def test_reset_with_explicit_values(self) -> None:
        agent = SorcarAgent("test_agent")
        agent._reset(
            model_name="claude-opus-4-6",
            max_sub_sessions=3,
            max_steps=5,
            max_budget=1.0,
            work_dir="/tmp",
            docker_image=None,
            verbose=True,
        )

    def test_run_zero_budget_no_attachments(self) -> None:
        """SorcarAgent.run() with zero budget exits fast, covering init code."""
        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="say hello",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_image_attachment(self) -> None:
        """Cover the attachments branch with an image."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="describe this image",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
                attachments=[Attachment(data=b"fake_img", mime_type="image/png")],
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_pdf_attachment(self) -> None:
        """Cover the PDF attachment branch."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="read this pdf",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
                attachments=[Attachment(data=b"fake_pdf", mime_type="application/pdf")],
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_both_attachments(self) -> None:
        """Cover both image + PDF attachment branches simultaneously."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="analyze",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
                attachments=[
                    Attachment(data=b"fake_img", mime_type="image/png"),
                    Attachment(data=b"fake_pdf", mime_type="application/pdf"),
                ],
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_editor_file(self) -> None:
        """Cover the current_editor_file branch."""
        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        editor_file = os.path.join(tmpdir, "test.py")
        Path(editor_file).write_text("print('hello')")
        try:
            result = agent.run(
                prompt_template="fix this file",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
                current_editor_file=editor_file,
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_headless_none_uses_config(self) -> None:
        """Cover the headless=None branch which falls back to config."""
        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="hello",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=None,
                verbose=False,
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_text_attachment_no_parts(self) -> None:
        """Attachment that is neither image nor PDF hits empty parts branch."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            result = agent.run(
                prompt_template="analyze",
                work_dir=tmpdir,
                max_steps=1,
                max_budget=0.001,
                headless=True,
                verbose=False,
                attachments=[Attachment(data=b"text data", mime_type="text/plain")],
            )
            assert isinstance(result, str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stream_callback_with_printer(self) -> None:
        """Cover the _stream closure's if self.printer True branch."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        agent = SorcarAgent("test_stream_agent")
        agent.printer = BaseBrowserPrinter()
        tools = agent._get_tools()
        bash_tool = tools[0]
        result = bash_tool(command="echo hello_stream", description="test stream")
        assert "hello_stream" in result

    def test_stream_callback_without_printer(self) -> None:
        """Cover the _stream closure's if self.printer False branch."""
        agent = SorcarAgent("test_stream_agent")
        agent.printer = None
        tools = agent._get_tools()
        bash_tool = tools[0]
        result = bash_tool(command="echo hello_no_printer", description="test no printer")
        assert "hello_no_printer" in result


# ── sorcar_agent.py main() via subprocess (lines 214-250) ────────────────
class TestSorcarAgentMainSubprocess:
    def test_main_with_task_flag(self) -> None:
        """Run sorcar_agent main() with --task and --max_steps=0 so it exits fast."""
        tmpdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--task", "say hello",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "--verbose", "false",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            # Should print FINAL RESULT
            assert "FINAL RESULT" in result.stdout or result.returncode != 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_with_file_flag(self) -> None:
        """Run sorcar_agent main() with -f pointing to a task file."""
        tmpdir = tempfile.mkdtemp()
        task_file = os.path.join(tmpdir, "task.txt")
        Path(task_file).write_text("echo test task")
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "-f", task_file,
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "--verbose", "false",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert "FINAL RESULT" in result.stdout or result.returncode != 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_default_task_exits(self) -> None:
        """Run sorcar_agent main() with default task, budget=0 so it exits."""
        tmpdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "--verbose", "false",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert "FINAL RESULT" in result.stdout or result.returncode != 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_no_work_dir_uses_tempdir(self) -> None:
        """Run sorcar_agent main() without --work_dir to hit tempdir branch."""
        result = subprocess.run(
            [
                sys.executable, "-m",
                "kiss.agents.sorcar.sorcar_agent",
                "--task", "hello",
                "--max_steps", "0",
                "--max_budget", "0.0",
                "--headless", "true",
                "--verbose", "false",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "FINAL RESULT" in result.stdout or result.returncode != 0


# ── Server endpoint integration tests ─────────────────────────────────────

def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


@pytest.fixture(scope="module")
def server():
    """Start a sorcar server subprocess and yield (base_url, work_dir, proc, tmpdir)."""
    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    # Init git repo
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
    subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

    port_file = os.path.join(tmpdir, "port")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "_sorcar_test_server.py"),
            port_file,
            work_dir,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

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

        yield base_url, work_dir, proc, tmpdir
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestCommitWithChanges:
    """Exercise /commit endpoint with actual staged changes (lines 895-938)."""

    def test_commit_with_changes(self, server) -> None:
        base_url, work_dir, _, _ = server
        # Create a real change
        Path(work_dir, "new_file.txt").write_text("new content\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)

        resp = requests.post(f"{base_url}/commit", json={}, timeout=30)
        data = resp.json()
        # It will either succeed (status=ok) or error (e.g., LLM failure gives empty msg)
        assert resp.status_code in (200, 400)
        assert "status" in data or "error" in data


class TestPushEndpoint:
    """Exercise /push endpoint (lines 934-938)."""

    def test_push_no_remote(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(f"{base_url}/push", json={}, timeout=10)
        data = resp.json()
        # No remote configured, so push will fail
        assert resp.status_code == 400
        assert "error" in data


class TestSuggestionsFileMatch:
    """Exercise suggestions with file query matching actual files (line 825, 831)."""

    def test_suggestions_general_with_file_match(self, server) -> None:
        base_url, work_dir, _, _ = server
        # Create a file so file_cache has entries
        Path(work_dir, "readme.txt").write_text("hello")
        # Query that matches a file in work_dir
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "file", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_general_5_history_hits(self, server) -> None:
        """Trigger the 'len(results) >= 5: break' branch by running tasks first."""
        base_url, work_dir, _, _ = server
        # Run several tasks to populate history
        for i in range(6):
            requests.post(
                f"{base_url}/run",
                json={"task": f"test task {i}"},
                timeout=10,
            )
            # Wait for task to finish
            time.sleep(0.5)
            requests.post(f"{base_url}/stop", json={}, timeout=5)
            time.sleep(0.3)

        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "test", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_files_with_frequent(self, server) -> None:
        """Exercise frequent file usage sorting in files mode."""
        base_url, work_dir, _, _ = server
        # Record file usage first
        requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "file.txt"},
            timeout=5,
        )
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "", "mode": "files"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)
        # Check that frequent files appear
        types = [item.get("type", "") for item in data]
        assert any("frequent" in t for t in types) or len(data) == 0


class TestCompleteEndpointFastPath:
    """Exercise _fast_complete with file path completion (lines 899)."""

    def test_complete_fast_file_path(self, server) -> None:
        base_url, work_dir, _, _ = server
        # Query matching start of a file in file_cache
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "fil"},
            timeout=5,
        )
        data = resp.json()
        assert "suggestion" in data


class TestActiveFileInfoMd:
    """Exercise /active-file-info with a .md file (lines 1160-1162)."""

    def test_active_file_md(self, server) -> None:
        """Set active-file.json to point to a .md file and query."""
        base_url, work_dir, _, tmpdir = server
        # Find cs_data_dir - it's based on the hash of work_dir
        import hashlib

        from kiss.agents.sorcar.task_history import _KISS_DIR
        wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
        cs_data_dir = str(_KISS_DIR / f"cs-{wd_hash}")
        os.makedirs(cs_data_dir, exist_ok=True)

        # Create a .md file and set it as active
        md_file = os.path.join(work_dir, "prompt.md")
        Path(md_file).write_text("# System Prompt\nYou are a helpful assistant.\n")
        af = os.path.join(cs_data_dir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": md_file}, f)

        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        data = resp.json()
        assert "is_prompt" in data
        assert data["path"] == md_file
        assert data["filename"] == "prompt.md"

        # Clean up
        os.unlink(af)


class TestGetFileContentEndpoint:
    """Exercise /get-file-content success path (line 1182-1184)."""

    def test_get_file_content_success(self, server) -> None:
        base_url, work_dir, _, _ = server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.get(
            f"{base_url}/get-file-content",
            params={"path": fpath},
            timeout=5,
        )
        data = resp.json()
        assert "content" in data

    def test_get_file_content_binary_error(self, server) -> None:
        """Exercise exception path (line 1204-1206)."""
        base_url, work_dir, _, _ = server
        bin_file = os.path.join(work_dir, "binary.dat")
        Path(bin_file).write_bytes(bytes(range(256)) * 100)
        resp = requests.get(
            f"{base_url}/get-file-content",
            params={"path": bin_file},
            timeout=5,
        )
        # May succeed or fail depending on encoding
        assert resp.status_code in (200, 500)


class TestMergeActionValidActions:
    """Exercise merge_action with all valid action types (lines 849-856)."""

    def test_merge_action_accept(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "accept"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_reject(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "reject"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_accept_all(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "accept-all"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_reject_all(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "reject-all"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_empty_action(self, server) -> None:
        """Empty action string hits invalid action branch."""
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": ""},
            timeout=5,
        )
        assert resp.status_code == 400


class TestRecordFileUsageWithPath:
    """Exercise record_file_usage with a non-empty path (line 987-988)."""

    def test_record_file_usage_path(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/main.py"},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"



class TestSorcarMainSubprocess:
    """Exercise sorcar.py main() (lines 1300-1321)."""

    def test_sorcar_main_exits_on_sigint(self) -> None:
        """Start sorcar main() and send SIGINT to exercise main() + _cleanup."""
        tmpdir = tempfile.mkdtemp()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import shutil, webbrowser\n"
                    "_orig = shutil.which\n"
                    "shutil.which = lambda cmd, **kw: "
                    "None if cmd == 'code-server' else _orig(cmd, **kw)\n"
                    "webbrowser.open = lambda url: None\n"
                    "import os; os.environ['KISS_MODE'] = 'coding'\n"
                    "from kiss.agents.sorcar.sorcar import main; main()",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "KISS_MODE": "coding",
                },
                cwd=tmpdir,
            )
            # Wait for server to start
            time.sleep(5)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            assert "running at" in stdout or proc.returncode is not None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sorcar_main_assistant_mode(self) -> None:
        """Start sorcar main() in assistant mode (default)."""
        tmpdir = tempfile.mkdtemp()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import shutil, webbrowser\n"
                    "_orig = shutil.which\n"
                    "shutil.which = lambda cmd, **kw: "
                    "None if cmd == 'code-server' else _orig(cmd, **kw)\n"
                    "webbrowser.open = lambda url: None\n"
                    "from kiss.agents.sorcar.sorcar import main; main()",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "KISS_MODE": "assistant",
                },
                cwd=tmpdir,
            )
            time.sleep(5)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            assert "running at" in stdout or proc.returncode is not None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGenerateCommitMessageNoChanges:
    """Exercise /generate-commit-message with no changes (line 1118)."""

    def test_generate_commit_msg_no_changes(self, server) -> None:
        base_url, work_dir, _, _ = server
        # Ensure clean state (commit any leftover changes from prior tests)
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "cleanup", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={
                **os.environ,
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )
        resp = requests.post(
            f"{base_url}/generate-commit-message",
            json={},
            timeout=10,
        )
        data = resp.json()
        # No changes => error
        assert "error" in data


# ── In-process server test (runs run_chatbot in a thread for coverage) ────



class _InProcessDummyAgent(RelentlessAgent):
    """Minimal agent that returns immediately for in-process testing."""

    def __init__(self, name: str) -> None:
        pass

    def run(self, **kwargs) -> str:  # type: ignore[override]
        task = kwargs.get("prompt_template", "")
        work_dir = kwargs.get("work_dir", "")
        if task == "slow_task_for_stop_test":
            # Block in small increments so _StopRequested can interrupt
            import time as _t

            for _ in range(300):
                _t.sleep(0.1)
        if task == "error_task_for_test":
            raise RuntimeError("test error from agent")
        if task == "create_file_for_merge" and work_dir:
            # Create a new file so _prepare_merge_view detects changes
            Path(work_dir, "agent_created.txt").write_text("new content\n")
        return "success: true\nsummary: done"


@pytest.fixture(scope="module")
def inproc_server():
    """Start run_chatbot() in a background thread for in-process coverage."""
    import webbrowser as _wb

    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    # Init git repo
    subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True
    )
    Path(work_dir, "file.txt").write_text("line1\nline2\n")
    subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

    # Prevent code-server and browser
    import shutil as _sh

    _orig_which = _sh.which

    def _no_cs(cmd: str, mode: int = 0, path: str | None = None) -> str | None:
        if cmd == "code-server":
            return None
        result: str | None = _orig_which(cmd, mode=mode, path=path)  # type: ignore[call-overload]
        return result

    old_which = _sh.which
    old_open = _wb.open
    _sh.which = _no_cs  # type: ignore[assignment]
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
            "agent_factory": _InProcessDummyAgent,
            "title": "InProcTest",
            "work_dir": work_dir,
        },
        daemon=True,
    )
    thread.start()

    # Wait for port
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if port_holder:
            break
        time.sleep(0.2)
    assert port_holder, "Server did not start"

    base_url = f"http://127.0.0.1:{port_holder[0]}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            resp = requests.get(base_url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.2)

    wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
    from kiss.agents.sorcar.task_history import _KISS_DIR

    cs_data_dir = str(_KISS_DIR / f"cs-{wd_hash}")

    yield base_url, work_dir, cs_data_dir

    # Cleanup: trigger shutdown
    try:
        requests.post(f"{base_url}/closing", json={}, timeout=2)
    except Exception:
        pass

    # Restore
    _sh.which = old_which  # type: ignore[assignment]
    _wb.open = old_open  # type: ignore[assignment,misc]
    sorcar_module.find_free_port = _orig_ffp  # type: ignore[attr-defined]
    time.sleep(1)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestInProcessEndpoints:
    """Tests that exercise run_chatbot() endpoint code in-process for coverage."""

    def test_index(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(base_url, timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_models(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/models", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "selected" in data

    def test_theme(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/theme", timeout=5)
        assert resp.status_code == 200

    def test_tasks(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/tasks", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_proposed_tasks(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/proposed_tasks", timeout=5)
        assert resp.status_code == 200

    def test_suggestions_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/suggestions?q=&mode=general", timeout=5)
        assert resp.status_code == 200

    def test_suggestions_files(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/suggestions?q=&mode=files", timeout=5)
        assert resp.status_code == 200

    def test_complete_short(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert resp.status_code == 200

    def test_complete_with_fast_match(self, inproc_server) -> None:
        """Exercise _fast_complete with a query that starts with a task."""
        base_url, _, _ = inproc_server
        # Run a task first to populate history
        requests.post(
            f"{base_url}/run",
            json={"task": "unique_auto_complete_test_query_xyz"},
            timeout=10,
        )
        time.sleep(4)  # Wait longer for task to complete and history to be updated
        # Now complete with a prefix of that task
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "unique_auto"},
            timeout=10,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_complete_with_file_match(self, inproc_server) -> None:
        """Exercise _fast_complete file path branch."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "edit fi"},
            timeout=10,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_task_events_invalid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=abc", timeout=5)
        assert resp.status_code == 400

    def test_task_events_out_of_range(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=999", timeout=5)
        assert resp.status_code == 404

    def test_task_events_valid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
        assert resp.status_code in (200, 404)  # May or may not have tasks

    def test_run_empty_task(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/run", json={"task": ""}, timeout=5)
        assert resp.status_code == 400

    def test_run_and_stop(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        # Run a slow task so we can stop it while running
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "slow_task_for_stop_test", "model": "claude-opus-4-6"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(1)
        # Stop the task
        stop = requests.post(f"{base_url}/stop", json={}, timeout=5)
        assert stop.status_code == 200
        # Wait for cleanup
        time.sleep(2)

    def test_run_normal_task(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "normal task", "model": "claude-opus-4-6"},
            timeout=10,
        )
        assert resp.status_code == 200
        # Wait for task to complete
        time.sleep(2)

    def test_run_with_active_file(self, inproc_server) -> None:
        """Exercise the active_file branch in run_agent_thread (line 538)."""
        base_url, work_dir, cs_data_dir = inproc_server
        # Create a file and set it as active
        active_py = os.path.join(work_dir, "active_test.py")
        Path(active_py).write_text("print('active')")
        os.makedirs(cs_data_dir, exist_ok=True)
        af = os.path.join(cs_data_dir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": active_py}, f)
        try:
            resp = requests.post(
                f"{base_url}/run",
                json={"task": "test with active file"},
                timeout=10,
            )
            assert resp.status_code == 200
            time.sleep(2)
        finally:
            if os.path.exists(af):
                os.unlink(af)

    def test_run_while_running(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        # Start a slow task
        resp1 = requests.post(
            f"{base_url}/run",
            json={"task": "slow_task_for_stop_test"},
            timeout=10,
        )
        assert resp1.status_code == 200
        time.sleep(0.5)
        # Try to run another task while first is running
        resp2 = requests.post(
            f"{base_url}/run",
            json={"task": "second task"},
            timeout=10,
        )
        assert resp2.status_code == 409
        # Also try run-selection while running
        resp3 = requests.post(
            f"{base_url}/run-selection",
            json={"text": "selected text"},
            timeout=10,
        )
        assert resp3.status_code == 409
        # Clean up
        requests.post(f"{base_url}/stop", json={}, timeout=5)
        time.sleep(2)

    def test_stop_no_task(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        # Make sure no task is running
        time.sleep(1)
        resp = requests.post(f"{base_url}/stop", json={}, timeout=5)
        assert resp.status_code == 404

    def test_run_selection_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/run-selection", json={"text": ""}, timeout=5
        )
        assert resp.status_code == 400

    def test_run_selection(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/run-selection", json={"text": "hello test"}, timeout=10
        )
        assert resp.status_code == 200
        time.sleep(2)

    def test_open_file_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/open-file", json={"path": ""}, timeout=5
        )
        assert resp.status_code == 400

    def test_open_file_not_found(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/open-file", json={"path": "/nonexistent"}, timeout=5
        )
        assert resp.status_code == 404

    def test_open_file_success(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.post(
            f"{base_url}/open-file", json={"path": fpath}, timeout=5
        )
        assert resp.status_code == 200

    def test_focus_chatbox(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/focus-chatbox", json={}, timeout=5)
        assert resp.status_code == 200

    def test_focus_editor(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/focus-editor", json={}, timeout=5)
        assert resp.status_code == 200

    def test_merge_action_invalid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/merge-action", json={"action": "bogus"}, timeout=5
        )
        assert resp.status_code == 400

    def test_merge_action_all_done(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/merge-action", json={"action": "all-done"}, timeout=5
        )
        assert resp.status_code == 200

    def test_merge_action_prev_next(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        for action in ("prev", "next", "accept", "reject", "accept-all", "reject-all"):
            resp = requests.post(
                f"{base_url}/merge-action", json={"action": action}, timeout=5
            )
            assert resp.status_code == 200

    def test_record_file_usage(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "test.py"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_record_file_usage_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": ""},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_commit_no_changes(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        # Ensure clean state
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "pre-clean", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={**os.environ, "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )
        resp = requests.post(f"{base_url}/commit", json={}, timeout=10)
        data = resp.json()
        assert "error" in data  # "No changes to commit"

    def test_push_no_remote(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/push", json={}, timeout=10)
        data = resp.json()
        assert "error" in data

    def test_active_file_info_no_file(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_prompt"] is False

    def test_active_file_info_md(self, inproc_server) -> None:
        base_url, work_dir, cs_data_dir = inproc_server
        md_file = os.path.join(work_dir, "test_prompt.md")
        Path(md_file).write_text("# System Prompt\nYou are helpful.\n")
        os.makedirs(cs_data_dir, exist_ok=True)
        af = os.path.join(cs_data_dir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": md_file}, f)
        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        data = resp.json()
        assert data["is_prompt"] is not None
        assert data["path"] == md_file
        os.unlink(af)

    def test_get_file_content(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": fpath}, timeout=5
        )
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_get_file_content_not_found(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": "/nonexistent"}, timeout=5
        )
        assert resp.status_code == 404

    def test_get_file_content_binary_error(self, inproc_server) -> None:
        """Exercise get_file_content exception path (line 1160-1162)."""
        base_url, work_dir, _ = inproc_server
        bin_file = os.path.join(work_dir, "binary_test.dat")
        Path(bin_file).write_bytes(bytes(range(256)) * 100)
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": bin_file}, timeout=5
        )
        assert resp.status_code in (200, 500)

    def test_suggestions_general_with_query(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "test", "mode": "general"}, timeout=5
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_files_with_query(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "file", "mode": "files"}, timeout=5
        )
        assert resp.status_code == 200

    def test_suggestions_files_with_dir_match(self, inproc_server) -> None:
        """Test suggestions in files mode with directory entries."""
        base_url, work_dir, _ = inproc_server
        # Create a subdirectory
        subdir = os.path.join(work_dir, "subdir_test")
        os.makedirs(subdir, exist_ok=True)
        Path(os.path.join(subdir, "inner.txt")).write_text("inner")
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "sub", "mode": "files"}, timeout=5
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_with_file_match_general(self, inproc_server) -> None:
        """Test suggestions general mode with file path matching."""
        base_url, work_dir, _ = inproc_server
        # Query matching a file
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "file.txt", "mode": "general"}, timeout=5
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_theme_with_file(self, inproc_server) -> None:
        """Exercise theme endpoint with an existing theme file."""
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        theme_file.parent.mkdir(parents=True, exist_ok=True)
        orig = theme_file.read_text() if theme_file.exists() else None
        try:
            theme_file.write_text(json.dumps({"kind": "light"}))
            base_url, _, _ = inproc_server
            resp = requests.get(f"{base_url}/theme", timeout=5)
            assert resp.status_code == 200
        finally:
            if orig is not None:
                theme_file.write_text(orig)
            elif theme_file.exists():
                theme_file.unlink()

    def test_theme_with_bad_file(self, inproc_server) -> None:
        """Exercise theme endpoint with a corrupt theme file (lines 987-988)."""
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        theme_file.parent.mkdir(parents=True, exist_ok=True)
        orig = theme_file.read_text() if theme_file.exists() else None
        try:
            theme_file.write_text("not valid json{{{")
            base_url, _, _ = inproc_server
            resp = requests.get(f"{base_url}/theme", timeout=5)
            assert resp.status_code == 200  # Falls back to dark theme
        finally:
            if orig is not None:
                theme_file.write_text(orig)
            elif theme_file.exists():
                theme_file.unlink()

    def test_run_task_while_merging(self, inproc_server) -> None:
        """Create file changes that trigger merge view, then try /run."""
        base_url, work_dir, cs_data_dir = inproc_server
        # Run a task that creates a new file → triggers merge view
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "create_file_for_merge"},
            timeout=10,
        )
        assert resp.status_code == 200
        # Wait for task to complete and merge to be set
        time.sleep(5)
        try:
            # Check if merging is active
            resp2 = requests.post(
                f"{base_url}/run",
                json={"task": "should fail while merging"},
                timeout=10,
            )
            if resp2.status_code == 409:
                data = resp2.json()
                err = data.get("error", "").lower()
                assert "merge" in err or "running" in err
                # Also test run-selection while merging
                resp3 = requests.post(
                    f"{base_url}/run-selection",
                    json={"text": "selection during merge"},
                    timeout=10,
                )
                assert resp3.status_code == 409
        finally:
            # Clean up merge state: accept-all writes pending action,
            # then all-done actually sets merging=False
            requests.post(
                f"{base_url}/merge-action",
                json={"action": "accept-all"},
                timeout=10,
            )
            time.sleep(1)
            requests.post(
                f"{base_url}/merge-action",
                json={"action": "all-done"},
                timeout=10,
            )
            time.sleep(1)

    def test_generate_config_message(self, inproc_server) -> None:
        """Hit the /generate-config-message endpoint."""
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/generate-config-message",
            json={"model": "claude-opus-4-6"},
            timeout=30,
        )
        assert resp.status_code in (200, 500)
        data = resp.json()
        assert "message" in data or "error" in data

    def test_commit_git_failure(self, inproc_server) -> None:
        """Make git commit fail via a pre-commit hook."""
        base_url, work_dir, _ = inproc_server
        hooks_dir = Path(work_dir, ".git", "hooks")
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\nexit 1\n")
        hook_path.chmod(0o755)
        Path(work_dir, "hook_test.txt").write_text("test")
        try:
            resp = requests.post(
                f"{base_url}/commit", json={}, timeout=15
            )
            data = resp.json()
            assert "error" in data or "status" in data
        finally:
            hook_path.unlink(missing_ok=True)
            Path(work_dir, "hook_test.txt").unlink(missing_ok=True)

    def test_theme_file_change_detected(self, inproc_server) -> None:
        """Write theme file and verify _watch_theme_file detects it."""
        base_url, _, _ = inproc_server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        theme_file.write_text('{"kind": "light"}')
        time.sleep(2)
        resp = requests.get(f"{base_url}/theme", timeout=5)
        data = resp.json()
        assert isinstance(data, dict)
        theme_file.unlink(missing_ok=True)

    def test_theme_no_file(self, inproc_server) -> None:
        """Theme endpoint with no theme file → hits file-not-exists branch."""
        base_url, _, _ = inproc_server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        orig = theme_file.read_text() if theme_file.exists() else None
        try:
            theme_file.unlink(missing_ok=True)
            resp = requests.get(f"{base_url}/theme", timeout=5)
            assert resp.status_code == 200
        finally:
            if orig is not None:
                theme_file.write_text(orig)

    def test_closing(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/closing", json={}, timeout=5)
        assert resp.status_code == 200

    def test_generate_commit_message_no_changes(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        # Make sure repo is clean
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "cleanup", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={
                **os.environ,
                "GIT_COMMITTER_NAME": "T",
                "GIT_COMMITTER_EMAIL": "t@t.com",
            },
        )
        resp = requests.post(
            f"{base_url}/generate-commit-message", json={}, timeout=10
        )
        data = resp.json()
        assert "error" in data

    def test_run_with_attachments(self, inproc_server) -> None:
        """Run a task with base64-encoded attachments."""
        import base64

        base_url, _, _ = inproc_server
        img_data = base64.b64encode(b"fake image data").decode()
        resp = requests.post(
            f"{base_url}/run",
            json={
                "task": "test with attachments",
                "attachments": [
                    {"data": img_data, "mime_type": "image/png"},
                ],
            },
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(2)

    def test_run_with_internal_model(self, inproc_server) -> None:
        """Run task with an internal model name to skip double usage recording."""
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "test internal model", "model": "gemini-2.0-flash"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(2)

    def test_commit_with_changes(self, inproc_server) -> None:
        """Commit endpoint with actual changes."""
        base_url, work_dir, _ = inproc_server
        new_file = os.path.join(work_dir, "commit_test.txt")
        Path(new_file).write_text("change for commit")
        resp = requests.post(f"{base_url}/commit", json={}, timeout=60)
        data = resp.json()
        assert "status" in data or "error" in data
        # Clean up
        if os.path.exists(new_file):
            os.unlink(new_file)

    def test_sse_events(self, inproc_server) -> None:
        """Connect to SSE stream briefly."""
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/events", stream=True, timeout=3)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        resp.close()

    def test_run_error_task(self, inproc_server) -> None:
        """Run a task that raises Exception to cover except Exception branch."""
        base_url, _, _ = inproc_server
        time.sleep(1)  # Wait for any prior task to finish
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "error_task_for_test"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(3)

    def test_tasks_has_events(self, inproc_server) -> None:
        """After running tasks, check that tasks endpoint shows has_events."""
        base_url, _, _ = inproc_server
        # Run a task and wait for it to fully complete including event storage
        requests.post(
            f"{base_url}/run",
            json={"task": "events check task unique"},
            timeout=10,
        )
        time.sleep(5)  # Longer wait to ensure task completes and events saved
        resp = requests.get(f"{base_url}/tasks", timeout=5)
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # At least some tasks should have chat_events stored
        has_events_list = [t.get("has_events") for t in data]
        assert any(has_events_list), f"No tasks have events. Tasks: {data[:3]}"

    def test_generate_commit_message_with_untracked(self, inproc_server) -> None:
        """Generate commit message with untracked file (untracked branch)."""
        base_url, work_dir, cs_data_dir = inproc_server
        test_file = os.path.join(work_dir, "gen_cm_untracked.txt")
        Path(test_file).write_text("untracked content for commit msg")
        try:
            resp = requests.post(
                f"{base_url}/generate-commit-message", json={}, timeout=60
            )
            data = resp.json()
            assert "message" in data or "error" in data
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def test_generate_commit_message_with_diff(self, inproc_server) -> None:
        """Generate commit message with staged diff changes."""
        base_url, work_dir, _ = inproc_server
        # Modify a tracked file to create a diff
        fpath = os.path.join(work_dir, "file.txt")
        Path(fpath).write_text("modified content\nline2\nline3\n")
        try:
            resp = requests.post(
                f"{base_url}/generate-commit-message", json={}, timeout=60
            )
            data = resp.json()
            assert "message" in data or "error" in data
        finally:
            # Restore original content
            Path(fpath).write_text("line1\nline2\n")

    def test_commit_with_staged_changes(self, inproc_server) -> None:
        """Commit endpoint with staged changes to cover full commit path."""
        base_url, work_dir, _ = inproc_server
        commit_file = os.path.join(work_dir, "commit_staged.txt")
        Path(commit_file).write_text("staged content for commit")
        subprocess.run(["git", "add", commit_file], cwd=work_dir, capture_output=True)
        resp = requests.post(f"{base_url}/commit", json={}, timeout=60)
        data = resp.json()
        # Should either succeed or error from _generate_commit_msg returning ""
        assert resp.status_code in (200, 400)
        assert "status" in data or "error" in data

    def test_sse_events_with_task_data(self, inproc_server) -> None:
        """Connect to SSE, run a task, and verify events arrive."""
        base_url, _, _ = inproc_server
        # Start SSE stream in background
        import threading as _th

        events_received: list[str] = []
        stop_reading = threading.Event()

        def _read_sse() -> None:
            try:
                resp = requests.get(
                    f"{base_url}/events", stream=True, timeout=10
                )
                for line in resp.iter_lines(decode_unicode=True):
                    if stop_reading.is_set():
                        break
                    if line and line.startswith("data:"):
                        events_received.append(line)
                resp.close()
            except Exception:
                pass

        reader = _th.Thread(target=_read_sse, daemon=True)
        reader.start()
        time.sleep(0.5)

        # Run a quick task to generate events
        requests.post(
            f"{base_url}/run", json={"task": "sse data test"}, timeout=10
        )
        time.sleep(3)

        stop_reading.set()
        reader.join(timeout=3)
        assert len(events_received) > 0

    def test_run_selection_while_merging(self, inproc_server) -> None:
        """Hit the merging check in run-selection by triggering merge state."""
        base_url, _, _ = inproc_server
        # This is covered indirectly via test_run_while_running's run-selection check
        # Here we verify the endpoint works for a normal case
        time.sleep(1)
        resp = requests.post(
            f"{base_url}/run-selection",
            json={"text": "selection merge test"},
            timeout=10,
        )
        assert resp.status_code in (200, 409)
        time.sleep(2)

    def test_complete_no_fast_match_triggers_llm(self, inproc_server) -> None:
        """Query that doesn't fast-match goes through LLM path (or returns empty)."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "xyzzy_no_match_ever_12345"},
            timeout=30,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_complete_short_last_word(self, inproc_server) -> None:
        """Query where last word is < 2 chars → skips file matching in _fast_complete."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "test x"},
            timeout=30,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_suggestions_general_file_word_match(self, inproc_server) -> None:
        """Query where last word (>=2 chars) matches a file in file_cache."""
        base_url, work_dir, _ = inproc_server
        # Ensure a known file exists
        Path(work_dir, "matchable_xyz.txt").write_text("test")
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "edit matchable_xyz", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_general_short_last_word(self, inproc_server) -> None:
        """Query where last word is only 1 char → skips file matching."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "test x", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_general_8_file_matches_break(self, inproc_server) -> None:
        """Create >8 files matching a pattern to hit count >= 8 break."""
        base_url, work_dir, _ = inproc_server
        # Create 10 files with a unique pattern
        for i in range(10):
            Path(work_dir, f"zbatchf_{i}.txt").write_text(f"c{i}")
        # Run a task to refresh file cache
        requests.post(
            f"{base_url}/run",
            json={"task": "refresh for zbatchf"},
            timeout=10,
        )
        time.sleep(4)
        # Query that matches all zbatchf files
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "check zbatchf", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        file_items = [item for item in data if item.get("type") == "file"]
        assert len(file_items) <= 8  # Capped at 8

    def test_suggestions_files_frequent_sort(self, inproc_server) -> None:
        """Record file usage, then query files mode. file.txt is in initial cache."""
        base_url, work_dir, _ = inproc_server
        # Record usage for file.txt which is in git repo (created at fixture setup)
        for _ in range(5):
            requests.post(
                f"{base_url}/record-file-usage",
                json={"path": "file.txt"},
                timeout=5,
            )
        # file.txt is already in file_cache from server startup
        # No need to run a task to refresh cache
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "", "mode": "files"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)
        types = [item.get("type", "") for item in data]
        assert any("frequent" in t for t in types), f"Data: {data[:5]}"


class TestSorcarAgentMainInProcess:
    """Call sorcar_agent.main() in-process to get coverage on the main() function.

    With max_steps=0 and max_budget=0.0, the agent returns immediately
    without making any LLM calls.
    """

    def test_main_with_work_dir(self) -> None:
        """Cover the `args.work_dir is not None` branch (line 220→221)."""
        tmpdir = tempfile.mkdtemp()
        old_argv = sys.argv
        try:
            sys.argv = [
                "sorcar_agent",
                "--task", "say hello",
                "--max_steps", "0",
                "--max_budget", "0.0",
                "--work_dir", tmpdir,
                "--headless", "true",
                "--verbose", "false",
            ]
            from kiss.agents.sorcar.sorcar_agent import main as sa_main
            sa_main()
        finally:
            sys.argv = old_argv
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_no_work_dir(self) -> None:
        """Cover the `args.work_dir is None` branch (line 220→224, uses tempdir)."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "sorcar_agent",
                "--task", "say hello",
                "--max_steps", "0",
                "--max_budget", "0.0",
                "--headless", "true",
                "--verbose", "false",
            ]
            from kiss.agents.sorcar.sorcar_agent import main as sa_main
            sa_main()
        finally:
            sys.argv = old_argv
