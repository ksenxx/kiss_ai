"""Integration tests for sorcar.py run_chatbot HTTP endpoints.

Launches a real sorcar server as a subprocess and exercises each endpoint
to maximise branch coverage of sorcar.py.  No mocks or test doubles.
"""

from __future__ import annotations

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


def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    """Poll until port file is written and return the port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


@pytest.fixture(scope="module")
def server():
    """Start a sorcar server subprocess and yield (base_url, work_dir, proc)."""
    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    # Initialize a git repo so commit endpoint works
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

    # Launch via test server helper
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

        # Wait for the server to be responsive
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

        yield base_url, work_dir, proc
    finally:
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)

class TestServerTasks:
    def test_tasks_returns_list(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/tasks", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_task_events_invalid_index(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/task-events?idx=bad", timeout=5)
        assert resp.status_code == 400

    def test_task_events_out_of_range(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/task-events?idx=99999", timeout=5)
        assert resp.status_code == 404

    def test_task_events_valid_index(self, server):
        base_url, _, _ = server
        # Get current tasks first
        tasks_resp = requests.get(f"{base_url}/tasks", timeout=5)
        tasks = tasks_resp.json()
        if tasks:
            resp = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
            assert resp.status_code == 200


class TestServerSuggestions:
    def test_suggestions_empty_query(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/suggestions?q=", timeout=5)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_suggestions_files_mode(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/suggestions?q=file&mode=files", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_general_mode(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/suggestions?q=test+something", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestServerComplete:
    def test_complete_empty_query(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/complete?q=", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["suggestion"] == ""

    def test_complete_short_query(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["suggestion"] == ""


class TestServerTheme:
    def test_theme_endpoint(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/theme", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "bg" in data


class TestServerRunTask:
    def test_run_empty_task(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": ""},
            timeout=5,
        )
        assert resp.status_code == 400


class TestServerRunSelection:
    def test_run_selection_empty_text(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/run-selection",
            json={"text": ""},
            timeout=5,
        )
        assert resp.status_code == 400

class TestServerOpenFile:
    def test_open_file_no_path(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/open-file",
            json={"path": ""},
            timeout=5,
        )
        assert resp.status_code == 400

    def test_open_file_not_found(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/open-file",
            json={"path": "/no/such/file.txt"},
            timeout=5,
        )
        assert resp.status_code == 404

    def test_open_file_success(self, server):
        base_url, work_dir, _ = server
        resp = requests.post(
            f"{base_url}/open-file",
            json={"path": os.path.join(work_dir, "file.txt")},
            timeout=5,
        )
        assert resp.status_code == 200


class TestServerMergeAction:
    def test_merge_action_invalid(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "invalid"},
            timeout=5,
        )
        assert resp.status_code == 400

    def test_merge_action_valid(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "next"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_all_done(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "all-done"},
            timeout=5,
        )
        assert resp.status_code == 200


class TestServerFocusEndpoints:
    def test_focus_chatbox(self, server):
        base_url, _, _ = server
        resp = requests.post(f"{base_url}/focus-chatbox", timeout=5)
        assert resp.status_code == 200

    def test_focus_editor(self, server):
        base_url, _, _ = server
        resp = requests.post(f"{base_url}/focus-editor", timeout=5)
        assert resp.status_code == 200


class TestServerClosing:
    def test_closing_endpoint(self, server):
        base_url, _, _ = server
        resp = requests.post(f"{base_url}/closing", timeout=5)
        assert resp.status_code == 200


class TestServerRecordFileUsage:
    def test_record_file_usage(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/test.py"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_record_file_usage_empty(self, server):
        base_url, _, _ = server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": ""},
            timeout=5,
        )
        assert resp.status_code == 200


class TestServerActiveFileInfo:
    def test_active_file_info_no_active_file(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_prompt" in data


class TestServerGetFileContent:
    def test_get_file_content_success(self, server):
        base_url, work_dir, _ = server
        path = os.path.join(work_dir, "file.txt")
        resp = requests.get(
            f"{base_url}/get-file-content?path={path}",
            timeout=5,
        )
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_get_file_content_not_found(self, server):
        base_url, _, _ = server
        resp = requests.get(
            f"{base_url}/get-file-content?path=/no/such/file.txt",
            timeout=5,
        )
        assert resp.status_code == 404

class TestServerUIState:
    def test_get_ui_state_empty(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/ui-state", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_save_and_get_ui_state(self, server):
        base_url, _, _ = server
        state = {"editorPct": 65.5}
        resp = requests.post(
            f"{base_url}/ui-state",
            json=state,
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get(f"{base_url}/ui-state", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["editorPct"] == 65.5

    def test_save_ui_state_overwrites(self, server):
        base_url, _, _ = server
        requests.post(
            f"{base_url}/ui-state",
            json={"editorPct": 40},
            timeout=5,
        )
        requests.post(
            f"{base_url}/ui-state",
            json={"editorPct": 70},
            timeout=5,
        )
        resp = requests.get(f"{base_url}/ui-state", timeout=5)
        assert resp.json()["editorPct"] == 70


class TestServerSuggestionsFilesMode:
    def test_suggestions_files_empty_query(self, server):
        base_url, _, _ = server
        resp = requests.get(f"{base_url}/suggestions?q=&mode=files", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
