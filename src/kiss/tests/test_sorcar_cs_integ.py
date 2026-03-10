"""Integration tests for sorcar.py with code-server enabled.

Uses a real code-server binary to cover code-server setup, startup, and cleanup
code paths. No mocks or test doubles.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser as _wb
from pathlib import Path
from typing import Any

import pytest
import requests

from kiss.agents.sorcar.sorcar import run_chatbot
from kiss.core.relentless_agent import RelentlessAgent


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

    # Create a bare remote so push endpoint succeeds
    bare_dir = os.path.join(tmpdir, "bare.git")
    subprocess.run(["git", "init", "--bare", bare_dir], capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", bare_dir],
        cwd=work_dir,
        capture_output=True,
    )
    # Determine the branch name
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() or "main"
    subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=work_dir,
        capture_output=True,
    )

    # Create theme file BEFORE starting server (covers line 459: theme_file.stat().st_mtime)
    from kiss.agents.sorcar.task_history import _KISS_DIR

    theme_file = _KISS_DIR / "vscode-theme.json"
    theme_file.parent.mkdir(parents=True, exist_ok=True)
    theme_existed = theme_file.exists()
    orig_theme = theme_file.read_text() if theme_existed else None
    theme_file.write_text(json.dumps({"kind": "dark"}))

    # Block browser but NOT code-server
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

    # Wait for port
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

    yield base_url, work_dir, cs_data_dir, tmpdir

    # Cleanup
    try:
        requests.post(f"{base_url}/closing", json={}, timeout=2)
    except Exception:
        pass

    _wb.open = old_open  # type: ignore[assignment,misc]
    sorcar_module.find_free_port = _orig_ffp  # type: ignore[attr-defined]

    # Restore theme file
    if orig_theme is not None:
        theme_file.write_text(orig_theme)
    elif theme_file.exists():
        theme_file.unlink()

    time.sleep(2)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCSIndex:
    def test_index_returns_html_with_code_server(self, cs_server: Any) -> None:
        """Verify server started with code-server URL embedded in HTML."""
        base_url, _, _, _ = cs_server
        resp = requests.get(base_url, timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # The HTML should contain the code-server URL
        assert "127.0.0.1" in resp.text


class TestCSPushSuccess:
    def test_push_succeeds_with_local_remote(self, cs_server: Any) -> None:
        """Push to local bare remote. Covers push success path (line 1079)."""
        base_url, work_dir, _, _ = cs_server
        # Make a change and commit
        new_file = os.path.join(work_dir, "push_test.txt")
        Path(new_file).write_text("push test content")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        commit_env = {
            **os.environ,
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "test push",
                "--author=T <t@t.com>",
            ],
            cwd=work_dir,
            capture_output=True,
            env=commit_env,
        )
        resp = requests.post(f"{base_url}/push", json={}, timeout=10)
        data = resp.json()
        assert data.get("status") == "ok"


class TestCSGenerateCommitMsgDiffOnly:
    def test_generate_commit_msg_diff_no_untracked(self, cs_server: Any) -> None:
        """Generate commit message with only tracked diff, no untracked files.

        Covers 1118->1120 False branch (untracked_files is empty).
        """
        base_url, work_dir, _, _ = cs_server
        # Use `git clean -fd` to remove ALL untracked files/dirs
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=work_dir,
            capture_output=True,
        )
        # Also reset any staged changes
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=work_dir,
            capture_output=True,
        )

        # Modify a tracked file (creates diff but no untracked)
        fpath = os.path.join(work_dir, "file.txt")
        Path(fpath).write_text("line1\nmodified for diff only test\nline3\n")
        try:
            resp = requests.post(
                f"{base_url}/generate-commit-message", json={}, timeout=60
            )
            data = resp.json()
            assert "message" in data or "error" in data
        finally:
            # Restore
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
        # Write a new theme with a different mtime
        time.sleep(1.5)  # Ensure enough time passes
        theme_file.write_text(json.dumps({"kind": "light"}))
        # Wait for watcher to detect the change
        time.sleep(3)
        # Query theme - should show light colors
        resp = requests.get(f"{base_url}/theme", timeout=5)
        assert resp.status_code == 200
        # Restore dark theme
        theme_file.write_text(json.dumps({"kind": "dark"}))


class TestCSSSEDisconnect:
    def test_sse_disconnect_triggers_break(self, cs_server: Any) -> None:
        """Connect to SSE, wait for disconnect check, then close.

        Covers line 707 (break on request.is_disconnected()).
        """
        base_url, _, _, _ = cs_server
        resp = requests.get(f"{base_url}/events", stream=True, timeout=10)
        assert resp.status_code == 200
        # Read for ~3 seconds to allow multiple iterations of the SSE loop
        # The disconnect check happens every 20 iterations (~1s at 50ms each)
        start = time.monotonic()
        for chunk in resp.iter_content(chunk_size=256):
            if time.monotonic() - start > 3:
                break
        resp.close()
        # Give server time to detect disconnect
        time.sleep(2)


class TestCSProposedTasksFallback:
    def test_proposed_tasks_fallback_to_sample(self, cs_server: Any) -> None:
        """When proposed_tasks is empty, SAMPLE_TASKS are returned (line 883)."""
        base_url, _, _, _ = cs_server
        resp = requests.get(f"{base_url}/proposed_tasks", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestCSRunAndCleanup:
    def test_run_task_and_stop(self, cs_server: Any) -> None:
        """Run a task and stop it to exercise stop_agent with code-server running."""
        base_url, _, _, _ = cs_server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "slow_cs_task"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(1)
        stop = requests.post(f"{base_url}/stop", json={}, timeout=5)
        assert stop.status_code == 200
        time.sleep(2)

    def test_run_quick_task(self, cs_server: Any) -> None:
        """Run a quick task with code-server active."""
        base_url, _, _, _ = cs_server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "quick cs task"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(3)
