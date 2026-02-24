"""Tests for the VS Code panel in the assistant.

Tests cover: _setup_code_server, _build_html split layout,
merge endpoints, file link data-path, and code-server lifecycle.
No mocks — uses real files, real git repos, and real sockets.
"""

import json
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.agents.assistant.assistant as assistant


class TestSetupCodeServer(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_settings_keys_present(self) -> None:
        assistant._setup_code_server(self.tmpdir)
        settings = json.loads(
            (Path(self.tmpdir) / "User" / "settings.json").read_text()
        )
        for key in assistant._CS_SETTINGS:
            assert key in settings, f"Missing setting: {key}"
            assert settings[key] == assistant._CS_SETTINGS[key]

    def test_settings_preserve_existing_keys(self) -> None:
        user_dir = Path(self.tmpdir) / "User"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text(
            json.dumps({"editor.fontSize": 16, "workbench.startupEditor": "welcomePage"})
        )
        assistant._setup_code_server(self.tmpdir)
        settings = json.loads((user_dir / "settings.json").read_text())
        assert settings["editor.fontSize"] == 16
        assert settings["workbench.startupEditor"] == "none"

    def test_settings_handles_corrupted_json(self) -> None:
        user_dir = Path(self.tmpdir) / "User"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text("not valid json{{{")
        assistant._setup_code_server(self.tmpdir)
        settings = json.loads((user_dir / "settings.json").read_text())
        assert settings["workbench.startupEditor"] == "none"

    def test_state_db_has_all_entries(self) -> None:
        assistant._setup_code_server(self.tmpdir)
        db_path = Path(self.tmpdir) / "User" / "globalStorage" / "state.vscdb"
        with sqlite3.connect(str(db_path)) as conn:
            rows = dict(conn.execute("SELECT key, value FROM ItemTable").fetchall())
        for key, value in assistant._CS_STATE_ENTRIES:
            assert key in rows, f"Missing state entry: {key}"
            assert rows[key] == value

    def test_state_db_idempotent(self) -> None:
        assistant._setup_code_server(self.tmpdir)
        assistant._setup_code_server(self.tmpdir)
        db_path = Path(self.tmpdir) / "User" / "globalStorage" / "state.vscdb"
        with sqlite3.connect(str(db_path)) as conn:
            keys = [r[0] for r in conn.execute("SELECT key FROM ItemTable").fetchall()]
        assert len(keys) == len(set(keys))

    def test_extension_files(self) -> None:
        assistant._setup_code_server(self.tmpdir)
        ext_dir = Path(self.tmpdir) / "extensions" / "kiss-init"
        pkg = json.loads((ext_dir / "package.json").read_text())
        assert pkg["name"] == "kiss-init"
        assert "onStartupFinished" in pkg["activationEvents"]
        assert (ext_dir / "extension.js").read_text() == assistant._CS_EXTENSION_JS

    def test_cleans_chat_sessions_preserves_other_files(self) -> None:
        ws = Path(self.tmpdir) / "User" / "workspaceStorage" / "abc123"
        ws.mkdir(parents=True)
        (ws / "meta.json").write_text('{"id":"abc123"}')
        for sub in ("chatSessions", "chatEditingSessions"):
            d = ws / sub
            d.mkdir()
            (d / "session.json").write_text("{}")
        assistant._setup_code_server(self.tmpdir)
        assert (ws / "meta.json").exists()
        assert not (ws / "chatSessions").exists()
        assert not (ws / "chatEditingSessions").exists()

    def test_constants_well_formed(self) -> None:
        json.dumps(assistant._CS_SETTINGS)
        for key, value in assistant._CS_STATE_ENTRIES:
            assert isinstance(key, str) and isinstance(value, str)
        assert "function activate" in assistant._CS_EXTENSION_JS
        assert "module.exports={activate}" in assistant._CS_EXTENSION_JS


class TestBuildHtmlSplitLayout(unittest.TestCase):

    def test_split_layout_structure(self) -> None:
        html = assistant._build_html("T", "S")
        for elem in ("split-container", "editor-panel", "divider", "assistant-panel"):
            assert f'id="{elem}"' in html, f"Missing #{elem}"
        assert "width:80%" in html

    def test_merge_overlay_and_header_buttons(self) -> None:
        html = assistant._build_html("T", "S")
        assert 'id="merge-overlay"' in html
        assert 'id="merge-btn"' in html
        assert 'title="Task history"' in html
        assert 'title="Suggested tasks"' in html

    def test_editor_fallback_without_code_server(self) -> None:
        html = assistant._build_html("T", "S")
        assert 'id="editor-fallback"' in html
        assert "<iframe" not in html

    def test_iframe_with_code_server_url(self) -> None:
        html = assistant._build_html("T", "S", "http://127.0.0.1:9999", "/tmp/work")
        assert '<iframe id="code-server-frame"' in html
        assert 'data-base-url="http://127.0.0.1:9999"' in html
        assert 'data-work-dir="/tmp/work"' in html
        assert 'id="editor-fallback"' not in html

    def test_iframe_folder_url_encoded(self) -> None:
        html = assistant._build_html("T", "S", "http://x:1", "/path with spaces")
        assert "path%20with%20spaces" in html


class TestBuildHtmlJavaScript(unittest.TestCase):
    html: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = assistant._build_html("T", "S", "http://x:1", "/w")

    def test_merge_functions(self) -> None:
        for fn in (
            "openMerge", "closeMerge", "parseDiff", "renderMerge",
            "mergeAcceptHunk", "mergeAcceptFile", "mergeAcceptAll",
            "mergeRevertHunk", "mergeRevertFile", "mergeUndoAll",
        ):
            assert f"function {fn}" in self.html, f"Missing: {fn}"

    def test_divider_and_editor_functions(self) -> None:
        assert "isDragging" in self.html
        assert "function openInEditor" in self.html
        assert "closest('[data-path]')" in self.html

    def test_js_balanced(self) -> None:
        start = self.html.find("<script>")
        end = self.html.find("</script>")
        js = self.html[start:end]
        assert js.count("{") == js.count("}")
        assert js.count("(") == js.count(")")


class TestBuildHtmlCSS(unittest.TestCase):

    def test_split_and_merge_css(self) -> None:
        html = assistant._build_html("T", "S")
        for pattern in (
            "#split-container{display:flex",
            "#editor-panel{position:relative",
            "cursor:col-resize",
            "#assistant-panel{",
            "#merge-overlay{",
            ".merge-line.added{",
            ".merge-line.removed{",
            ".tp[data-path]{cursor:pointer",
            "#assistant-panel .logo span{display:none}",
            "#assistant-panel #suggestions{grid-template-columns:1fr",
        ):
            assert pattern in html, f"Missing CSS: {pattern}"


class TestMergeEndpoints(unittest.TestCase):

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.repo, capture_output=True,
        )
        (Path(self.repo) / "file.txt").write_text("original\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.repo, capture_output=True)
        subprocess.run(["git", "branch", "-M", "main"], cwd=self.repo, capture_output=True)
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=self.repo, capture_output=True)
        (Path(self.repo) / "file.txt").write_text("modified\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change"], cwd=self.repo, capture_output=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_diff_contains_changes(self) -> None:
        result = subprocess.run(
            ["git", "diff", "main"], capture_output=True, text=True, cwd=self.repo,
        )
        assert "original" in result.stdout and "modified" in result.stdout

    def test_revert_file(self) -> None:
        subprocess.run(
            ["git", "checkout", "main", "--", "file.txt"],
            capture_output=True, text=True, cwd=self.repo, check=True,
        )
        assert (Path(self.repo) / "file.txt").read_text() == "original\n"

    def test_revert_with_patch(self) -> None:
        diff = subprocess.run(
            ["git", "diff", "main"], capture_output=True, text=True, cwd=self.repo,
        ).stdout
        subprocess.run(
            ["git", "apply", "--reverse"], input=diff,
            capture_output=True, text=True, cwd=self.repo, check=True,
        )
        assert (Path(self.repo) / "file.txt").read_text() == "original\n"

    def test_revert_nonexistent_file_fails(self) -> None:
        result = subprocess.run(
            ["git", "checkout", "main", "--", "no_such_file.txt"],
            capture_output=True, text=True, cwd=self.repo,
        )
        assert result.returncode != 0

    def test_not_a_git_repo(self) -> None:
        non_git = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, cwd=non_git,
            )
            assert result.returncode != 0
        finally:
            shutil.rmtree(non_git, ignore_errors=True)

    def test_revert_all_files(self) -> None:
        changed = subprocess.run(
            ["git", "diff", "--name-only", "main"],
            capture_output=True, text=True, cwd=self.repo,
        ).stdout.strip().split("\n")
        for f in changed:
            subprocess.run(
                ["git", "checkout", "main", "--", f],
                capture_output=True, text=True, cwd=self.repo, check=True,
            )
        diff = subprocess.run(
            ["git", "diff", "main"], capture_output=True, text=True, cwd=self.repo,
        ).stdout
        assert diff.strip() == ""


class TestFixedPortLogic(unittest.TestCase):

    def test_detects_open_port(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                connected = True
        except (ConnectionRefusedError, OSError):
            connected = False
        finally:
            s.close()
        assert connected

    def test_detects_closed_port(self) -> None:
        from kiss.core.browser_ui import find_free_port
        port = find_free_port()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                connected = True
        except (ConnectionRefusedError, OSError):
            connected = False
        assert not connected


if __name__ == "__main__":
    unittest.main()
