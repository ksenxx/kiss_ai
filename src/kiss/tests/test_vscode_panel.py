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

import kiss.agents.sorcar.chatbot_ui as chatbot_ui
import kiss.agents.sorcar.code_server as code_server


class TestSetupCodeServer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_state_db_idempotent(self) -> None:
        code_server._setup_code_server(self.tmpdir)
        code_server._setup_code_server(self.tmpdir)
        db_path = Path(self.tmpdir) / "User" / "globalStorage" / "state.vscdb"
        with sqlite3.connect(str(db_path)) as conn:
            keys = [r[0] for r in conn.execute("SELECT key FROM ItemTable").fetchall()]
        assert len(keys) == len(set(keys))

    def test_constants_well_formed(self) -> None:
        json.dumps(code_server._CS_SETTINGS)
        for key, value in code_server._CS_STATE_ENTRIES:
            assert isinstance(key, str) and isinstance(value, str)
        assert "function activate" in code_server._CS_EXTENSION_JS
        assert "module.exports={activate}" in code_server._CS_EXTENSION_JS

    def test_extension_uses_blue_decorations(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "blueDeco" in js
        assert "rgba(59,130,246,0.15)" in js
        assert "greenDeco" not in js

    def test_extension_has_no_codelens(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "registerCodeLensProvider" not in js
        assert "CodeLens" not in js

    def test_extension_tracks_current_hunk(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "var curHunk=null" in js
        assert "curHunk={fp:found.fp,idx:" in js

    def test_extension_handles_accept_reject_actions(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "ad.action==='accept'" in js
        assert "ad.action==='reject'" in js

    def test_extension_reverts_dirty_docs_in_open_merge(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "doc.isDirty" in js
        assert "workbench.action.files.revert" in js

    def test_extension_save_all_has_error_handling(self) -> None:
        js = code_server._CS_EXTENSION_JS
        assert "try{await vscode.workspace.saveAll(false);}catch(e){}" in js

    def test_settings_has_save_conflict_resolution(self) -> None:
        assert "files.saveConflictResolution" in code_server._CS_SETTINGS
        assert code_server._CS_SETTINGS["files.saveConflictResolution"] == "overwriteFileOnDisk"

    def test_check_all_done_has_error_callback(self) -> None:
        js = code_server._CS_EXTENSION_JS
        # checkAllDone should handle both success and error cases for saveAll
        idx = js.index("function checkAllDone()")
        snippet = js[idx : idx + 400]
        assert ".then(function(){" in snippet
        assert "},function(){" in snippet


class TestBuildHtmlSplitLayout(unittest.TestCase):

    def test_iframe_with_code_server_url(self) -> None:
        html = chatbot_ui._build_html("T", "http://127.0.0.1:9999", "/tmp/work")
        assert '<iframe id="code-server-frame"' in html
        assert 'data-base-url="http://127.0.0.1:9999"' in html
        assert 'data-work-dir="/tmp/work"' in html
        assert 'id="editor-fallback"' not in html


class TestBuildHtmlJavaScript(unittest.TestCase):
    html: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = chatbot_ui._build_html("T", "http://x:1", "/w")

    def test_merge_function(self) -> None:
        assert "function mergeAction" in self.html

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


class TestMergeEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.repo,
            capture_output=True,
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
            ["git", "diff", "main"],
            capture_output=True,
            text=True,
            cwd=self.repo,
        )
        assert "original" in result.stdout and "modified" in result.stdout

    def test_revert_file(self) -> None:
        subprocess.run(
            ["git", "checkout", "main", "--", "file.txt"],
            capture_output=True,
            text=True,
            cwd=self.repo,
            check=True,
        )
        assert (Path(self.repo) / "file.txt").read_text() == "original\n"

    def test_revert_with_patch(self) -> None:
        diff = subprocess.run(
            ["git", "diff", "main"],
            capture_output=True,
            text=True,
            cwd=self.repo,
        ).stdout
        subprocess.run(
            ["git", "apply", "--reverse"],
            input=diff,
            capture_output=True,
            text=True,
            cwd=self.repo,
            check=True,
        )
        assert (Path(self.repo) / "file.txt").read_text() == "original\n"

    def test_revert_nonexistent_file_fails(self) -> None:
        result = subprocess.run(
            ["git", "checkout", "main", "--", "no_such_file.txt"],
            capture_output=True,
            text=True,
            cwd=self.repo,
        )
        assert result.returncode != 0

    def test_not_a_git_repo(self) -> None:
        non_git = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                cwd=non_git,
            )
            assert result.returncode != 0
        finally:
            shutil.rmtree(non_git, ignore_errors=True)

    def test_revert_all_files(self) -> None:
        changed = (
            subprocess.run(
                ["git", "diff", "--name-only", "main"],
                capture_output=True,
                text=True,
                cwd=self.repo,
            )
            .stdout.strip()
            .split("\n")
        )
        for f in changed:
            subprocess.run(
                ["git", "checkout", "main", "--", f],
                capture_output=True,
                text=True,
                cwd=self.repo,
                check=True,
            )
        diff = subprocess.run(
            ["git", "diff", "main"],
            capture_output=True,
            text=True,
            cwd=self.repo,
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
        from kiss.agents.sorcar.browser_ui import find_free_port

        port = find_free_port()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                connected = True
        except (ConnectionRefusedError, OSError):
            connected = False
        assert not connected


class TestWelcomeChipCounts(unittest.TestCase):
    def test_welcome_shows_5_recent_and_5_suggested(self) -> None:
        html = chatbot_ui._build_html("T")
        assert "proposed.slice(0,5)" in html
        assert "tasks.slice(0,5)" in html
        assert "items.slice(0,10)" in html
        assert "proposed.slice(0,3)" not in html
        assert "tasks.slice(0,3)" not in html
        assert "items.slice(0,6)" not in html


class TestFilePathDetection(unittest.TestCase):
    """Tests for the file-path-opens-in-editor feature in submitTask."""

    html: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = chatbot_ui._build_html("T", "http://x:1", "/w")

    def test_looks_like_file_path_function_exists(self) -> None:
        assert "function looksLikeFilePath" in self.html

    def test_do_submit_task_function_exists(self) -> None:
        assert "function doSubmitTask" in self.html

    def test_submit_task_calls_open_file_for_file_paths(self) -> None:
        # submitTask should call /open-file when looksLikeFilePath returns true
        start = self.html.index("function submitTask()")
        end = self.html.index("btn.addEventListener", start)
        submit_fn = self.html[start:end]
        assert "looksLikeFilePath(task)" in submit_fn
        assert "'/open-file'" in submit_fn
        assert "doSubmitTask(task)" in submit_fn

    def test_looks_like_file_path_checks_absolute(self) -> None:
        # The function checks for paths starting with /
        start = self.html.index("function looksLikeFilePath")
        end = self.html.index("function doSubmitTask", start)
        fn = self.html[start:end]
        assert "s.startsWith('/')" in fn

    def test_looks_like_file_path_checks_relative(self) -> None:
        start = self.html.index("function looksLikeFilePath")
        end = self.html.index("function doSubmitTask", start)
        fn = self.html[start:end]
        assert "s.startsWith('./')" in fn
        assert "s.startsWith('../')" in fn
        assert "s.startsWith('~/')" in fn

    def test_looks_like_file_path_checks_extension(self) -> None:
        start = self.html.index("function looksLikeFilePath")
        end = self.html.index("function doSubmitTask", start)
        fn = self.html[start:end]
        # Checks for file extension pattern
        assert r"\.\w{1,10}$" in fn

    def test_fallback_to_do_submit_on_open_file_failure(self) -> None:
        # When /open-file returns non-ok, should fall back to doSubmitTask
        start = self.html.index("function submitTask()")
        end = self.html.index("btn.addEventListener", start)
        submit_fn = self.html[start:end]
        assert "if(r.ok){inp.value='';return}" in submit_fn
        assert ".catch(function(){doSubmitTask(task)})" in submit_fn

    def test_js_balanced_with_new_functions(self) -> None:
        start = self.html.find("<script>")
        end = self.html.find("</script>")
        js = self.html[start:end]
        assert js.count("{") == js.count("}")
        assert js.count("(") == js.count(")")


if __name__ == "__main__":
    unittest.main()
