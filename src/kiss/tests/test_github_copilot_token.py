"""Tests for GitHub Copilot authentication token persistence.

Covers:
- _load_github_token: all branches (valid, missing, corrupt, empty, no key)
- _CS_EXTENSION_JS: token-saving JS syntax and content
- _setup_code_server: extension.js includes token-saving code
- Path consistency between JS (ghTokenFile) and Python (_load_github_token)
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _CS_EXTENSION_JS,
    _GH_TOKEN_FILENAME,
    _disable_copilot_scm_button,
    _load_github_token,
    _setup_code_server,
)


class TestLoadGithubToken:
    """Test _load_github_token with real files."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        # cs_data_dir is like ~/.kiss/cs-abc12345
        self.cs_data_dir = os.path.join(self.tmpdir, "kiss", "cs-test1234")
        os.makedirs(self.cs_data_dir, exist_ok=True)
        # Token file lives in parent: ~/.kiss/github-copilot-token.json
        self.token_file = Path(self.cs_data_dir).parent / _GH_TOKEN_FILENAME

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_token(self) -> None:
        """Valid token file returns the accessToken."""
        self.token_file.write_text(json.dumps({
            "accessToken": "gho_abc123xyz",
            "account": {"label": "testuser", "id": "12345"},
            "id": "session-id",
        }))
        assert _load_github_token(self.cs_data_dir) == "gho_abc123xyz"

    def test_missing_file(self) -> None:
        """No token file returns None."""
        assert _load_github_token(self.cs_data_dir) is None

    def test_corrupt_json(self) -> None:
        """Corrupt JSON returns None without raising."""
        self.token_file.write_text("not valid json {{{{")
        assert _load_github_token(self.cs_data_dir) is None

    def test_empty_file(self) -> None:
        """Empty file returns None (JSONDecodeError)."""
        self.token_file.write_text("")
        assert _load_github_token(self.cs_data_dir) is None

    def test_missing_access_token_key(self) -> None:
        """JSON without accessToken key returns None."""
        self.token_file.write_text(json.dumps({"account": "user"}))
        assert _load_github_token(self.cs_data_dir) is None

    def test_empty_access_token(self) -> None:
        """Empty string accessToken returns None."""
        self.token_file.write_text(json.dumps({"accessToken": ""}))
        assert _load_github_token(self.cs_data_dir) is None

    def test_null_access_token(self) -> None:
        """null accessToken returns None."""
        self.token_file.write_text(json.dumps({"accessToken": None}))
        assert _load_github_token(self.cs_data_dir) is None

    def test_nonexistent_parent_dir(self) -> None:
        """cs_data_dir whose parent doesn't exist returns None."""
        assert _load_github_token("/nonexistent/path/cs-test") is None

    def test_unreadable_file(self) -> None:
        """Unreadable token file returns None (OSError)."""
        self.token_file.write_text(json.dumps({"accessToken": "gho_secret"}))
        os.chmod(str(self.token_file), 0o000)
        try:
            assert _load_github_token(self.cs_data_dir) is None
        finally:
            os.chmod(str(self.token_file), 0o644)


class TestExtensionJSTokenCode:
    """Verify _CS_EXTENSION_JS contains correct GitHub token persistence code."""

    def test_js_syntax_valid(self) -> None:
        """Extension JS must be syntactically valid Node.js."""
        result = subprocess.run(
            ["node", "--check", "--input-type=commonjs"],
            input=_CS_EXTENSION_JS,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"JS syntax error: {result.stderr}"

    def test_contains_token_file_path(self) -> None:
        """JS must write to github-copilot-token.json in dataDir parent."""
        assert "github-copilot-token.json" in _CS_EXTENSION_JS

    def test_token_file_in_parent_dir(self) -> None:
        """ghTokenFile path goes up one dir from dataDir (to ~/.kiss/)."""
        assert "path.join(dataDir,'..','github-copilot-token.json')" in _CS_EXTENSION_JS

    def test_calls_authentication_get_session(self) -> None:
        """JS uses vscode.authentication.getSession for GitHub provider."""
        assert "vscode.authentication.getSession(" in _CS_EXTENSION_JS
        assert "'github'" in _CS_EXTENSION_JS

    def test_requests_correct_scopes(self) -> None:
        """JS requests user:email and repo scopes."""
        assert "'user:email'" in _CS_EXTENSION_JS
        assert "'repo'" in _CS_EXTENSION_JS

    def test_uses_create_if_none_false(self) -> None:
        """JS uses createIfNone:false (don't prompt user)."""
        assert "createIfNone:false" in _CS_EXTENSION_JS

    def test_writes_with_mode_0600(self) -> None:
        """Token file must be written with mode 0o600 for security."""
        assert "mode:0o600" in _CS_EXTENSION_JS

    def test_saves_on_auth_change(self) -> None:
        """JS listens for auth session changes."""
        assert "onDidChangeSessions" in _CS_EXTENSION_JS

    def test_saves_periodically(self) -> None:
        """JS saves token on an interval."""
        assert "ghInterval" in _CS_EXTENSION_JS
        assert "setInterval(saveGitHubToken" in _CS_EXTENSION_JS

    def test_saves_on_startup(self) -> None:
        """saveGitHubToken is called immediately on activate."""
        # After function definition, it's called directly
        js = _CS_EXTENSION_JS
        # Find saveGitHubToken() standalone call (not the definition)
        def_end = js.index("async function saveGitHubToken(){")
        # After the function body, find the direct call
        remaining = js[def_end:]
        # The call saveGitHubToken(); happens after the function definition
        assert "saveGitHubToken();" in remaining

    def test_token_saving_inside_activate(self) -> None:
        """Token-saving code must be inside activate(), not outside."""
        # The JS ends with: } (closing activate) then module.exports={activate};
        stripped = _CS_EXTENSION_JS.strip()
        assert stripped.endswith("module.exports={activate};")
        # saveGitHubToken must appear BEFORE the closing } of activate
        idx_save = stripped.index("saveGitHubToken")
        idx_exports = stripped.index("module.exports")
        assert idx_save < idx_exports


class TestSetupCodeServerWritesTokenCode:
    """Verify _setup_code_server writes extension.js with token-saving code."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extension_js_contains_token_code(self) -> None:
        """_setup_code_server writes extension.js that includes token persistence."""
        shared_ext = os.path.join(self.tmpdir, "shared-extensions")
        _setup_code_server(self.tmpdir, shared_ext)
        ext_js = Path(shared_ext) / "kiss-init" / "extension.js"
        assert ext_js.exists()
        content = ext_js.read_text()
        assert "github-copilot-token.json" in content
        assert "saveGitHubToken" in content
        assert "onDidChangeSessions" in content

    def test_shared_extensions_dir(self) -> None:
        """_setup_code_server writes kiss-init to shared extensions_dir."""
        shared_ext = os.path.join(self.tmpdir, "shared-extensions")
        _setup_code_server(self.tmpdir, shared_ext)
        ext_js = Path(shared_ext) / "kiss-init" / "extension.js"
        assert ext_js.exists()
        # Should NOT be in the data dir's extensions
        assert not (Path(self.tmpdir) / "extensions" / "kiss-init").exists()


class TestDisableCopilotScmButton:
    """Verify _disable_copilot_scm_button uses shared extensions_dir."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uses_shared_extensions_dir(self) -> None:
        """_disable_copilot_scm_button looks in the shared extensions directory."""
        shared = os.path.join(self.tmpdir, "shared")
        ext_dir = os.path.join(shared, "github.copilot-chat-1.0.0")
        os.makedirs(ext_dir)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {"command": "github.copilot.git.generateCommitMessage", "when": "true"}
                    ]
                }
            }
        }
        Path(ext_dir, "package.json").write_text(json.dumps(pkg))
        _disable_copilot_scm_button(shared)
        result = json.loads(Path(ext_dir, "package.json").read_text())
        item = result["contributes"]["menus"]["scm/inputBox"][0]
        assert item["when"] == "false"


class TestTokenPathConsistency:
    """Verify JS and Python agree on the token file location."""

    def test_filename_matches(self) -> None:
        """_GH_TOKEN_FILENAME matches the filename used in JS."""
        assert _GH_TOKEN_FILENAME in _CS_EXTENSION_JS

    def test_relative_path_from_data_dir(self) -> None:
        """Both JS and Python resolve token file relative to cs_data_dir parent.

        JS:  path.join(dataDir, '..', 'github-copilot-token.json')
        Python: Path(cs_data_dir).parent / _GH_TOKEN_FILENAME

        For cs_data_dir = ~/.kiss/cs-abc12345:
          JS:     ~/.kiss/cs-abc12345/../github-copilot-token.json
          Python: ~/.kiss/github-copilot-token.json
        """
        cs_data_dir = "/home/user/.kiss/cs-abc12345"
        python_path = Path(cs_data_dir).parent / _GH_TOKEN_FILENAME
        # JS equivalent: path.resolve(dataDir, '..', filename)
        js_path = Path(cs_data_dir) / ".." / _GH_TOKEN_FILENAME
        assert python_path.resolve() == js_path.resolve()
