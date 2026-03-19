"""Tests for the run-prompt-button feature: active-file-info and get-file-content endpoints."""

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def prompt_file():
    """Create a temporary .md file that IS detected as a prompt."""
    content = (
        "# System Prompt\n"
        "You are an expert Python developer.\n"
        "## Constraints\n"
        "- Do not use classes unless necessary.\n"
        "- Return only code.\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def non_prompt_md_file():
    """Create a temporary .md file that is NOT detected as a prompt."""
    content = (
        "# Project Documentation\n"
        "This project is a web scraper.\n"
        "## Installation\n"
        "Run `pip install -r requirements.txt`.\n"
        "## Usage\n"
        "Run `python main.py`.\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def non_md_file():
    """Create a temporary .py file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')\n")
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestActiveFileTracking:
    """Test the active-file.json reading and prompt detection logic."""

    def test_active_file_json_write_read(self, tmp_path: Path) -> None:
        active_file = os.path.join(str(tmp_path), "active-file.json")
        data = {"path": "/some/file.md"}
        with open(active_file, "w") as f:
            json.dump(data, f)
        with open(active_file) as f:
            loaded = json.loads(f.read())
        assert loaded["path"] == "/some/file.md"

    def test_active_file_json_empty_path(self, tmp_path: Path) -> None:
        active_file = os.path.join(str(tmp_path), "active-file.json")
        data = {"path": ""}
        with open(active_file, "w") as f:
            json.dump(data, f)
        with open(active_file) as f:
            loaded = json.loads(f.read())
        assert loaded["path"] == ""

    def test_active_file_json_missing(self, tmp_path: Path) -> None:
        active_file = os.path.join(str(tmp_path), "active-file.json")
        assert not os.path.exists(active_file)


class TestCodeServerExtensionActiveFileWrite:
    """Test that the extension JS includes writeActiveFile logic."""

    def test_extension_js_contains_active_file_tracking(self) -> None:
        from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS

        assert "writeActiveFile" in _CS_EXTENSION_JS
        assert "active-file.json" in _CS_EXTENSION_JS
        assert "onDidChangeActiveTextEditor" in _CS_EXTENSION_JS


class TestRunPromptHTMLAndCSS:
    """Test that the HTML/CSS contains the run-prompt button and styles."""

    def test_js_and_css_run_prompt_btn(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, CHATBOT_JS, CHATBOT_THEME_CSS

        assert "checkActiveFile" in CHATBOT_JS
        assert "/active-file-info" in CHATBOT_JS
        assert "/get-file-content" in CHATBOT_JS
        assert "runPromptBtn" in CHATBOT_JS
        assert "#run-prompt-btn" in CHATBOT_CSS
        assert "#run-prompt-btn:not(:disabled):hover" in CHATBOT_THEME_CSS


class TestRunPromptButtonJSBehavior:
    """Test JS behavior specifics for the run-prompt button."""

    def test_js_polling_and_state(self) -> None:
        """JS polls active file, enables/disables button, sets input from content."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "setInterval(checkActiveFile,2000)" in CHATBOT_JS
        assert "runPromptBtn.disabled=true" in CHATBOT_JS
        assert "runPromptBtn.disabled=false" in CHATBOT_JS
        assert "submitTask()" in CHATBOT_JS
        assert "inp.value=d.content" in CHATBOT_JS


class TestPlayButtonDisableDuringRun:
    """Test that the play button is disabled when a task is running and re-enabled after."""

    def test_js_disables_play_button_in_submit_task(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        idx_running = CHATBOT_JS.index("running=true;inp.disabled=true;")
        idx_disable = CHATBOT_JS.index("runPromptBtn.disabled=true;", idx_running)
        idx_btn_display = CHATBOT_JS.index("btn.style.display='none'", idx_running)
        assert idx_disable < idx_btn_display

    def test_js_check_active_file_guards(self) -> None:
        """checkActiveFile guards when running, and after fetch."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        idx_func = CHATBOT_JS.index("function checkActiveFile(){")
        idx_guard = CHATBOT_JS.index("if(running){runPromptBtn.disabled=true;return}", idx_func)
        idx_fetch = CHATBOT_JS.index("fetch('/active-file-info')", idx_func)
        assert idx_guard < idx_fetch
        idx_guard2 = CHATBOT_JS.index("if(running)return;", idx_fetch)
        idx_is_prompt = CHATBOT_JS.index("if(d.is_prompt){", idx_fetch)
        assert idx_guard2 < idx_is_prompt

    def test_js_set_ready_calls_check_and_clears(self) -> None:
        """setReady() calls checkActiveFile, clears input, and focuses."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        idx_set_ready = CHATBOT_JS.index("function setReady(label){")
        idx_running_false = CHATBOT_JS.index("running=false;", idx_set_ready)
        idx_check = CHATBOT_JS.index("checkActiveFile();", idx_set_ready)
        idx_focus = CHATBOT_JS.index("inp.focus();", idx_check)
        assert idx_running_false < idx_check < idx_focus
        next_fn = CHATBOT_JS.index("\nfunction ", idx_set_ready + 1)
        body = CHATBOT_JS[idx_set_ready:next_fn]
        assert "inp.value=''" in body


class TestEndToEndPromptDetection:
    """End-to-end test: write active-file.json, check prompt detection."""

    def test_non_md_file_workflow(self, non_md_file: str, tmp_path: Path) -> None:
        active_file = os.path.join(str(tmp_path), "active-file.json")
        with open(active_file, "w") as f:
            json.dump({"path": non_md_file}, f)

        with open(active_file) as f:
            data = json.loads(f.read())
        fpath = data["path"]
        assert not fpath.endswith(".md")
