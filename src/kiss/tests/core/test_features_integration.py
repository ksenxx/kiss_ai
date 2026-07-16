# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for: budget limit, custom endpoint+key, web browser toggle,
and API key setup/deletion in the configuration panel.

Each test uses real HTTP servers, real file I/O, and real objects —
no mocks, patches, fakes, or test doubles (except monkeypatch for env
isolation, which is not a test double).
"""

from __future__ import annotations

import json
import os
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.server.vscode_config import (
    API_KEY_ENV_VARS,
    load_config,
    save_api_key_to_shell,
    save_config,
)


@pytest.fixture(autouse=True)
def _restore_default_config():
    """Restore the DEFAULT_CONFIG binding after every test.

    ``save_config``/``save_api_key_to_shell`` trigger
    ``vscode_config._refresh_config`` which rebinds
    ``kiss.core.config.DEFAULT_CONFIG`` from the (test-redirected)
    config file; without restoration the rebound object leaks into
    later test modules (e.g. ``test_build_config_cli``) and breaks
    their default-value assertions.
    """
    saved = config_module.DEFAULT_CONFIG
    snapshot = saved.model_copy(deep=True).__dict__
    yield
    saved.__dict__.update(snapshot)
    config_module.DEFAULT_CONFIG = saved


def _finish_response(model: str = "gpt-4o-mini") -> dict:
    """OpenAI chat-completion that calls ``finish`` with result='done'."""
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fin",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": '{"result": "done"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call_response_expensive() -> dict:
    """Non-finish tool call with huge token usage (blows the budget)."""
    return {
        "id": "chatcmpl-exp",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 500_000,
            "completion_tokens": 500_000,
            "total_tokens": 1_000_000,
        },
    }


class _FinishHandler(BaseHTTPRequestHandler):
    """Always returns a ``finish`` tool call."""

    received_headers: dict[str, str] = {}
    request_count: int = 0

    def do_POST(self) -> None:  # noqa: N802
        type(self).received_headers = dict(self.headers)
        type(self).request_count += 1
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        body = json.dumps(_finish_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _ExpensiveHandler(BaseHTTPRequestHandler):
    """Returns non-finish tool calls with massive token usage."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        body = json.dumps(_tool_call_response_expensive()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server(
    handler: type[BaseHTTPRequestHandler],
) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


class TestBudgetLimitRealHTTP:
    """Budget exceeded through real HTTP agent loop."""

    def test_tiny_budget_raises(self) -> None:
        """Agent stops with KISSError when cost exceeds max_budget."""
        srv, url = _start_server(_ExpensiveHandler)
        try:
            agent = KISSAgent("budget-feat")

            def noop() -> str:
                """No-op."""
                return "ok"

            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop.",
                    tools=[noop],
                    is_agentic=True,
                    max_steps=50,
                    max_budget=0.01,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "k"},
                )
            assert agent.budget_used > 0.01
        finally:
            srv.shutdown()

    def test_large_budget_allows_finish(self) -> None:
        """Agent finishes normally when budget is sufficient."""
        srv, url = _start_server(_FinishHandler)
        try:
            agent = KISSAgent("budget-ok")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish.",
                is_agentic=True,
                max_steps=10,
                max_budget=10.0,
                verbose=False,
                model_config={"base_url": url, "api_key": "k"},
            )
            assert result == "done"
        finally:
            srv.shutdown()


class TestCustomEndpointRealHTTP:
    """Agent uses the custom base_url and api_key from model_config."""

    def test_custom_endpoint_receives_request(self) -> None:
        """Real HTTP request goes to the custom endpoint URL."""
        _FinishHandler.request_count = 0
        _FinishHandler.received_headers = {}
        srv, url = _start_server(_FinishHandler)
        try:
            agent = KISSAgent("custom-ep")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish.",
                is_agentic=True,
                max_steps=5,
                max_budget=10.0,
                verbose=False,
                model_config={
                    "base_url": url,
                    "api_key": "sk-custom-secret-key",
                },
            )
            assert result == "done"
            assert _FinishHandler.request_count >= 1
            auth = _FinishHandler.received_headers.get("Authorization", "")
            assert "sk-custom-secret-key" in auth
        finally:
            srv.shutdown()

class TestWebBrowserToggle:
    """web_tools parameter controls browser tool availability."""

    def test_web_tools_false_no_browser_tools(self) -> None:
        """When web_tools=False, SorcarAgent._setup_tools skips web tools."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("no-web")
        agent._use_web_tools = False
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        browser_names = {
            "go_to_url", "click", "type_text", "press_key",
            "scroll", "screenshot", "get_page_content", "close_browser",
        }
        assert not browser_names.intersection(tool_names)
        assert agent.web_use_tool is None

    def test_web_tools_true_has_browser_tools(self) -> None:
        """When web_tools=True, _setup_tools includes web tools."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("with-web")
        agent._use_web_tools = True
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "go_to_url" in tool_names
        assert agent.web_use_tool is not None
        agent.web_use_tool.close()
        agent.web_use_tool = None

    def test_config_use_web_browser_false_saved_and_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """use_web_browser=False persists through save/load cycle."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "kiss.server.vscode_config.CONFIG_DIR", fake_home / ".kiss",
        )
        monkeypatch.setattr(
            "kiss.server.vscode_config.CONFIG_PATH",
            fake_home / ".kiss" / "config.json",
        )
        save_config({"use_web_browser": False})
        cfg = load_config()
        assert cfg["use_web_browser"] is False


class TestApiKeySetupAndDeletion:
    """Full lifecycle: save key → verify → delete key → verify gone."""

    @pytest.fixture(autouse=True)
    def _isolate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "kiss.server.vscode_config.CONFIG_DIR", fake_home / ".kiss",
        )
        monkeypatch.setattr(
            "kiss.server.vscode_config.CONFIG_PATH",
            fake_home / ".kiss" / "config.json",
        )
        monkeypatch.setenv("SHELL", "/bin/zsh")
        for k in API_KEY_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        from kiss.core import config as config_module

        monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.DEFAULT_CONFIG)

    def test_save_api_key_sets_env_and_rc(self) -> None:
        """Saving a key writes to RC file AND sets os.environ."""
        save_api_key_to_shell("GEMINI_API_KEY", "gem-test-123")
        assert os.environ["GEMINI_API_KEY"] == "gem-test-123"
        rc = Path.home() / ".zshrc"
        # H3 fix uses shlex.quote which omits quotes for shell-safe values.
        assert (
            f"export GEMINI_API_KEY={shlex.quote('gem-test-123')}"
            in rc.read_text()
        )

    def test_overwrite_key_replaces_in_rc(self) -> None:
        """Saving a new value for an existing key replaces the old one."""
        save_api_key_to_shell("OPENAI_API_KEY", "old-val")
        save_api_key_to_shell("OPENAI_API_KEY", "new-val")
        rc = Path.home() / ".zshrc"
        content = rc.read_text()
        assert "old-val" not in content
        assert f"export OPENAI_API_KEY={shlex.quote('new-val')}" in content
        assert os.environ["OPENAI_API_KEY"] == "new-val"

    def test_delete_key_by_saving_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving an empty key via config panel skips writing to RC.

        The VSCodeServer saveConfig handler skips empty keys, so
        after the key is removed from the env and not written to RC,
        it is effectively deleted.
        """
        from kiss.server.server import VSCodeServer

        server = VSCodeServer()

        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": "ant-key-to-delete"},
        })
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key-to-delete"
        rc = Path.home() / ".zshrc"
        assert "ant-key-to-delete" in rc.read_text()

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": ""},
        })
        assert os.environ.get("ANTHROPIC_API_KEY") is None

    def test_multiple_keys_independent(self) -> None:
        """Saving/deleting one key doesn't affect others."""
        save_api_key_to_shell("GEMINI_API_KEY", "gem-val")
        save_api_key_to_shell("OPENAI_API_KEY", "oai-val")
        rc = Path.home() / ".zshrc"
        content = rc.read_text()
        assert "gem-val" in content
        assert "oai-val" in content
        save_api_key_to_shell("GEMINI_API_KEY", "gem-new")
        content = rc.read_text()
        assert "gem-new" in content
        assert "oai-val" in content
