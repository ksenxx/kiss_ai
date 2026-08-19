# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent


class TestCloseDbNotInitialized:
    """Cover _close_db early return when _db_conn is None (line 52)."""

    def test_close_db_when_not_initialized(self) -> None:
        """Closing DB when connection was never opened should be a no-op."""
        original_conn = th._db_conn
        original_path = th._DB_PATH
        try:
            th._db_conn = None
            th._close_db()
            assert th._db_conn is None
        finally:
            th._db_conn = original_conn
            th._DB_PATH = original_path


class TestResumeChatNoMatch:
    """Cover resume_chat branches."""

    def test_resume_chat_by_id_empty(self) -> None:
        """resume_chat_by_id("") should be a no-op."""
        agent = ChatSorcarAgent("test")
        original_chat_id = agent.chat_id
        agent.resume_chat_by_id("")
        assert agent.chat_id == original_chat_id


class TestValidTabSwitch:
    """Cover successful tab switch in go_to_url (lines 235-236)."""

    @pytest.fixture()
    def http_server(self, tmp_path: Path) -> Generator[str]:
        """Start a minimal HTTP server for testing."""
        import http.server
        import socketserver

        html = "<html><body><h1>Tab Switch Test</h1></body></html>"
        (tmp_path / "index.html").write_text(html)

        handler = http.server.SimpleHTTPRequestHandler
        srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        old_dir = os.getcwd()
        os.chdir(str(tmp_path))
        yield f"http://127.0.0.1:{port}/index.html"
        os.chdir(old_dir)
        srv.shutdown()

    def test_valid_tab_switch(self, http_server: str, tmp_path: Path) -> None:
        """Switching to tab 0 should succeed (lines 235-236)."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        profile = str(tmp_path / "browser_profile")
        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            tool.go_to_url(http_server)
            result = tool.go_to_url("tab:0")
            assert "Error" not in result
            assert "Tab Switch Test" in result or "Page:" in result
        finally:
            tool.close()
