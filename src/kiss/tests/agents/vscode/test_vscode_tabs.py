# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Static ``media/main.js`` checks that stay behind from
``kiss.tests.agents.vscode.test_vscode_tabs`` (now
``kiss.tests.server.test_vscode_tabs``): they read the real
``media/main.js`` asset (bundled under ``src/kiss/agents/vscode/media``),
so they live in tests/agents/vscode while the server-only majority of
the original file moved to tests/server.
"""

import unittest


class TestClearChatDedup(unittest.TestCase):
    """When the secondary panel is closed and re-opened, clicking the KS
    button fires newConversation which sends clearChat.  The webview
    already has a fresh empty tab from initialization, so clearChat
    must NOT create a second one.

    The fix: the clearChat handler in main.js checks whether the active
    tab is already an empty new-chat tab (no backendChatId, welcome
    visible) and skips createNewTab() in that case.
    """

    js_src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        js_path = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "main.js"
        )
        cls.js_src = js_path.read_text()

    def _get_clear_chat_block(self) -> str:
        idx = self.js_src.index("case 'clearChat':")
        end = self.js_src.index("case 'showWelcome':", idx)
        return self.js_src[idx:end]

    def test_clear_chat_checks_backend_chat_id(self) -> None:
        """clearChat handler guards against creating a duplicate empty tab
        by checking that the active tab has no backendChatId."""
        block = self._get_clear_chat_block()
        assert "backendChatId" in block, (
            "clearChat handler must check backendChatId to avoid "
            "creating a duplicate empty tab when the panel is freshly opened"
        )

    def test_clear_chat_checks_welcome_visible(self) -> None:
        """clearChat handler checks that the welcome screen is still visible
        (i.e. the tab has no output content) before skipping tab creation."""
        block = self._get_clear_chat_block()
        assert "welcome" in block.lower(), (
            "clearChat handler must check welcome visibility to detect "
            "that the tab is still empty"
        )


if __name__ == "__main__":
    unittest.main()
