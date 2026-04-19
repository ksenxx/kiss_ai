"""Integration tests for the Cmd+D toggle-focus feature.

Validates that the extension correctly toggles focus between the editor
and the chat input, using actual source inspection of the TypeScript
and JavaScript files — no mocks, patches, or test doubles.

Bug: ``toggleFocus`` checked ``sidebarView.visible`` which is ``true``
whenever the sidebar is on screen, even when the editor has focus.
This meant Cmd+D always focused the editor and never toggled back
to the chat input.

Fix: Track webview focus state via ``webviewFocusChanged`` messages
and use ``sidebarView.hasFocus`` instead of ``sidebarView.visible``.
"""

from pathlib import Path

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


class TestToggleFocusExtensionTs:
    """Verify extension.ts uses hasFocus (not visible) for toggle."""

    src: str

    @classmethod
    def setup_class(cls) -> None:
        cls.src = (_VSCODE_DIR / "src" / "extension.ts").read_text()

    def test_toggle_uses_has_focus(self) -> None:
        """toggleFocus command checks hasFocus, not visible."""
        assert "sidebarView!.hasFocus" in self.src

    def test_toggle_does_not_use_visible(self) -> None:
        """toggleFocus no longer checks .visible for the toggle decision."""
        # Extract the toggleFocus handler body
        start = self.src.index("kissSorcar.toggleFocus")
        # Find the next two occurrences of "finally" to delimit the handler
        end = self.src.index("_focusToggling = false", start)
        handler = self.src[start:end]
        assert "sidebarView!.visible" not in handler


class TestToggleFocusSidebarView:
    """Verify SorcarSidebarView tracks webview focus state."""

    src: str

    @classmethod
    def setup_class(cls) -> None:
        cls.src = (_VSCODE_DIR / "src" / "SorcarSidebarView.ts").read_text()

    def test_has_focus_getter_exists(self) -> None:
        """SorcarSidebarView exposes a hasFocus getter."""
        assert "get hasFocus()" in self.src

    def test_webview_focus_flag_exists(self) -> None:
        """SorcarSidebarView has _webviewHasFocus property."""
        assert "_webviewHasFocus" in self.src

    def test_handles_webview_focus_changed(self) -> None:
        """_handleMessage handles 'webviewFocusChanged' messages."""
        assert "'webviewFocusChanged'" in self.src

    def test_focus_changed_updates_flag(self) -> None:
        """webviewFocusChanged handler updates _webviewHasFocus."""
        start = self.src.index("'webviewFocusChanged'")
        # Look at the next few lines after the case label
        snippet = self.src[start : start + 200]
        assert "_webviewHasFocus" in snippet


class TestToggleFocusMainJs:
    """Verify main.js sends focus/blur events to the extension."""

    src: str

    @classmethod
    def setup_class(cls) -> None:
        cls.src = (_VSCODE_DIR / "media" / "main.js").read_text()

    def test_window_focus_listener_exists(self) -> None:
        """main.js listens for window 'focus' events."""
        assert "window.addEventListener('focus'" in self.src

    def test_window_blur_listener_exists(self) -> None:
        """main.js listens for window 'blur' events."""
        assert "window.addEventListener('blur'" in self.src

    def test_posts_webview_focus_changed_on_focus(self) -> None:
        """Focus listener posts webviewFocusChanged with focused: true."""
        start = self.src.index("window.addEventListener('focus'")
        snippet = self.src[start : start + 200]
        assert "webviewFocusChanged" in snippet
        assert "focused: true" in snippet

    def test_posts_webview_focus_changed_on_blur(self) -> None:
        """Blur listener posts webviewFocusChanged with focused: false."""
        start = self.src.index("window.addEventListener('blur'")
        snippet = self.src[start : start + 200]
        assert "webviewFocusChanged" in snippet
        assert "focused: false" in snippet


class TestToggleFocusTypes:
    """Verify types.ts includes webviewFocusChanged message type."""

    src: str

    @classmethod
    def setup_class(cls) -> None:
        cls.src = (_VSCODE_DIR / "src" / "types.ts").read_text()

    def test_from_webview_includes_focus_changed(self) -> None:
        """FromWebviewMessage union includes webviewFocusChanged."""
        assert "webviewFocusChanged" in self.src
