"""Tests for the magic commit message button feature."""

import unittest

from kiss.agents.assistant.chatbot_ui import _build_html


class TestMagicButtonHTML(unittest.TestCase):
    def test_html_contains_magic_button(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert 'id="magic-btn"' in html

    def test_html_magic_button_has_title(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert 'title="Generate commit message"' in html

    def test_html_magic_button_has_svg_icon(self) -> None:
        html = _build_html("Test", "", "/tmp")
        start = html.index('id="magic-btn"')
        end = html.index("</button>", start)
        snippet = html[start:end]
        assert "<svg" in snippet
        assert "<path" in snippet

    def test_html_contains_magic_btn_css(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "#magic-btn{" in html or "#magic-btn " in html

    def test_html_contains_magic_btn_js_handler(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "generate-commit-message" in html

    def test_html_magic_button_before_upload_button(self) -> None:
        html = _build_html("Test", "", "/tmp")
        magic_pos = html.index('id="magic-btn"')
        upload_pos = html.index('id="upload-btn"')
        assert magic_pos < upload_pos

    def test_magic_btn_loading_animation_css(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "magicSpin" in html
        assert "#magic-btn.loading" in html

    def test_html_does_not_reference_old_config_message_in_js(self) -> None:
        html = _build_html("Test", "", "/tmp")
        js_start = html.index("<script>")
        js_end = html.index("</script>")
        js_section = html[js_start:js_end]
        assert "generate-config-message" not in js_section


if __name__ == "__main__":
    unittest.main()
