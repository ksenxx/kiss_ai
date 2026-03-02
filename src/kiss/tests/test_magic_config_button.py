"""Tests that the magic commit message button was removed from chatbot UI."""

import unittest

from kiss.agents.sorcar.chatbot_ui import _build_html


class TestMagicButtonRemoved(unittest.TestCase):
    def test_html_does_not_contain_magic_button(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert 'id="magic-btn"' not in html

    def test_html_no_magic_button_title(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert 'title="Generate commit message"' not in html

    def test_html_no_magic_btn_css(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "#magic-btn{" not in html
        assert "#magic-btn " not in html

    def test_html_no_magic_btn_js_handler(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "magicBtn" not in html

    def test_html_no_magic_spin_animation(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "magicSpin" not in html
        assert "#magic-btn.loading" not in html

    def test_upload_button_still_present(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert 'id="upload-btn"' in html

    def test_generate_commit_message_not_in_chatbot_js(self) -> None:
        html = _build_html("Test", "", "/tmp")
        js_start = html.index("<script>")
        js_end = html.index("</script>")
        js_section = html[js_start:js_end]
        assert "generate-commit-message" not in js_section


if __name__ == "__main__":
    unittest.main()
