# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: API-key inputs in the settings panel are secret.

The settings panel previously rendered the API-key fields as plain
``type="text"`` inputs, exposing saved keys to shoulder-surfing.  The
fields must now be masked (``type="password"``) by default, with an
eye-toggle button — identical to the remote-password field — that
reveals the key on click and masks it again on a second click.

These tests pin the exact HTML delivered to browser clients by
``_build_html()`` and the toggle wiring in the served ``main.js`` so a
future refactor that reverts an API-key field to a plain text input
(or drops its eye toggle) fails loudly.
"""

from __future__ import annotations

import unittest

from kiss.agents.vscode.web_server import MEDIA_DIR, _build_html

#: Every settings-panel input that holds an API key and must be masked.
API_KEY_INPUT_IDS = [
    "cfg-key-GEMINI_API_KEY",
    "cfg-key-OPENAI_API_KEY",
    "cfg-key-ANTHROPIC_API_KEY",
    "cfg-key-TOGETHER_API_KEY",
    "cfg-key-OPENROUTER_API_KEY",
    "cfg-key-ZAI_API_KEY",
    "cfg-key-MOONSHOT_API_KEY",
    "cfg-custom-api-key",
]


class TestApiKeyInputsMaskedInHtml(unittest.TestCase):
    """The served settings-panel HTML masks every API-key input."""

    def test_api_key_inputs_are_password_type(self) -> None:
        """Each API-key input is ``type="password"`` with autocomplete off."""
        html = _build_html()
        for input_id in API_KEY_INPUT_IDS:
            self.assertRegex(
                html,
                rf'<input type="password" autocomplete="off" id="{input_id}"',
                f"{input_id} must be masked by default so saved API keys "
                "are not readable over the user's shoulder.",
            )

    def test_no_api_key_input_is_plain_text(self) -> None:
        """No API-key input may regress to a visible text input."""
        html = _build_html()
        for input_id in API_KEY_INPUT_IDS:
            self.assertNotRegex(
                html,
                rf'<input type="text"[^>]*id="{input_id}"',
                f"{input_id} must not be a plain text input.",
            )

    def test_remote_password_toggle_present_as_clone_prototype(self) -> None:
        """The remote-password eye toggle (cloned for API keys) exists.

        ``setupSecretInput`` in main.js clones the eye/eye-off SVG
        button from ``#cfg-remote-password-toggle`` so the icon markup
        lives in exactly one place.  If that prototype disappears the
        API-key toggles silently stop rendering.
        """
        html = _build_html()
        self.assertIn('id="cfg-remote-password-toggle"', html)
        self.assertIn('class="icon-eye"', html)
        self.assertIn('class="icon-eye-off"', html)


class TestEyeToggleWiringInMainJs(unittest.TestCase):
    """The served main.js wires a show/hide eye toggle per API key."""

    def test_every_api_key_input_gets_secret_toggle(self) -> None:
        """main.js calls setupSecretInput for every API-key input id."""
        js = (MEDIA_DIR / "main.js").read_text(encoding="utf-8")
        self.assertIn(
            "function setupSecretInput",
            js,
            "main.js must define setupSecretInput, which masks an input "
            "and attaches the eye toggle cloned from the remote-password "
            "toggle.",
        )
        self.assertIn(
            "].forEach(setupSecretInput)",
            js,
            "setupEventListeners must wire setupSecretInput over the "
            "API-key input ids.",
        )
        for input_id in API_KEY_INPUT_IDS:
            self.assertIn(
                f"'{input_id}'",
                js,
                f"{input_id} must be in the setupSecretInput wiring list.",
            )

    def test_toggle_flips_between_password_and_text(self) -> None:
        """The shared toggle handler flips input.type on each click.

        This is the mechanism by which one click reveals the key and a
        second click masks it again.
        """
        js = (MEDIA_DIR / "main.js").read_text(encoding="utf-8")
        self.assertIn("const showing = inp.type === 'text';", js)
        self.assertIn("inp.type = showing ? 'password' : 'text';", js)


if __name__ == "__main__":
    unittest.main()
