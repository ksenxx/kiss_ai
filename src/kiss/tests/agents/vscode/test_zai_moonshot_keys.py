"""End-to-end tests for Z.AI + Moonshot API key support (and MiniMax removal).

These tests verify the surfaces that read real files or route through
``kiss.server``:

* The provider/vendor display used by the model picker
  (``kiss.server.helpers.model_vendor``).
* The VS Code settings panel HTML inputs and JS env mapping
  (``agents/vscode/media``).

The ``kiss.core`` config/allowlist/availability tests moved to
``tests/core/test_zai_moonshot_keys.py``; the provider-routing and
catalog tests live in ``tests/core/models/test_zai_moonshot_keys.py``.

Run with::

    uv run pytest src/kiss/tests/agents/vscode/test_zai_moonshot_keys.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiss.server import helpers

_VSCODE_MEDIA = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"


def test_model_vendor_zai_and_moonshot() -> None:
    """`model_vendor` routes glm-* to Z.AI and kimi-*/moonshot-* to Moonshot."""
    assert helpers.model_vendor("glm-4.6")[0] == "Z.AI"
    assert helpers.model_vendor("kimi-k2.6")[0] == "Moonshot"
    assert helpers.model_vendor("moonshot-v1-32k")[0] == "Moonshot"
    assert helpers.model_vendor("minimax-m2.5")[0] != "MiniMax"


def test_settings_panel_html_has_new_inputs() -> None:
    html = (_VSCODE_MEDIA / "chat.html").read_text()
    assert 'id="cfg-key-ZAI_API_KEY"' in html
    assert 'id="cfg-key-MOONSHOT_API_KEY"' in html
    assert "MINIMAX_API_KEY" not in html
    assert re.search(r"Z\.?AI API Key", html, flags=re.IGNORECASE)
    assert re.search(r"Moonshot API Key", html, flags=re.IGNORECASE)


def test_settings_panel_js_registers_new_keys() -> None:
    js = (_VSCODE_MEDIA / "main.js").read_text()
    assert "'cfg-key-ZAI_API_KEY'" in js
    assert "'cfg-key-MOONSHOT_API_KEY'" in js
    assert "ZAI_API_KEY" in js
    assert "MOONSHOT_API_KEY" in js
    assert "MINIMAX_API_KEY" not in js
    assert "minimax_api_key" not in js


if __name__ == "__main__":  # pragma: no cover - manual debugging entrypoint
    raise SystemExit(pytest.main([__file__, "-v"]))
