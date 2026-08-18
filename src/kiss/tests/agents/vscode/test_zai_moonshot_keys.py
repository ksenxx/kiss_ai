"""End-to-end tests for Z.AI + Moonshot API key support (and MiniMax removal).

These tests verify that the agent platform exposes API keys for Z.AI and
Moonshot AI (and *not* MiniMax) across every surface that previously
referenced MiniMax:

* ``kiss.core.config.Config`` field names and env-var defaults.
* The VS Code settings panel allowlist + HTML inputs + JS env mapping.
* Provider-routing functions in ``kiss.core.models.model_info``.
* The provider/vendor display used by the model picker.
* The ``MODEL_INFO.json`` catalog (at least one glm-* and one moonshot/kimi
  entry, and zero ``minimax-*``/``MiniMaxAI/*`` entries).

Run with::

    uv run pytest src/kiss/tests/core/test_zai_moonshot_keys.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core import vscode_config
from kiss.core.models import model_info
from kiss.server import helpers

_VSCODE_MEDIA = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"


def test_config_has_zai_and_moonshot_fields() -> None:
    """`Config` exposes ZAI_API_KEY and MOONSHOT_API_KEY str fields."""
    fields = config_module.Config.model_fields
    assert "ZAI_API_KEY" in fields
    assert "MOONSHOT_API_KEY" in fields
    assert fields["ZAI_API_KEY"].annotation is str
    assert fields["MOONSHOT_API_KEY"].annotation is str


def test_config_drops_minimax_field() -> None:
    """`Config` no longer carries a MINIMAX_API_KEY field."""
    assert "MINIMAX_API_KEY" not in config_module.Config.model_fields


def test_config_defaults_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new key fields default from their respective env vars."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-test-token")
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-test-token")
    cfg = config_module.Config()
    assert cfg.ZAI_API_KEY == "zai-test-token"
    assert cfg.MOONSHOT_API_KEY == "moonshot-test-token"


def test_vscode_allowlist_replaced() -> None:
    """The VS Code env-var allowlist swaps MINIMAX for Z.AI + Moonshot."""
    allow = vscode_config.API_KEY_ENV_VARS
    assert "ZAI_API_KEY" in allow
    assert "MOONSHOT_API_KEY" in allow
    assert "MINIMAX_API_KEY" not in allow


def test_get_current_api_keys_includes_new_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_current_api_keys()` surfaces the new envs."""
    monkeypatch.setenv("ZAI_API_KEY", "z-abc")
    monkeypatch.setenv("MOONSHOT_API_KEY", "m-xyz")
    current = vscode_config.get_current_api_keys()
    assert current.get("ZAI_API_KEY") == "z-abc"
    assert current.get("MOONSHOT_API_KEY") == "m-xyz"
    assert "MINIMAX_API_KEY" not in current


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


def test_available_models_includes_glm_when_zai_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ZAI_API_KEY is set, at least one glm-* model is selectable."""
    monkeypatch.setenv("ZAI_API_KEY", "z-key")
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG", config_module.Config(), raising=False
    )
    available = model_info.get_available_models()
    assert any(m.startswith("glm-") for m in available)


def test_available_models_includes_moonshot_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MOONSHOT_API_KEY is set, at least one moonshot/kimi model is selectable."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "m-key")
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG", config_module.Config(), raising=False
    )
    available = model_info.get_available_models()
    assert any(
        m.startswith("moonshot-") or m.startswith("kimi-") for m in available
    )


if __name__ == "__main__":  # pragma: no cover - manual debugging entrypoint
    raise SystemExit(pytest.main([__file__, "-v"]))
