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

import json
import re
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core import vscode_config
from kiss.core.models import model_info
from kiss.server import helpers

# ---------------------------------------------------------------------------
# config.Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# VS Code config allowlist
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# helpers.model_vendor (model picker grouping)
# ---------------------------------------------------------------------------

def test_model_vendor_zai_and_moonshot() -> None:
    """`model_vendor` routes glm-* to Z.AI and kimi-*/moonshot-* to Moonshot."""
    assert helpers.model_vendor("glm-4.6")[0] == "Z.AI"
    assert helpers.model_vendor("kimi-k2.6")[0] == "Moonshot"
    assert helpers.model_vendor("moonshot-v1-32k")[0] == "Moonshot"
    # MiniMax routing must be gone — minimax-* now falls through to the
    # default "Together AI" bucket (no dedicated MiniMax branch).
    assert helpers.model_vendor("minimax-m2.5")[0] != "MiniMax"


# ---------------------------------------------------------------------------
# model_info routing
# ---------------------------------------------------------------------------

def test_get_model_provider_zai_and_moonshot() -> None:
    """`get_model_provider` recognizes Z.AI and Moonshot prefixes."""
    assert model_info.get_model_provider("glm-4.6") == "Z.AI"
    assert model_info.get_model_provider("kimi-k2.6") == "Moonshot"
    assert model_info.get_model_provider("moonshot-v1-32k") == "Moonshot"


def test_get_model_provider_no_minimax_branch() -> None:
    """The dedicated MiniMax provider label is no longer returned."""
    # The function should never return "MiniMax" for any of the historical
    # MiniMax names — they either resolve elsewhere or to "Unknown".
    for name in ("minimax-m2.5", "minimax-m2.5-lightning", "minimax-m1"):
        assert model_info.get_model_provider(name) != "MiniMax"


def test_model_listing_advertises_new_providers() -> None:
    """The model listing reports the new providers' configuration status."""
    listing = model_info.get_generation_model_listing()
    providers = {entry[1] for entry in listing}
    # At least one model from each of the new providers must be present.
    assert "Z.AI" in providers
    assert "Moonshot" in providers
    assert "MiniMax" not in providers


# ---------------------------------------------------------------------------
# MODEL_INFO.json catalog
# ---------------------------------------------------------------------------

_MODEL_INFO_PATH = (
    Path(model_info.__file__).resolve().parent / "MODEL_INFO.json"
)


def test_model_info_json_has_zai_and_moonshot_entries() -> None:
    raw = json.loads(_MODEL_INFO_PATH.read_text())
    glm_models = [k for k in raw if k.startswith("glm-")]
    moonshot_models = [
        k for k in raw if k.startswith("moonshot-") or k.startswith("kimi-")
    ]
    assert glm_models, "Expected at least one glm-* entry in MODEL_INFO.json"
    assert moonshot_models, (
        "Expected at least one moonshot-*/kimi-* entry in MODEL_INFO.json"
    )


def test_model_info_json_has_no_minimax_entries() -> None:
    raw = json.loads(_MODEL_INFO_PATH.read_text())
    bad = [
        k
        for k in raw
        if k.startswith("minimax-")
        or k.startswith("MiniMaxAI/")
        or k.startswith("openrouter/minimax/")
    ]
    assert not bad, f"MiniMax entries remain in MODEL_INFO.json: {bad!r}"


# ---------------------------------------------------------------------------
# Settings panel HTML + JS
# ---------------------------------------------------------------------------

_VSCODE_MEDIA = Path(__file__).resolve().parents[2] / "agents" / "vscode" / "media"


def test_settings_panel_html_has_new_inputs() -> None:
    html = (_VSCODE_MEDIA / "chat.html").read_text()
    assert 'id="cfg-key-ZAI_API_KEY"' in html
    assert 'id="cfg-key-MOONSHOT_API_KEY"' in html
    assert "MINIMAX_API_KEY" not in html
    # Visible labels must mention the new providers.
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


# ---------------------------------------------------------------------------
# get_available_models honors the new envs
# ---------------------------------------------------------------------------

def test_available_models_includes_glm_when_zai_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ZAI_API_KEY is set, at least one glm-* model is selectable."""
    monkeypatch.setenv("ZAI_API_KEY", "z-key")
    # Refresh the singleton config so the env change is observed.
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
