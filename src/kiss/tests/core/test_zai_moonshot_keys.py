"""End-to-end tests for Z.AI + Moonshot API key support (and MiniMax removal).

Split out of ``tests/agents/vscode/test_zai_moonshot_keys.py``: these
methods depend only on ``kiss.core`` (``config``, ``vscode_config``) and
``kiss.core.models.model_info``, so they belong in ``tests/core`` per the
placement invariants.  The vendor-display and settings-panel asset tests
remain in ``tests/agents/vscode``; the provider-routing and catalog tests
live in ``tests/core/models/test_zai_moonshot_keys.py``.

* ``kiss.core.config.Config`` field names and env-var defaults.
* The VS Code env-var allowlist and current-key surface in
  ``kiss.core.vscode_config``.
* Key-gated model availability in ``kiss.core.models.model_info``.

Run with::

    uv run pytest src/kiss/tests/core/test_zai_moonshot_keys.py -v
"""

from __future__ import annotations

import pytest

from kiss.core import config as config_module
from kiss.core import vscode_config
from kiss.core.models import model_info


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
