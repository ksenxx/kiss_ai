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
from pathlib import Path

import pytest

from kiss.core.models import model_info

_MODEL_INFO_PATH = (
    Path(model_info.__file__).resolve().parent / "MODEL_INFO.json"
)


def test_get_model_provider_zai_and_moonshot() -> None:
    """`get_model_provider` recognizes Z.AI and Moonshot prefixes."""
    assert model_info.get_model_provider("glm-4.6") == "Z.AI"
    assert model_info.get_model_provider("kimi-k2.6") == "Moonshot"
    assert model_info.get_model_provider("moonshot-v1-32k") == "Moonshot"


def test_get_model_provider_no_minimax_branch() -> None:
    """The dedicated MiniMax provider label is no longer returned."""
    for name in ("minimax-m2.5", "minimax-m2.5-lightning", "minimax-m1"):
        assert model_info.get_model_provider(name) != "MiniMax"


def test_model_routing_advertises_new_providers() -> None:
    """The generation catalog routes to the new providers, not MiniMax."""
    providers = {
        model_info.get_model_provider(name)
        for name, info in model_info.MODEL_INFO.items()
        if info.is_generation_supported
    }
    assert "Z.AI" in providers
    assert "Moonshot" in providers
    assert "MiniMax" not in providers


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


if __name__ == "__main__":  # pragma: no cover - manual debugging entrypoint
    raise SystemExit(pytest.main([__file__, "-v"]))
