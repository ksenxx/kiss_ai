# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for preview model handling in update_models.py compute_changes."""

from kiss.scripts.update_models import (
    _add_codex_candidates,
    compute_changes,
    find_deprecated_models,
)


def _make_current() -> dict[str, dict]:
    """Minimal current MODEL_INFO for testing."""
    return {
        "openrouter/google/gemini-2.5-flash": {
            "context_length": 1048576,
            "input_price_per_1M": 0.30,
            "output_price_per_1M": 2.50,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }


def test_openrouter_preview_with_zero_pricing_is_added():
    """Free preview models from OpenRouter should be added."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/cool-model-preview": {
            "context_length": 128000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/cool-model-preview" in names
    model = next(m for m in new_models if m["name"] == "openrouter/acme/cool-model-preview")
    assert model["needs_pricing"] is True


def test_openrouter_variant_endpoints_still_skipped():
    """Variant endpoints (:free, :thinking) should still be skipped."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/model-preview:free": {
            "context_length": 128000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/model-preview:free" not in names


def test_together_preview_with_zero_pricing_is_added():
    """Free preview models from Together should be added."""
    current = _make_current()
    together = {
        "meta-llama/some-model-preview": {
            "context_length": 131072,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "together",
            "type": "chat",
            "is_embedding": False,
        },
    }
    _, new_models = compute_changes(current, {}, together, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "meta-llama/some-model-preview" in names
    model = next(m for m in new_models if m["name"] == "meta-llama/some-model-preview")
    assert model["needs_pricing"] is True


def test_together_non_preview_zero_pricing_is_not_added():
    """Non-preview Together models with zero pricing should be filtered out."""
    current = _make_current()
    together = {
        "meta-llama/some-free-model": {
            "context_length": 131072,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "together",
            "type": "chat",
            "is_embedding": False,
        },
    }
    _, new_models = compute_changes(current, {}, together, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "meta-llama/some-free-model" not in names


def test_gemini_preview_model_is_added():
    """Preview models from Gemini should be added (they always have needs_pricing)."""
    current = _make_current()
    gemini = {
        "gemini-99-flash-preview": {
            "context_length": 1048576,
            "source": "gemini",
            "is_embedding": False,
            "is_generation": True,
        },
    }
    _, new_models = compute_changes(current, {}, {}, gemini, {}, {})
    names = [m["name"] for m in new_models]
    assert "gemini-99-flash-preview" in names
    model = next(m for m in new_models if m["name"] == "gemini-99-flash-preview")
    assert model["needs_pricing"] is True


def test_existing_model_not_duplicated():
    """Models already in current should not be added as new."""
    current = _make_current()
    openrouter = {
        "openrouter/google/gemini-2.5-flash": {
            "context_length": 1048576,
            "input_price_per_1M": 0.30,
            "output_price_per_1M": 2.50,
            "source": "openrouter",
        },
    }
    updates, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    new_names = [m["name"] for m in new_models]
    assert "openrouter/google/gemini-2.5-flash" not in new_names
    assert len(updates) == 0


def test_codex_candidates_added_from_codex_models_json():
    """Only models in the Codex CLI models.json get codex/ entries."""
    current = _make_current()
    codex_slugs = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini"}
    _, new_models = compute_changes(
        current, {}, {}, {}, {}, {}, codex_slugs=codex_slugs
    )
    names = {m["name"] for m in new_models}
    for expected in (
        "codex/gpt-5.5",
        "codex/gpt-5.4",
        "codex/gpt-5.4-mini",
    ):
        assert expected in names
    # Unsupported models like gpt-5.5-pro should NOT appear
    assert "codex/gpt-5.5-pro" not in names
    for nm in new_models:
        if nm["name"].startswith("codex/"):
            assert nm["input_price_per_1M"] == 0.0
            assert nm["output_price_per_1M"] == 0.0
            assert nm["source"] == "codex"
            assert nm["needs_pricing"] is False
            assert nm["gen"] is True
            assert nm["fc"] is True
            assert nm["emb"] is False
            assert nm["context_length"] >= 400000


def test_codex_candidate_uses_openrouter_context_when_available():
    """Context length for codex/<name> should come from OpenRouter when present."""
    current: dict[str, dict] = {}
    codex_slugs = {"gpt-5.5"}
    openrouter = {
        "openrouter/openai/gpt-5.5": {
            "context_length": 1050000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        },
    }
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, openrouter, new_models)
    [entry] = [m for m in new_models if m["name"] == "codex/gpt-5.5"]
    assert entry["context_length"] == 1050000


def test_codex_candidate_only_adds_supported_slugs():
    """Only slugs from the Codex CLI models.json should get entries."""
    current: dict[str, dict] = {}
    codex_slugs = {"gpt-5.5", "gpt-5.4"}
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert names == {"codex/gpt-5.5", "codex/gpt-5.4"}


def test_codex_candidate_skips_unsupported_models():
    """Models not in the Codex models.json must not get codex/ entries."""
    current: dict[str, dict] = {}
    # Empty slug set means no codex models should be added
    codex_slugs: set[str] = set()
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    assert new_models == []


def test_codex_candidate_not_added_when_already_present():
    """codex/<name> already in current MODEL_INFO must not be re-added."""
    current = {
        "codex/gpt-5.4": {
            "context_length": 400000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    codex_slugs = {"gpt-5.4", "gpt-5.5"}
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert "codex/gpt-5.4" not in names
    assert "codex/gpt-5.5" in names


def test_openrouter_preview_zero_context_not_added():
    """Preview models with zero context length should not be added."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/cool-model-preview": {
            "context_length": 0,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/cool-model-preview" not in names


def test_find_deprecated_codex_entries_with_slugs():
    """codex/* models not in the Codex CLI models.json should be deprecated.

    codex/default is always kept. When codex_slugs is provided, only models
    whose slug appears in the set are kept; all others are deprecated.
    When codex_slugs is None (fetch failed), no codex models are deprecated.
    """
    current = {
        "codex/default": {"source": "codex"},
        "codex/gpt-5.5": {"source": "codex"},
        "codex/gpt-5.5-pro": {"source": "codex"},
        "codex/gpt-5.4": {"source": "codex"},
        "gpt-foo-removed": {"source": "openai"},
    }
    codex_slugs = {"gpt-5.5", "gpt-5.4"}
    openai = {"gpt-5": {"source": "openai"}, "gpt-5-2025-08-07": {"source": "openai"}}
    deprecated = find_deprecated_models(
        current, {}, {}, {}, openai, codex_slugs=codex_slugs
    )
    names = {d["name"] for d in deprecated}
    # codex/default is always kept
    assert "codex/default" not in names
    # Supported slugs are kept
    assert "codex/gpt-5.5" not in names
    assert "codex/gpt-5.4" not in names
    # Unsupported slug is deprecated
    assert "codex/gpt-5.5-pro" in names


def test_find_deprecated_codex_entries_without_slugs():
    """When codex_slugs is None (fetch failed), no codex models are deprecated."""
    current = {
        "codex/default": {"source": "codex"},
        "codex/gpt-5.5-pro": {"source": "codex"},
    }
    deprecated = find_deprecated_models(
        current, {}, {}, {}, {}, codex_slugs=None
    )
    names = {d["name"] for d in deprecated}
    assert "codex/default" not in names
    assert "codex/gpt-5.5-pro" not in names
