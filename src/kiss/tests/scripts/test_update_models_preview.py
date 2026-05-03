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


def test_codex_candidates_added_for_gpt5_plus_models():
    """Every GPT-5+ model from OpenAI should produce a codex/ counterpart."""
    current = _make_current()
    openai = {
        "gpt-5": {"source": "openai"},
        "gpt-5-codex": {"source": "openai"},
        "gpt-5.1-codex-max": {"source": "openai"},
        "gpt-5.5": {"source": "openai"},
        "gpt-6": {"source": "openai"},
    }
    _, new_models = compute_changes(current, {}, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    for expected in (
        "codex/gpt-5",
        "codex/gpt-5-codex",
        "codex/gpt-5.1-codex-max",
        "codex/gpt-5.5",
        "codex/gpt-6",
    ):
        assert expected in names
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
    openai = {"gpt-5.5": {"source": "openai"}}
    openrouter = {
        "openrouter/openai/gpt-5.5": {
            "context_length": 1050000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        },
    }
    new_models: list[dict] = []
    _add_codex_candidates(openai, current, openrouter, new_models)
    [entry] = [m for m in new_models if m["name"] == "codex/gpt-5.5"]
    assert entry["context_length"] == 1050000


def test_codex_candidate_skips_pre_gpt5_models():
    """Old GPT families must not get codex/ entries."""
    current: dict[str, dict] = {}
    openai = {
        "gpt-3.5-turbo": {"source": "openai"},
        "gpt-4": {"source": "openai"},
        "gpt-4o": {"source": "openai"},
        "gpt-4.1-mini": {"source": "openai"},
        "gpt-4-turbo-2024-04-09": {"source": "openai"},
    }
    new_models: list[dict] = []
    _add_codex_candidates(openai, current, {}, new_models)
    assert new_models == []


def test_codex_candidate_skips_dated_snapshots_and_multimodal():
    """Date-suffixed snapshots and image/audio variants must be skipped."""
    current: dict[str, dict] = {}
    openai = {
        "gpt-5-2025-08-07": {"source": "openai"},
        "gpt-5.1-2025-11-13": {"source": "openai"},
        "gpt-5-image": {"source": "openai"},
        "gpt-5-audio-preview": {"source": "openai"},
        "gpt-5-search-preview": {"source": "openai"},
        "gpt-5-mini-tts": {"source": "openai"},
    }
    new_models: list[dict] = []
    _add_codex_candidates(openai, current, {}, new_models)
    assert new_models == []


def test_codex_candidate_not_added_when_already_present():
    """codex/<name> already in current MODEL_INFO must not be re-added."""
    current = {
        "codex/gpt-5": {
            "context_length": 400000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    openai = {"gpt-5": {"source": "openai"}, "gpt-5.5": {"source": "openai"}}
    new_models: list[dict] = []
    _add_codex_candidates(openai, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert "codex/gpt-5" not in names
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


def test_find_deprecated_skips_codex_entries():
    """codex/* entries must NOT be flagged deprecated based on the OpenAI API.

    They are aliases accepted by the codex CLI's -m flag, billed via the
    ChatGPT subscription, and intentionally absent from OpenAI's REST model
    listing. They are managed separately by _add_codex_candidates.
    """
    current = {
        "codex/default": {"source": "codex"},
        "codex/gpt-5": {"source": "codex"},
        "codex/gpt-5-codex": {"source": "codex"},
        "codex/gpt-5.5-pro": {"source": "codex"},
        "gpt-foo-removed": {"source": "openai"},
    }
    openai = {"gpt-5": {"source": "openai"}, "gpt-5-2025-08-07": {"source": "openai"}}
    deprecated = find_deprecated_models(current, {}, {}, {}, openai)
    names = {d["name"] for d in deprecated}
    assert "codex/default" not in names
    assert "codex/gpt-5" not in names
    assert "codex/gpt-5-codex" not in names
    assert "codex/gpt-5.5-pro" not in names
    assert "gpt-foo-removed" in names
