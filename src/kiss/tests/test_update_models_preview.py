"""Tests for preview model handling in update_models.py compute_changes."""

from kiss.scripts.update_models import compute_changes


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
    _, new_models = compute_changes(current, openrouter, {}, {}, {})
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
    _, new_models = compute_changes(current, openrouter, {}, {}, {})
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
    _, new_models = compute_changes(current, {}, together, {}, {})
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
    _, new_models = compute_changes(current, {}, together, {}, {})
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
    _, new_models = compute_changes(current, {}, {}, gemini, {})
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
    updates, new_models = compute_changes(current, openrouter, {}, {}, {})
    new_names = [m["name"] for m in new_models]
    assert "openrouter/google/gemini-2.5-flash" not in new_names
    assert len(updates) == 0


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
    _, new_models = compute_changes(current, openrouter, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/cool-model-preview" not in names
