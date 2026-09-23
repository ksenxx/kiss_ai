# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A new OpenAI model unknown to OpenRouter must never land with context 0.

The 2026-09-22 catalog refresh added ``gpt-6-luna`` / ``gpt-6-sol`` (and
their thinking aliases) with ``"context_length": 0`` because
``compute_changes`` defaulted new OpenAI models to ``0`` when OpenRouter had
no listing yet.  ``KISSAgent._handoff_context_tokens`` then evaluates to
``0.0`` and the agent raises ``ContextWindowExceededError`` on its very first
step.  The Anthropic path already fell back to 200000 and the codex path to
400000; the OpenAI path now falls back to the same 400000 default, and the
openrouter-xref backfill pass repairs existing zero-context entries.

These tests drive the production ``compute_changes`` end-to-end with
synthetic vendor payloads, exactly like the live script does.
"""

from kiss.scripts.update_models import (
    _OPENAI_DEFAULT_CONTEXT_LENGTH,
    compute_changes,
)


def _new_model(new_models: list[dict], name: str) -> dict:
    return next(m for m in new_models if m["name"] == name)


def test_new_openai_model_without_openrouter_listing_gets_default_context() -> None:
    """No OpenRouter twin: context falls back to the OpenAI default, pricing stays unknown."""
    openai = {"gpt-6-luna": {"source": "openai"}}
    _, new_models = compute_changes({}, {}, {}, {}, {}, openai)
    entry = _new_model(new_models, "gpt-6-luna")
    assert entry["context_length"] == _OPENAI_DEFAULT_CONTEXT_LENGTH
    assert entry["context_length"] > 0
    assert entry["needs_pricing"] is True


def test_new_openai_model_with_openrouter_listing_uses_its_context() -> None:
    """An OpenRouter twin's context length wins over the default."""
    openrouter = {
        "openrouter/openai/gpt-6-luna": {
            "context_length": 262144,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 25.0,
            "source": "openrouter",
        }
    }
    openai = {"gpt-6-luna": {"source": "openai"}}
    _, new_models = compute_changes({}, openrouter, {}, {}, {}, openai)
    entry = _new_model(new_models, "gpt-6-luna")
    assert entry["context_length"] == 262144
    assert entry["input_price_per_1M"] == 5.0
    assert entry["needs_pricing"] is False


def test_existing_zero_context_openai_entry_is_repaired_without_openrouter() -> None:
    """A catalog entry already written with context 0 gets the default on the next refresh."""
    current = {
        "gpt-6-luna": {
            "context_length": 0,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": False,
            "emb": False,
            "gen": True,
        }
    }
    openai = {"gpt-6-luna": {"source": "openai"}}
    updates, new_models = compute_changes(current, {}, {}, {}, {}, openai)
    assert new_models == []
    upd = next(u for u in updates if u["name"] == "gpt-6-luna")
    assert upd["source"] == "openrouter-xref"
    assert upd["changes"] == {"context_length": _OPENAI_DEFAULT_CONTEXT_LENGTH}


def test_existing_zero_context_anthropic_entry_is_repaired_without_openrouter() -> None:
    """The repair uses the per-vendor default: Anthropic entries get 200000."""
    current = {
        "claude-nova-6": {
            "context_length": 0,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    anthropic = {"claude-nova-6": {"source": "anthropic"}}
    updates, _ = compute_changes(current, {}, {}, {}, anthropic, {})
    upd = next(u for u in updates if u["name"] == "claude-nova-6")
    assert upd["changes"] == {"context_length": 200000}


def test_vendor_reported_context_is_not_overwritten_by_the_default() -> None:
    """Gemini's own ``inputTokenLimit`` for a zero-context entry beats the vendor default."""
    current = {
        "gemini-example": {
            "context_length": 0,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    gemini = {"gemini-example": {"context_length": 128000, "source": "gemini"}}
    updates, _ = compute_changes(current, {}, {}, gemini, {}, {})
    matching = [u for u in updates if u["name"] == "gemini-example"]
    assert matching == [
        {"name": "gemini-example", "changes": {"context_length": 128000}, "source": "gemini"}
    ]


def test_existing_zero_context_entry_prefers_openrouter_context() -> None:
    """When OpenRouter lists the model, its real context length beats the default."""
    current = {
        "gpt-6-luna": {
            "context_length": 0,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": False,
            "emb": False,
            "gen": True,
        }
    }
    openrouter = {
        "openrouter/openai/gpt-6-luna": {
            "context_length": 262144,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 25.0,
            "source": "openrouter",
        }
    }
    openai = {"gpt-6-luna": {"source": "openai"}}
    updates, _ = compute_changes(current, openrouter, {}, {}, {}, openai)
    upd = next(u for u in updates if u["name"] == "gpt-6-luna")
    assert upd["changes"]["context_length"] == 262144
    assert upd["changes"]["input_price_per_1M"] == 5.0
    assert upd["changes"]["output_price_per_1M"] == 25.0


def test_priced_entry_with_context_is_left_alone() -> None:
    """A complete entry is not touched by the backfill pass."""
    current = {
        "gpt-6-luna": {
            "context_length": 400000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 25.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    openai = {"gpt-6-luna": {"source": "openai"}}
    updates, new_models = compute_changes(current, {}, {}, {}, {}, openai)
    assert new_models == []
    assert [u for u in updates if u["name"] == "gpt-6-luna"] == []


def test_zero_priced_entry_with_context_but_no_openrouter_twin_is_left_alone() -> None:
    """Pricing cannot be invented: with no OpenRouter twin and a valid context, nothing changes."""
    current = {
        "gpt-6-luna": {
            "context_length": 400000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    openai = {"gpt-6-luna": {"source": "openai"}}
    updates, _ = compute_changes(current, {}, {}, {}, {}, openai)
    assert [u for u in updates if u["name"] == "gpt-6-luna"] == []
