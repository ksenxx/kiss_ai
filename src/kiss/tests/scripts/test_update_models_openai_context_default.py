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

import json
from pathlib import Path

import pytest

from kiss.scripts import update_models
from kiss.scripts.update_models import (
    _DEFAULT_CONTEXT_LENGTH_BY_SOURCE,
    _OPENAI_DEFAULT_CONTEXT_LENGTH,
    compute_changes,
)


def _new_model(new_models: list[dict], name: str) -> dict:
    return next(m for m in new_models if m["name"] == name)


def test_new_openai_model_without_openrouter_listing_gets_default_context() -> None:
    """No OpenRouter twin: context falls back to the OpenAI default, pricing stays unknown."""
    openai = {"gpt-6-luna": {"source": "openai"}}
    _, new_models = compute_changes({}, {}, {}, {}, openai)
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
    _, new_models = compute_changes({}, openrouter, {}, {}, openai)
    entry = _new_model(new_models, "gpt-6-luna")
    assert entry["context_length"] == 262144
    assert entry["input_price_per_1M"] == 5.0
    assert entry["needs_pricing"] is False


def test_new_gemini_model_without_a_reported_limit_never_lands_with_context_zero() -> None:
    """Audit SC1: a new Gemini model listed without ``inputTokenLimit`` was written with context
    0 until the next refresh repaired it; it gets the OpenRouter twin's window, else the vendor
    default, at once.  A limit the listing does report still wins."""
    gemini = {
        "gemini-9-flash": {"context_length": 0, "source": "gemini"},
        "gemini-9-pro": {"context_length": 0, "source": "gemini"},
        "gemini-9-lite": {"context_length": 64000, "source": "gemini"},
    }
    openrouter = {
        f"openrouter/google/gemini-9-{tier}": {
            "context_length": 262144,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 12.0,
            "source": "openrouter",
        }
        for tier in ("pro", "lite")
    }
    _, new_models = compute_changes({}, openrouter, gemini, {}, {})
    assert _new_model(new_models, "gemini-9-flash")["context_length"] == (
        _DEFAULT_CONTEXT_LENGTH_BY_SOURCE["gemini"]
    )
    assert _new_model(new_models, "gemini-9-pro")["context_length"] == 262144
    assert _new_model(new_models, "gemini-9-lite")["context_length"] == 64000


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
    updates, new_models = compute_changes(current, {}, {}, {}, openai)
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
    updates, _ = compute_changes(current, {}, {}, anthropic, {})
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
    updates, _ = compute_changes(current, {}, gemini, {}, {})
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
    updates, _ = compute_changes(current, openrouter, {}, {}, openai)
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
    updates, new_models = compute_changes(current, {}, {}, {}, openai)
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
    updates, _ = compute_changes(current, {}, {}, {}, openai)
    assert [u for u in updates if u["name"] == "gpt-6-luna"] == []


def test_zero_context_generated_alias_takes_the_repaired_context_of_its_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit X3-2: the repair looked a generated alias up by its own name, which OpenRouter
    does not list, and its vendor default replaced the twin's window the alias copies from its
    base.  A real model whose name merely ends in ``-high`` is still repaired by that name."""
    row = {"context_length": 0, "input_price_per_1M": 5.0, "output_price_per_1M": 25.0}
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(
        json.dumps(
            {
                "gpt-6-luna": {**row, "thinking": "high"},
                "gpt-6-luna-high": {**row, "thinking": "high", "alias_of": "gpt-6-luna"},
                "gpt-6-terra-high": row,
            }
        )
    )
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", catalog)
    twin = {**row, "context_length": 262144, "source": "openrouter"}
    current = update_models.get_current_model_info()
    updates, _ = compute_changes(current, {"openrouter/openai/gpt-6-luna": twin}, {}, {}, {})
    update_models.apply_updates_to_file(updates, [], [], current)
    written = json.loads(catalog.read_text())
    assert {name: entry["context_length"] for name, entry in written.items()} == {
        "gpt-6-luna": 262144,
        "gpt-6-luna-low": 262144,
        "gpt-6-luna-medium": 262144,
        "gpt-6-luna-high": 262144,
        "gpt-6-terra-high": _OPENAI_DEFAULT_CONTEXT_LENGTH,
    }
